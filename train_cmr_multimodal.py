"""
train_cmr_multimodal.py
=======================
Multi-modal CMR CLIP pre-training.

Supports all 10 modalities:
    Cine, LGE, Mapping, Flow2d, Aorta,
    Perfusion, T1w, T2w, Tagging, T1rho

Key design choices
------------------
- Text is generated at runtime from mat_path (no 'text' column in CSV).
  Across 10 modalities the combination of modality+view+vendor+field gives
  sufficient uniqueness for contrastive pre-training.
- frame_map (plain dict) replaces group_df to prevent DataLoader memory leaks.
- Checkpoints are written locally then async-copied to NAS.
- Full-corpus N×N retrieval evaluation instead of noisy batch-level R@1.
- Soft labels for same-modality and same-view pairs in the CLIP loss.
- Modality-accuracy is reported as an additional proxy metric.
"""

import os
import re
import math
import random
import argparse
import tempfile
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Path metadata parser  (same logic as cmr_multimodal_dataset.py)
# ---------------------------------------------------------------------------

ALL_MODALITIES = [
    "Cine", "LGE", "Mapping", "Flow2d", "Aorta",
    "Perfusion", "T1w", "T2w", "Tagging", "T1rho",
]

MODALITY_TEXT = {
    "Cine":      "cine cardiac MRI",
    "LGE":       "late gadolinium enhancement cardiac MRI",
    "Mapping":   "cardiac mapping MRI",
    "Flow2d":    "2D flow cardiac MRI",
    "Aorta":     "aorta MRI",
    "Perfusion": "cardiac perfusion MRI",
    "T1w":       "T1-weighted cardiac MRI",
    "T2w":       "T2-weighted cardiac MRI",
    "Tagging":   "cardiac tagging MRI",
    "T1rho":     "T1rho cardiac MRI",
}

VIEW_TEXT = {
    "cine_sax": "short-axis", "cine_lax": "long-axis",
    "cine_lax_2ch": "2-chamber", "cine_lax_3ch": "3-chamber",
    "cine_lax_4ch": "4-chamber", "cine_lvot": "LVOT",
    "cine_rvot": "RVOT", "cine_ot": "outflow tract",
    "lge_sax": "short-axis", "lge_lax": "long-axis",
    "lge_lax_2ch": "2-chamber", "lge_lax_3ch": "3-chamber",
    "lge_lax_4ch": "4-chamber",
    "T1map": "T1 mapping", "T1mappost": "post-contrast T1 mapping",
    "T2map": "T2 mapping", "T2smap": "T2-star mapping",
    "flow2d": "through-plane flow", "flow2d_inplane": "in-plane flow",
    "flow2d_M": "flow magnitude", "flow2d_P": "flow phase",
    "flow2d_throughplane_D": "through-plane flow magnitude",
    "flow2d_throughplane_M": "through-plane flow phase",
    "aorta_sag": "sagittal aorta", "aorta_tra": "transverse aorta",
    "perfusion": "myocardial perfusion",
    "T1w": "T1-weighted", "T2w": "T2-weighted",
    "tagging": "myocardial tagging", "T1rho": "T1rho",
    # short view names used in Cine npy dirs
    "sax": "short-axis", "lax": "long-axis",
    "2ch": "2-chamber", "3ch": "3-chamber", "4ch": "4-chamber",
    "lvot": "LVOT", "rvot": "RVOT",
}

VENDOR_FULL = {
    "siemens": "Siemens", "uih": "United Imaging",
    "philips": "Philips", "ge": "GE", "canon": "Canon",
}
FIELD_NORM = {
    "30t": "3.0T", "3t": "3.0T",
    "15t": "1.5T", "1t": "1.5T",
    "055t": "0.55T",
}
_VENDOR_PREFIXES = tuple(VENDOR_FULL.keys())
_FIELD_RE        = re.compile(r"\d+[.,]?\d*[Tt]", re.IGNORECASE)


def parse_mat_path(mat_path: str) -> dict:
    parts    = str(mat_path).replace("\\", "/").split("/")
    filename = os.path.splitext(os.path.basename(mat_path))[0]
    modality = next((p for p in parts if p in ALL_MODALITIES), "unknown")
    split    = next((p for p in parts if p.endswith("Set") and p[0].isupper()), "unknown")
    center   = next((p for p in parts if p.startswith("Center")), "")
    scanner_seg = next(
        (p for p in parts
         if any(p.lower().startswith(v) for v in _VENDOR_PREFIXES)
         and _FIELD_RE.search(p)), ""
    )
    vendor, field, scanner = "", "", ""
    if scanner_seg:
        sp = scanner_seg.split("_", 2)
        vendor  = VENDOR_FULL.get(sp[0].lower(), sp[0])
        fr      = sp[1].lower().replace(".", "").replace(",", "") if len(sp) > 1 else ""
        field   = FIELD_NORM.get(fr, sp[1] if len(sp) > 1 else "")
        scanner = sp[2] if len(sp) > 2 else ""
    return dict(modality=modality, split=split, center=center,
                vendor=vendor, field=field, scanner=scanner, sequence=filename)


def build_text_from_path(mat_path: str, sequence: str = "", train: bool = True) -> str:
    """Build descriptive text from mat_path + sequence name."""
    meta = parse_mat_path(mat_path)

    modality_text = MODALITY_TEXT.get(meta["modality"], "cardiac MRI")
    view_key      = sequence or meta["sequence"]
    view_text     = VIEW_TEXT.get(view_key, view_key.replace("_", " "))

    vendor  = meta["vendor"]
    field   = meta["field"]
    scanner = meta["scanner"]
    scanner_parts = [p for p in [vendor, scanner, field] if p]
    scanner_desc  = " ".join(scanner_parts) if scanner_parts else "unknown scanner"

    templates = [
        f"{modality_text} in {view_text} view acquired on {scanner_desc}",
        f"{view_text} {modality_text} from {scanner_desc}",
        f"cardiac MRI: {modality_text}, {view_text} view, {scanner_desc}",
        f"{modality_text}, {view_text}, scanned with {scanner_desc}",
        f"a {view_text} {modality_text} image on {scanner_desc}",
    ]
    if vendor and view_text:
        templates += [
            f"{vendor} {field} {modality_text} showing {view_text}",
            f"{view_text} view of {modality_text} on {vendor} {scanner}".strip(),
        ]

    if not train:
        return templates[0]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_npy_image(npy_path: str) -> np.ndarray:
    img = np.load(npy_path)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D npy, got {img.shape} from {npy_path}")
    return np.asarray(img, dtype=np.float32)


def normalize_image(img: np.ndarray, mode: str = "zscore") -> np.ndarray:
    if mode == "zscore":
        mean = float(img.mean())
        std  = float(img.std())
        img  = (img - mean) / std if std >= 1e-6 else img - mean
        img  = np.clip((img + 3.0) / 6.0, 0.0, 1.0)
    elif mode == "minmax":
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn) if mx - mn >= 1e-6 else np.zeros_like(img)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SequenceTransform:
    """[T, H, W] → [T, image_size, image_size] with shared geometry."""

    def __init__(
        self,
        image_size: int          = 224,
        train: bool              = True,
        random_rotate_deg: float = 10.0,
        translate: float         = 0.05,
        scale_min: float         = 0.95,
        scale_max: float         = 1.05,
        p_affine: float          = 0.5,
        p_noise: float           = 0.3,
        noise_std_max: float     = 0.05,
        p_cutout: float          = 0.3,
        cutout_min_frac: float   = 1 / 6,
        cutout_max_frac: float   = 1 / 3,
    ):
        self.image_size        = image_size
        self.train             = train
        self.random_rotate_deg = random_rotate_deg
        self.translate         = translate
        self.scale_min         = scale_min
        self.scale_max         = scale_max
        self.p_affine          = p_affine
        self.p_noise           = p_noise
        self.noise_std_max     = noise_std_max
        self.p_cutout          = p_cutout
        self.cutout_min_frac   = cutout_min_frac
        self.cutout_max_frac   = cutout_max_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        if not self.train:
            return x.clamp(0.0, 1.0)

        if random.random() < self.p_affine:
            angle  = random.uniform(-self.random_rotate_deg, self.random_rotate_deg)
            max_d  = self.translate * self.image_size
            transl = (int(round(random.uniform(-max_d, max_d))),
                      int(round(random.uniform(-max_d, max_d))))
            scale  = random.uniform(self.scale_min, self.scale_max)
            x = torch.cat([
                TF.affine(x[i].unsqueeze(0), angle=angle, translate=transl,
                          scale=scale, shear=[0.0, 0.0],
                          interpolation=TF.InterpolationMode.BILINEAR)
                for i in range(x.shape[0])
            ], dim=0)

        if random.random() < self.p_noise:
            x = x + torch.randn_like(x) * random.uniform(0.005, self.noise_std_max)

        if random.random() < self.p_cutout:
            _, H, W = x.shape
            ch = random.randint(int(H * self.cutout_min_frac), int(H * self.cutout_max_frac))
            cw = random.randint(int(W * self.cutout_min_frac), int(W * self.cutout_max_frac))
            x[:, random.randint(0, H - ch):, random.randint(0, W - cw):][
                :, :ch, :cw] = 0.0

        return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalCMRDataset(Dataset):
    """
    Multi-modal CMR sequence dataset for contrastive pre-training.

    CSV/Parquet columns required:
        mat_path, npy_path, modality, split, slice_idx, frame_idx
        [optional: sequence, view, vendor, field, scanner, center]

    Text is generated at runtime from mat_path — no 'text' column needed.
    frame_map (plain dict) replaces group_df to prevent worker memory leaks.
    """

    REQUIRED = ["npy_path", "mat_path", "modality", "slice_idx", "frame_idx"]

    def __init__(
        self,
        csv_path: str,
        image_size: int              = 224,
        normalize_mode: str          = "zscore",
        use_modalities: list[str] | None = None,
        train: bool                  = True,
        num_slices: int              = 3,
        num_frames: int              = 4,
        sample_repeats: int          = 1,
        p_noise: float               = 0.3,
        noise_std_max: float         = 0.05,
        p_cutout: float              = 0.3,
        temporal_dropout_prob: float = 0.3,
    ):
        # ---- Load CSV / Parquet (pyarrow for speed) ----
        if str(csv_path).endswith(".parquet"):
            df = pd.read_parquet(csv_path)
        else:
            try:
                df = pd.read_csv(csv_path, engine="pyarrow")
            except Exception:
                df = pd.read_csv(csv_path)

        for c in self.REQUIRED:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        df = df.dropna(subset=["npy_path", "mat_path", "modality"])

        if use_modalities is not None:
            allowed = {m.lower() for m in use_modalities}
            df = df[df["modality"].astype(str).str.lower().isin(allowed)]

        df = df.reset_index(drop=True)

        self.normalize_mode        = normalize_mode
        self.train                 = train
        self.num_slices            = num_slices
        self.num_frames            = num_frames
        self.temporal_dropout_prob = temporal_dropout_prob

        self.transform = SequenceTransform(
            image_size=image_size, train=train,
            p_affine=0.5 if train else 0.0,
            p_noise=p_noise if train else 0.0,
            noise_std_max=noise_std_max,
            p_cutout=p_cutout if train else 0.0,
        )

        # ---- Build sample list — one entry per mat_path ----
        base_samples: list[dict] = []
        mod_counts: dict[str, int] = defaultdict(int)

        for mat_path, g in df.groupby("mat_path"):
            g = g.copy()

            def _first(col, default=""):
                return str(g[col].iloc[0]) if col in g.columns and pd.notna(g[col].iloc[0]) else default

            modality = _first("modality")
            sequence = _first("sequence", _first("view", ""))
            split    = _first("split", "TrainingSet")

            frame_map: dict[tuple[int, int], str] = {}
            fallback = ""
            for row in g.itertuples(index=False):
                key = (int(row.slice_idx), int(row.frame_idx))
                frame_map[key] = row.npy_path
                if not fallback:
                    fallback = row.npy_path

            base_samples.append({
                "mat_path":    str(mat_path),
                "frame_map":   frame_map,
                "fallback":    fallback,
                "modality":    modality,
                "sequence":    sequence,
                "split":       split,
                "vendor":      _first("vendor"),
                "field":       _first("field"),
                "scanner":     _first("scanner"),
                "slice_values": sorted({k[0] for k in frame_map}),
                "frame_values": sorted({k[1] for k in frame_map}),
            })
            mod_counts[modality] += 1

        repeats = sample_repeats if train else 1
        self.samples = base_samples * repeats

        if train:
            print(f"MultiModalCMRDataset ({csv_path.split('/')[-1]}): "
                  f"{len(base_samples)} sequences ×{repeats} = {len(self.samples)} samples")
            for mod, cnt in sorted(mod_counts.items()):
                print(f"  {mod:12s}: {cnt:5d} sequences")

    def __len__(self):
        return len(self.samples)

    def _choose(self, values: list, k: int) -> list:
        n = len(values)
        if not self.train:
            idxs = np.clip(np.round(np.linspace(0, n - 1, k)).astype(int), 0, n - 1)
            return [values[i] for i in idxs]
        if n <= k:
            return random.choices(values, k=k)
        bucket = n / k
        return [values[random.randint(int(b * bucket),
                                      min(max(int(b * bucket) + 1, int((b + 1) * bucket)), n) - 1)]
                for b in range(k)]

    def __getitem__(self, idx: int) -> dict:
        s         = self.samples[idx]
        frame_map = s["frame_map"]

        chosen_slices = self._choose(s["slice_values"], self.num_slices)

        # ↓↓↓ 改这里：帧数不足时用全部帧，不重复填充
        frame_values = s["frame_values"]
        if len(frame_values) <= self.num_frames:
            chosen_frames = frame_values          # 全部用上，不重复
        else:
            chosen_frames = self._choose(frame_values, self.num_frames)  # 正常分层采样

        # temporal dropout 保持不变（只对多帧序列生效）
        if self.train and len(chosen_frames) > 1 and random.random() < self.temporal_dropout_prob:
            keep          = max(1, len(chosen_frames) - 1)
            chosen_frames = sorted(random.sample(chosen_frames, k=keep))
            # ↓ 注意：这里不再补齐到 num_frames，直接用实际帧数
            # while len(chosen_frames) < self.num_frames:  ← 删掉这两行
            #     chosen_frames.append(chosen_frames[-1])

        frames = []
        for sl in chosen_slices:
            for fr in chosen_frames:
                path = frame_map.get((sl, fr), s["fallback"])
                img  = load_npy_image(path)
                img  = normalize_image(img, mode=self.normalize_mode)
                frames.append(img)

        x    = torch.from_numpy(np.stack(frames).astype(np.float32))
        x    = self.transform(x)  
        x    = x.mean(0, keepdim=True)       # transform 已经支持任意 T
        text = build_text_from_path(s["mat_path"], s["sequence"], train=self.train)

        return {
            "image":    x,          # shape: [num_slices * actual_frames, H, W]
            "text":     text,
            "modality": s["modality"],
            "view":     s["sequence"],
        }

# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    image:          torch.Tensor
    input_ids:      torch.Tensor
    attention_mask: torch.Tensor
    raw_texts:      list
    modalities:     list
    views:          list


class MultiModalCollator:
    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[dict]) -> Batch:
        tok = self.tokenizer(
            [x["text"] for x in items],
            padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return Batch(
            image=torch.stack([x["image"] for x in items]),
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            raw_texts=[x["text"] for x in items],
            modalities=[x["modality"] for x in items],
            views=[x["view"] for x in items],
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_BACKBONE_CFGS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1,  512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class ImageEncoder(nn.Module):
    """Frame-level CNN backbone + temporal mean-pool + MLP projection."""

    def __init__(self, embed_dim=256, dropout=0.1, backbone_name="resnet50"):
        super().__init__()
        build_fn, weights, in_features = _BACKBONE_CFGS[backbone_name]
        backbone = build_fn(weights=weights)

        old = backbone.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new
        backbone.fc    = nn.Identity()
        self.backbone  = backbone

        self.proj = nn.Sequential(
            nn.Linear(in_features, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        feat = self.backbone(x.view(B * T, 1, H, W)).view(B, T, -1).mean(1)
        return F.normalize(self.proj(feat), dim=-1)


class TextEncoder(nn.Module):
    """Transformer encoder + mean-pool + MLP projection."""

    def __init__(self, model_name="distilbert-base-uncased", embed_dim=256, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden        = self.backbone.config.hidden_size
        self.proj     = nn.Sequential(
            nn.Linear(hidden, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        feat = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return F.normalize(self.proj(feat), dim=-1)


class MultiModalCLIP(nn.Module):
    def __init__(self, text_model="distilbert-base-uncased",
                 embed_dim=256, dropout=0.1, backbone="resnet50"):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim, dropout, backbone)
        self.text_encoder  = TextEncoder(text_model, embed_dim, dropout)
        self.logit_scale   = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images, input_ids, attention_mask):
        img_emb  = self.image_encoder(images)
        txt_emb  = self.text_encoder(input_ids, attention_mask)
        scale    = self.logit_scale.exp().clamp(max=100.0)
        logits_i = scale * img_emb @ txt_emb.t()
        return img_emb, txt_emb, logits_i, logits_i.t()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def clip_loss_multimodal(
    logits_i: torch.Tensor,
    logits_t: torch.Tensor,
    modalities: list[str],
    views: list[str],
    same_modality_soft: float = 0.05,
    same_view_soft: float     = 0.05,
) -> torch.Tensor:
    """CLIP loss with soft labels for same-modality / same-view pairs."""
    bs      = logits_i.size(0)
    targets = torch.eye(bs, device=logits_i.device)
    for i in range(bs):
        for j in range(bs):
            if i == j:
                continue
            if modalities[i] == modalities[j]:
                targets[i, j] += same_modality_soft
            if views[i] == views[j]:
                targets[i, j] += same_view_soft
    targets = targets / targets.sum(dim=1, keepdim=True)
    loss_i  = -(targets * F.log_softmax(logits_i, dim=1)).sum(1).mean()
    loss_t  = -(targets * F.log_softmax(logits_t, dim=1)).sum(1).mean()
    return 0.5 * (loss_i + loss_t)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_retrieval_metrics(logits_i: torch.Tensor) -> dict:
    bs      = logits_i.size(0)
    targets = torch.arange(bs, device=logits_i.device)

    def r_at_k(logits, k):
        return (logits.topk(min(k, bs), dim=1).indices
                == targets.unsqueeze(1)).any(1).float().mean().item()

    return {
        "i2t@1": r_at_k(logits_i,    1),
        "t2i@1": r_at_k(logits_i.t(), 1),
        "i2t@5": r_at_k(logits_i,    5),
    }


@torch.no_grad()
def full_retrieval_eval(model, loader, device, desc="Full-eval") -> dict:
    """Full N×N retrieval + modality-level accuracy."""
    model.eval()
    all_img, all_txt, all_mod = [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        img_emb, txt_emb, _, _ = model(
            batch.image.to(device),
            batch.input_ids.to(device),
            batch.attention_mask.to(device),
        )
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
        all_mod.extend(batch.modalities)

    img_emb = F.normalize(torch.cat(all_img), dim=-1)
    txt_emb = F.normalize(torch.cat(all_txt), dim=-1)
    sim     = img_emb @ txt_emb.t()
    N       = sim.size(0)
    targets = torch.arange(N)

    def r_at_k(logits, k):
        topk = logits.topk(min(k, N), dim=1).indices
        return (topk == targets.unsqueeze(1)).any(1).float().mean().item()

    # Modality accuracy: does top-1 retrieved text have same modality?
    mod2id  = {m: i for i, m in enumerate(sorted(set(all_mod)))}
    mod_ids = torch.tensor([mod2id[m] for m in all_mod])
    mod_acc = (mod_ids[sim.argmax(dim=1)] == mod_ids).float().mean().item()

    return {
        "i2t@1": r_at_k(sim,    1),
        "i2t@5": r_at_k(sim,    5),
        "i2t@10": r_at_k(sim,  10),
        "t2i@1": r_at_k(sim.t(), 1),
        "t2i@5": r_at_k(sim.t(), 5),
        "mod_acc": mod_acc,
    }


# ---------------------------------------------------------------------------
# Optimiser & scheduler
# ---------------------------------------------------------------------------

def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.image_encoder.backbone.parameters():
        p.requires_grad = trainable
    for p in model.text_encoder.backbone.parameters():
        p.requires_grad = trainable


def build_optimizer(model, lr_backbone, lr_head, weight_decay):
    bb   = list(model.image_encoder.backbone.parameters()) + \
           list(model.text_encoder.backbone.parameters())
    head = list(model.image_encoder.proj.parameters()) + \
           list(model.text_encoder.proj.parameters()) + \
           [model.logit_scale]
    return torch.optim.AdamW([
        {"params": bb,   "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head, "lr": lr_head,     "weight_decay": weight_decay},
    ])


def build_scheduler(optimizer, epochs, freeze_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - freeze_epochs, 1), eta_min=1e-7,
    )


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def _accum(running, new, bs):
    return {k: running.get(k, 0.0) + new[k] * bs for k in new}


def train_one_epoch(model, loader, optimizer, device, epoch,
                    amp=True, accum_steps=1,
                    same_modality_soft=0.05, same_view_soft=0.05):
    model.train()
    scaler  = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    running = {}
    count   = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch}")
    for step, batch in enumerate(pbar):
        images = batch.image.to(device,          non_blocking=True)
        ids    = batch.input_ids.to(device,      non_blocking=True)
        mask   = batch.attention_mask.to(device,  non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            _, _, logits_i, logits_t = model(images, ids, mask)
            loss = clip_loss_multimodal(
                logits_i, logits_t,
                batch.modalities, batch.views,
                same_modality_soft, same_view_soft,
            ) / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs      = images.size(0)
        metrics = batch_retrieval_metrics(logits_i)
        metrics["loss"] = loss.item() * accum_steps
        running = _accum(running, metrics, bs)
        count  += bs

        pbar.set_postfix({
            "loss":  f"{running['loss'] / count:.4f}",
            "i2t@1": f"{running['i2t@1'] / count:.4f}",
        })

    return {k: v / max(count, 1) for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch):
    model.eval()
    running = {}
    count   = 0

    pbar = tqdm(loader, desc=f"Val {epoch}")
    for batch in pbar:
        images = batch.image.to(device,         non_blocking=True)
        ids    = batch.input_ids.to(device,     non_blocking=True)
        mask   = batch.attention_mask.to(device, non_blocking=True)
        _, _, logits_i, logits_t = model(images, ids, mask)
        loss    = clip_loss_multimodal(logits_i, logits_t, batch.modalities, batch.views)
        metrics = batch_retrieval_metrics(logits_i)
        metrics["loss"] = loss.item()
        bs      = images.size(0)
        running = _accum(running, metrics, bs)
        count  += bs
        pbar.set_postfix({
            "loss":  f"{running['loss'] / count:.4f}",
            "i2t@1": f"{running['i2t@1'] / count:.4f}",
        })

    return {k: v / max(count, 1) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Checkpoint (local write + async NAS copy)
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "best_metric": best_metric, "args": vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer      = AutoTokenizer.from_pretrained(args.text_model)
    use_modalities = [m.strip() for m in args.modalities.split(",")] \
                     if args.modalities else None

    ds_kwargs = dict(
        image_size=args.image_size, normalize_mode=args.normalize_mode,
        use_modalities=use_modalities,
        num_slices=args.num_slices, num_frames=args.num_frames,
        p_noise=args.p_noise, noise_std_max=args.noise_std_max,
        p_cutout=args.p_cutout, temporal_dropout_prob=args.temporal_dropout_prob,
    )

    train_ds = MultiModalCMRDataset(
        args.train_csv, train=True, sample_repeats=args.sample_repeats, **ds_kwargs
    )
    val_ds   = MultiModalCMRDataset(args.val_csv,  train=False, **ds_kwargs)
    test_ds  = MultiModalCMRDataset(args.test_csv, train=False, **ds_kwargs)

    collator     = MultiModalCollator(tokenizer, max_length=args.max_length)
    make_loader  = lambda ds, shuffle, drop: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collator, drop_last=drop,
    )
    train_loader = make_loader(train_ds, True,  True)
    val_loader   = make_loader(val_ds,   False, False)
    test_loader  = make_loader(test_ds,  False, False)

    print(f"Train batches/epoch : {len(train_loader)}")
    print(f"Val   batches/epoch : {len(val_loader)}")

    model = MultiModalCLIP(
        text_model=args.text_model, embed_dim=args.embed_dim,
        dropout=args.dropout, backbone=args.backbone,
    ).to(device)

    img_p = sum(p.numel() for p in model.image_encoder.parameters()) / 1e6
    txt_p = sum(p.numel() for p in model.text_encoder.parameters())  / 1e6
    print(f"Image encoder: {img_p:.1f}M  Text encoder: {txt_p:.1f}M")

    optimizer = build_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.freeze_backbone_epochs)

    # Local checkpoint dir (avoid NAS write blocking training)
    os.makedirs(args.out_dir, exist_ok=True)
    local_ckpt = tempfile.mkdtemp(prefix="cmr_multimodal_ckpt_")
    print(f"Local checkpoint dir: {local_ckpt}")

    copy_threads: list = []

    def _async_copy(src, dst):
        def _do():
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[warn] NAS copy failed: {e}")
        t = threading.Thread(target=_do, daemon=True)
        t.start()
        return t

    best_metric = 0.0   # full-corpus i2t@1
    no_improve  = 0
    start_epoch = 1

    # ---- Resume from checkpoint ----
    if args.resume:
        ckpt_path = args.resume
        if not os.path.isfile(ckpt_path):
            # Try NAS out_dir/last.pt as default
            ckpt_path = os.path.join(args.out_dir, "last.pt")
        if os.path.isfile(ckpt_path):
            print(f"Resuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch  = ckpt["epoch"] + 1
            best_metric  = ckpt.get("best_metric", 0.0)
            print(f"  Resumed from epoch {ckpt['epoch']}  "
                  f"best_metric={best_metric:.4f}  "
                  f"continuing from epoch {start_epoch}")
        else:
            print(f"[warn] Resume checkpoint not found at {ckpt_path}, starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):

        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(model, False)
            if epoch == 1:
                print(f"Backbones frozen for {args.freeze_backbone_epochs} epoch(s).")
        else:
            set_backbone_trainable(model, True)
            if epoch == args.freeze_backbone_epochs + 1:
                print("Backbones unfrozen.")

        train_m = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            amp=args.amp, accum_steps=args.accum_steps,
            same_modality_soft=args.same_modality_soft,
            same_view_soft=args.same_view_soft,
        )

        # Batch-level val (fast)
        val_m = validate_one_epoch(model, val_loader, device, epoch)

        # Full-corpus val (accurate)
        full_m = full_retrieval_eval(model, val_loader, device, f"Full-eval {epoch}")

        if epoch > args.freeze_backbone_epochs:
            scheduler.step()

        is_best = full_m["i2t@1"] > best_metric
        if is_best:
            best_metric = full_m["i2t@1"]
            no_improve  = 0
        else:
            no_improve += 1

        print(
            f"[Epoch {epoch:3d}] "
            f"train loss={train_m['loss']:.4f} i2t@1={train_m['i2t@1']:.4f} | "
            f"val(batch) loss={val_m['loss']:.4f} | "
            f"val(full) i2t@1={full_m['i2t@1']:.4f} i2t@5={full_m['i2t@5']:.4f} "
            f"i2t@10={full_m['i2t@10']:.4f} mod_acc={full_m['mod_acc']:.4f}"
            + (" ← best" if is_best else f"  (no improve {no_improve}/{args.patience})")
        )

        # Save locally, copy to NAS in background
        local_last = os.path.join(local_ckpt, "last.pt")
        save_checkpoint(local_last, model, optimizer, scheduler, epoch, best_metric, args)
        copy_threads.append(_async_copy(local_last, os.path.join(args.out_dir, "last.pt")))

        if is_best:
            local_best = os.path.join(local_ckpt, "best.pt")
            shutil.copy2(local_last, local_best)
            copy_threads.append(_async_copy(local_best, os.path.join(args.out_dir, "best.pt")))

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print("\nWaiting for NAS sync ...")
    for t in copy_threads:
        t.join()

    print("\nFinal evaluation on test set (best checkpoint, full-corpus):")
    best_ckpt = torch.load(os.path.join(local_ckpt, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_m = full_retrieval_eval(model, test_loader, device, "Test full-eval")
    print(
        f"test i2t@1={test_m['i2t@1']:.4f}  i2t@5={test_m['i2t@5']:.4f}  "
        f"i2t@10={test_m['i2t@10']:.4f}  "
        f"t2i@1={test_m['t2i@1']:.4f}  t2i@5={test_m['t2i@5']:.4f}  "
        f"mod_acc={test_m['mod_acc']:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-modal CMR CLIP pre-training")

    # Data
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--test_csv",   required=True)
    p.add_argument("--modalities", default=None,
                   help="Comma-separated modalities (default: all)")

    # Output
    p.add_argument("--out_dir", default="./runs/cmr_multimodal")

    # Model
    p.add_argument("--backbone",   default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--text_model", default="distilbert-base-uncased")
    p.add_argument("--embed_dim",  type=int,   default=256)
    p.add_argument("--dropout",    type=float, default=0.1)

    # Image
    p.add_argument("--image_size",     type=int, default=224)
    p.add_argument("--normalize_mode", default="zscore", choices=["zscore", "minmax"])
    p.add_argument("--num_slices",     type=int, default=3)
    p.add_argument("--num_frames",     type=int, default=4)

    # Augmentation
    p.add_argument("--p_noise",               type=float, default=0.3)
    p.add_argument("--noise_std_max",         type=float, default=0.05)
    p.add_argument("--p_cutout",              type=float, default=0.3)
    p.add_argument("--temporal_dropout_prob", type=float, default=0.3)

    # Training
    p.add_argument("--sample_repeats",         type=int,   default=2)
    p.add_argument("--batch_size",             type=int,   default=32)
    p.add_argument("--accum_steps",            type=int,   default=2)
    p.add_argument("--epochs",                 type=int,   default=30)
    p.add_argument("--lr_backbone",            type=float, default=1e-5)
    p.add_argument("--lr_head",                type=float, default=1e-4)
    p.add_argument("--weight_decay",           type=float, default=5e-4)
    p.add_argument("--freeze_backbone_epochs", type=int,   default=2)
    p.add_argument("--same_modality_soft",     type=float, default=0.05)
    p.add_argument("--same_view_soft",         type=float, default=0.05)
    p.add_argument("--patience",               type=int,   default=7)

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from. "
                        "Defaults to --out_dir/last.pt if flag is set without value.")

    # Misc
    p.add_argument("--max_length",  type=int,  default=64)
    p.add_argument("--num_workers", type=int,  default=8)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--amp",         action="store_true")

    args = p.parse_args()
    main(args)