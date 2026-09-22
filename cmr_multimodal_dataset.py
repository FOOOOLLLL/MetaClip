"""
CMR Multi-Modal Dataset
=======================
Supports all modalities found under /xueyi/:
  Cine, LGE, Mapping, Flow2d, Aorta, Perfusion, T1w, T2w, Tagging, T1rho

Path structure (extracted from data.txt):
  /media/NAS_R02/USER_PATH/xueyi/{Modality}/{Split}/GTSOS/{Center}/{Vendor}_{Field}_{Scanner}/{Patient}/{filename}.mat

Key design decisions
--------------------
- Text is built from path components (modality + view + vendor + field + scanner).
  Across 7057 mat files spanning 10 modalities, this gives sufficient
  uniqueness for contrastive pre-training.
- No os.path.exists() checks during __init__ — too slow on NAS.
  Bad paths surface as FileNotFoundError at load time.
- CSV/Parquet is read with pyarrow for speed.
- frame_map replaces group_df to avoid DataLoader worker memory leaks.
- Checkpoints are written locally then async-copied to NAS.
"""

import os
import re
import math
import random
import argparse
import tempfile
import shutil
import threading
from dataclasses import dataclass
from collections import defaultdict

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
# Constants
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

# filename stem → human-readable view phrase
VIEW_TEXT = {
    # Cine
    "cine_sax":     "short-axis",
    "cine_lax":     "long-axis",
    "cine_lax_2ch": "2-chamber",
    "cine_lax_3ch": "3-chamber",
    "cine_lax_4ch": "4-chamber",
    "cine_lvot":    "left ventricular outflow tract",
    "cine_rvot":    "right ventricular outflow tract",
    "cine_ot":      "outflow tract",
    "cine_lax_r2ch":"right 2-chamber",
    # LGE
    "lge_sax":      "short-axis",
    "lge_lax":      "long-axis",
    "lge_lax_2ch":  "2-chamber",
    "lge_lax_3ch":  "3-chamber",
    "lge_lax_4ch":  "4-chamber",
    # Mapping
    "T1map":        "T1 mapping",
    "T1mappost":    "post-contrast T1 mapping",
    "T2map":        "T2 mapping",
    "T2smap":       "T2-star mapping",
    # Flow2d
    "flow2d":            "through-plane flow",
    "flow2d_inplane":    "in-plane flow",
    "flow2d_throughplane_D": "through-plane flow magnitude",
    "flow2d_throughplane_M": "through-plane flow phase",
    # Aorta
    "aorta_sag":    "sagittal aorta",
    "aorta_tra":    "transverse aorta",
    # Others
    "perfusion":    "myocardial perfusion",
    "T1w":          "T1-weighted",
    "T2w":          "T2-weighted",
    "tagging":      "myocardial tagging",
    "T1rho":        "T1rho",
}

VENDOR_FULL = {
    "siemens": "Siemens",
    "uih":     "United Imaging",
    "philips": "Philips",
    "ge":      "GE",
    "canon":   "Canon",
}

FIELD_NORM = {
    "30t": "3.0T", "3t": "3.0T", "30":  "3.0T",
    "15t": "1.5T", "1t": "1.5T", "15":  "1.5T",
    "055t": "0.55T",
}


# ---------------------------------------------------------------------------
# Path parser
# ---------------------------------------------------------------------------

def parse_mat_path(mat_path: str) -> dict:
    """
    Extract structured metadata from a mat file path.

    Path structure:
        .../xueyi/{Modality}/{Split}/GTSOS/{Center}/{Vendor}_{Field}_{Scanner}/{Patient}/{file}.mat

    Returns a dict with keys:
        modality, split, center, vendor, field, scanner, patient, sequence, view_text
    """
    parts    = mat_path.replace("\\", "/").split("/")
    filename = os.path.splitext(os.path.basename(mat_path))[0]

    # Modality — exact match against known list
    modality = next((p for p in parts if p in ALL_MODALITIES), "unknown")

    # Split
    split = next(
        (p for p in parts if p.endswith("Set") and p[0].isupper()),
        "unknown",
    )

    # Center
    center = next((p for p in parts if p.startswith("Center")), "")

    # Scanner segment: must start with a known vendor prefix AND contain a
    # field-strength token (e.g. "30T", "15T"). This prevents the modality
    # name "LGE" or "GE" (the company) from being misidentified as a vendor
    # when there is no valid scanner segment in the path.
    _VENDOR_PREFIXES = tuple(VENDOR_FULL.keys())   # siemens, uih, philips, ge, canon
    _FIELD_RE        = re.compile(r"\d+[.,]?\d*[Tt]", re.IGNORECASE)

    scanner_seg = ""
    for p in parts:
        if any(p.lower().startswith(v) for v in _VENDOR_PREFIXES) and _FIELD_RE.search(p):
            scanner_seg = p
            break

    vendor, field, scanner_model = _parse_scanner_seg(scanner_seg)

    # Patient
    patient = next((p for p in parts if re.match(r"^P\d+$", p)), "")

    # View text from filename
    view_text = VIEW_TEXT.get(filename, filename.replace("_", " "))

    return {
        "modality":  modality,
        "split":     split,
        "center":    center,
        "vendor":    vendor,
        "field":     field,
        "scanner":   scanner_model,
        "patient":   patient,
        "sequence":  filename,
        "view_text": view_text,
    }


def _parse_scanner_seg(seg: str) -> tuple[str, str, str]:
    """
    Parse scanner segment into (vendor, field, model).

    Examples:
        "Siemens_30T_Vida"   -> ("Siemens",         "3.0T", "Vida")
        "UIH_30T_umr780"     -> ("United Imaging",   "3.0T", "umr780")
        "Siemens_30T_CIMA.X" -> ("Siemens",         "3.0T", "CIMA.X")
        "Siemens_15T_Aera"   -> ("Siemens",         "1.5T", "Aera")

    Split on first two underscores only so model names like "CIMA.X" are kept intact.
    """
    if not seg:
        return "", "", ""

    parts = seg.split("_", 2)   # max 3 parts

    vendor_raw    = parts[0].lower()
    vendor        = VENDOR_FULL.get(vendor_raw, parts[0])

    field_raw     = parts[1].lower().replace(".", "").replace(",", "") if len(parts) > 1 else ""
    field         = FIELD_NORM.get(field_raw, parts[1] if len(parts) > 1 else "")

    scanner_model = parts[2] if len(parts) > 2 else ""

    return vendor, field, scanner_model



# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def build_text_from_path(mat_path: str, train: bool = True) -> str:
    """
    Build a descriptive text string from mat file path.

    In training mode: randomly sample from multiple templates to improve
    text encoder robustness.
    In val/test mode: deterministic — always use the first template.

    The combination of modality + view + vendor + field + scanner gives
    sufficient uniqueness across 7057 files for contrastive pre-training.
    """
    meta = parse_mat_path(mat_path)

    modality_text = MODALITY_TEXT.get(meta["modality"], "cardiac MRI")
    view_text     = meta["view_text"]
    vendor        = meta["vendor"]
    field         = meta["field"]
    scanner       = meta["scanner"]

    # Build scanner description
    scanner_parts = [p for p in [vendor, scanner, field] if p]
    scanner_desc  = " ".join(scanner_parts) if scanner_parts else "unknown scanner"

    templates = [
        f"{modality_text} in {view_text} view acquired on {scanner_desc}",
        f"{view_text} {modality_text} from {scanner_desc}",
        f"cardiac MRI sequence: {modality_text}, {view_text} view, {scanner_desc}",
        f"{modality_text}, {view_text}, scanned with {scanner_desc}",
        f"a {view_text} {modality_text} image on {scanner_desc}",
    ]

    # Add more specific templates when both vendor and view are available
    if vendor and view_text:
        templates += [
            f"{vendor} {field} {modality_text} showing {view_text}",
            f"{view_text} view of {modality_text} on {vendor} {scanner}".strip(),
        ]

    if not train:
        return templates[0]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Verify parser on known paths
# ---------------------------------------------------------------------------

EXAMPLE_PATHS = [
    "/media/NAS_R02/USER_PATH/xueyi/Cine/TrainingSet/GTSOS/Center001/UIH_30T_umr780/P001/cine_lax_4ch.mat",
    "/media/NAS_R02/USER_PATH/xueyi/LGE/TestSet/GTSOS/Center002/Siemens_30T_CIMA.X/P006/lge_sax.mat",
    "/media/NAS_R02/USER_PATH/xueyi/Mapping/TrainingSet/GTSOS/Center001/UIH_30T_umr780/P001/T1map.mat",
    "/media/NAS_R02/USER_PATH/xueyi/Flow2d/TestSet/GTSOS/Center010/UIH_30T_umr790/P001/flow2d_inplane.mat",
    "/media/NAS_R02/USER_PATH/xueyi/Aorta/TrainingSet/GTSOS/Center015/Siemens_30T_Vida/P002/aorta_sag.mat",
    "/media/NAS_R02/USER_PATH/xueyi/Tagging/TrainingSet/GTSOS/Center015/Siemens_30T_Vida/P002/tagging.mat",
]


def verify_parser():
    """Quick sanity check — run as: python cmr_multimodal_dataset.py --verify"""
    print("=" * 70)
    print("Path parser verification")
    print("=" * 70)
    for path in EXAMPLE_PATHS:
        meta = parse_mat_path(path)
        text_train = build_text_from_path(path, train=True)
        text_val   = build_text_from_path(path, train=False)
        print(f"\nPath    : .../{'/'.join(path.split('/')[-3:])}")
        print(f"Modality: {meta['modality']:12s}  View: {meta['view_text']}")
        print(f"Vendor  : {meta['vendor']:12s}  Field: {meta['field']:8s}  Scanner: {meta['scanner']}")
        print(f"Text(val): {text_val}")
        print(f"Text(trn): {text_train}")
    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_npy_image(npy_path: str) -> np.ndarray:
    img = np.load(npy_path)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D npy, got shape={img.shape} from {npy_path}")
    return np.asarray(img, dtype=np.float32)


def normalize_image(img: np.ndarray, mode: str = "zscore") -> np.ndarray:
    if mode == "zscore":
        mean = float(img.mean())
        std  = float(img.std())
        img  = (img - mean) / std if std >= 1e-6 else (img - mean)
        img  = np.clip((img + 3.0) / 6.0, 0.0, 1.0)
    elif mode == "minmax":
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn) if mx - mn >= 1e-6 else np.zeros_like(img)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SequenceTransform:
    """
    Input : [T, H, W]  float32 in [0, 1]
    Output: [T, image_size, image_size]

    Geometric transforms are shared across all frames (temporal consistency).
    Noise and cutout are applied independently.
    """

    def __init__(
        self,
        image_size: int          = 224,
        train: bool              = True,
        random_rotate_deg: float = 10.0,
        translate: float         = 0.05,
        scale_min: float         = 0.95,
        scale_max: float         = 1.05,
        p_affine: float          = 0.5,
        p_hflip: float           = 0.0,
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
        self.p_hflip           = p_hflip
        self.p_noise           = p_noise
        self.noise_std_max     = noise_std_max
        self.p_cutout          = p_cutout
        self.cutout_min_frac   = cutout_min_frac
        self.cutout_max_frac   = cutout_max_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [T,H,W], got {tuple(x.shape)}")

        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if not self.train:
            return x.clamp(0.0, 1.0)

        if random.random() < self.p_affine:
            angle  = random.uniform(-self.random_rotate_deg, self.random_rotate_deg)
            max_d  = self.translate * self.image_size
            transl = (
                int(round(random.uniform(-max_d, max_d))),
                int(round(random.uniform(-max_d, max_d))),
            )
            scale  = random.uniform(self.scale_min, self.scale_max)
            frames = [
                TF.affine(
                    x[i].unsqueeze(0),
                    angle=angle, translate=transl, scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                for i in range(x.shape[0])
            ]
            x = torch.cat(frames, dim=0)

        if self.p_hflip > 0 and random.random() < self.p_hflip:
            x = torch.cat(
                [TF.hflip(x[i].unsqueeze(0)) for i in range(x.shape[0])], dim=0
            )

        if random.random() < self.p_noise:
            std = random.uniform(0.005, self.noise_std_max)
            x   = x + torch.randn_like(x) * std

        if random.random() < self.p_cutout:
            _, H, W = x.shape
            ch = random.randint(int(H * self.cutout_min_frac), int(H * self.cutout_max_frac))
            cw = random.randint(int(W * self.cutout_min_frac), int(W * self.cutout_max_frac))
            y0 = random.randint(0, H - ch)
            x0 = random.randint(0, W - cw)
            x[:, y0:y0 + ch, x0:x0 + cw] = 0.0

        return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalCMRDataset(Dataset):
    """
    Multi-modal CMR sequence dataset for contrastive pre-training.

    Reads a CSV/Parquet file with columns:
        npy_path, mat_path, modality, view, slice_idx, frame_idx
        [optional: split, center, vendor, field, scanner]

    Text is built from mat_path (all metadata is encoded in the path).
    frame_map (dict) replaces group_df to avoid DataLoader memory leaks.

    sample_repeats > 1 repeats each sequence in the epoch with different
    random slice/frame draws and augmentations.
    """

    REQUIRED_COLS = ["npy_path", "mat_path", "modality", "view", "slice_idx", "frame_idx"]

    def __init__(
        self,
        csv_path: str,
        image_size: int              = 224,
        normalize_mode: str          = "zscore",
        use_modalities: list[str] | None = None,   # None = all
        use_views: list[str] | None  = None,        # None = all
        train: bool                  = True,
        num_slices: int              = 3,
        num_frames: int              = 4,
        sample_repeats: int          = 1,
        p_noise: float               = 0.3,
        noise_std_max: float         = 0.05,
        p_cutout: float              = 0.3,
        temporal_dropout_prob: float = 0.3,
    ):
        # ---- Load CSV / Parquet ----
        if str(csv_path).endswith(".parquet"):
            df = pd.read_parquet(csv_path)
        else:
            try:
                df = pd.read_csv(csv_path, engine="pyarrow")
            except Exception:
                df = pd.read_csv(csv_path)

        for c in self.REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        df = df.dropna(subset=["npy_path", "mat_path", "modality", "view"])

        # ---- Filter modalities ----
        if use_modalities is not None:
            allowed = {m.lower() for m in use_modalities}
            df = df[df["modality"].astype(str).str.lower().isin(allowed)]

        # ---- Filter views ----
        if use_views is not None:
            allowed = {v.lower() for v in use_views}
            df = df[df["view"].astype(str).str.lower().isin(allowed)]

        df = df.reset_index(drop=True)

        self.normalize_mode        = normalize_mode
        self.train                 = train
        self.num_slices            = num_slices
        self.num_frames            = num_frames
        self.temporal_dropout_prob = temporal_dropout_prob

        self.transform = SequenceTransform(
            image_size=image_size, train=train,
            random_rotate_deg=10.0, translate=0.05,
            scale_min=0.95, scale_max=1.05,
            p_affine=0.5 if train else 0.0, p_hflip=0.0,
            p_noise=p_noise if train else 0.0,
            noise_std_max=noise_std_max,
            p_cutout=p_cutout if train else 0.0,
        )

        # ---- Build sample list (one entry per mat_path) ----
        base_samples: list[dict] = []
        for mat_path, g in df.groupby("mat_path"):
            g = g.copy()

            def _first(col: str, default: str = "") -> str:
                return (
                    str(g[col].iloc[0])
                    if col in g.columns and pd.notna(g[col].iloc[0])
                    else default
                )

            # Lightweight frame_map: (slice_idx, frame_idx) -> npy_path
            frame_map: dict[tuple[int, int], str] = {}
            fallback_path = ""
            for row in g.itertuples(index=False):
                key = (int(row.slice_idx), int(row.frame_idx))
                frame_map[key] = row.npy_path
                if not fallback_path:
                    fallback_path = row.npy_path

            slice_values = sorted({k[0] for k in frame_map})
            frame_values = sorted({k[1] for k in frame_map})

            # Parse text metadata from mat_path
            meta = parse_mat_path(str(mat_path))

            base_samples.append({
                "mat_path":     str(mat_path),
                "frame_map":    frame_map,
                "fallback_path": fallback_path,
                "modality":     _first("modality", meta["modality"]),
                "view":         _first("view",     meta["sequence"]),
                "set_name":     _first("set_name", meta["split"]),
                "vendor":       meta["vendor"],
                "field":        meta["field"],
                "scanner":      meta["scanner"],
                "view_text":    meta["view_text"],
                "slice_values": slice_values,
                "frame_values": frame_values,
            })

        repeats = sample_repeats if train else 1
        self.samples = base_samples * repeats

        # Print dataset summary
        if train:
            mod_counts: dict[str, int] = defaultdict(int)
            for s in base_samples:
                mod_counts[s["modality"]] += 1
            print(f"MultiModalCMRDataset: {len(base_samples)} sequences "
                  f"(×{repeats} = {len(self.samples)} samples)")
            for mod, cnt in sorted(mod_counts.items()):
                print(f"  {mod:12s}: {cnt}")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Index selection
    # ------------------------------------------------------------------

    def _choose_uniform(self, values: list, k: int) -> list:
        idxs = np.round(np.linspace(0, len(values) - 1, k)).astype(int)
        idxs = np.clip(idxs, 0, len(values) - 1)
        return [values[i] for i in idxs]

    def _choose_random(self, values: list, k: int) -> list:
        """Bucket-based random sampling — full coverage, variety across repeats."""
        n = len(values)
        if n <= k:
            return random.choices(values, k=k)
        bucket = n / k
        chosen = []
        for b in range(k):
            lo = int(b * bucket)
            hi = min(max(lo + 1, int((b + 1) * bucket)), n)
            chosen.append(values[random.randint(lo, hi - 1)])
        return chosen

    def _choose(self, values: list, k: int) -> list:
        return self._choose_random(values, k) if self.train else self._choose_uniform(values, k)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        sample    = self.samples[idx]
        frame_map = sample["frame_map"]
        fallback  = sample["fallback_path"]

        chosen_slices = self._choose(sample["slice_values"], self.num_slices)
        chosen_frames = self._choose(sample["frame_values"], self.num_frames)

        # Temporal dropout: randomly drop one frame, repeat last
        if (
            self.train
            and len(chosen_frames) > 1
            and random.random() < self.temporal_dropout_prob
        ):
            keep          = max(1, len(chosen_frames) - 1)
            chosen_frames = sorted(random.sample(chosen_frames, k=keep))
            while len(chosen_frames) < self.num_frames:
                chosen_frames.append(chosen_frames[-1])

        frames = []
        for s in chosen_slices:
            for t in chosen_frames:
                npy_path = frame_map.get((s, t), fallback)
                img      = load_npy_image(npy_path)
                img      = normalize_image(img, mode=self.normalize_mode)
                frames.append(img)

        x = torch.from_numpy(np.stack(frames, axis=0).astype(np.float32))
        x = self.transform(x)   # [T, H, W]

        # Build text from mat_path (richer than using stored metadata)
        text = build_text_from_path(sample["mat_path"], train=self.train)

        return {
            "image":    x,
            "text":     text,
            "modality": sample["modality"],
            "view":     sample["view"],
            "set_name": sample["set_name"],
            "mat_path": sample["mat_path"],
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
        images     = torch.stack([x["image"]    for x in items])
        texts      = [x["text"]     for x in items]
        modalities = [x["modality"] for x in items]
        views      = [x["view"]     for x in items]
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return Batch(
            image=images,
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            raw_texts=texts,
            modalities=modalities,
            views=views,
        )


# ---------------------------------------------------------------------------
# Loss: soft-label CLIP with same-modality smoothing
# ---------------------------------------------------------------------------

def clip_loss_multimodal(
    logits_per_image: torch.Tensor,
    logits_per_text:  torch.Tensor,
    modalities: list[str],
    views: list[str],
    same_modality_soft: float = 0.05,
    same_view_soft: float     = 0.05,
) -> torch.Tensor:
    """
    Symmetric cross-entropy CLIP loss with soft labels.

    Same-modality pairs get a small positive label (same_modality_soft).
    Same-modality AND same-view pairs get a slightly larger label
    (same_modality_soft + same_view_soft).
    This prevents over-penalising semantically similar negatives.
    """
    bs      = logits_per_image.size(0)
    device  = logits_per_image.device
    targets = torch.eye(bs, device=device)

    for i in range(bs):
        for j in range(bs):
            if i == j:
                continue
            if modalities[i] == modalities[j]:
                targets[i, j] += same_modality_soft
            if views[i] == views[j]:
                targets[i, j] += same_view_soft

    targets = targets / targets.sum(dim=1, keepdim=True)

    loss_i = -(targets * F.log_softmax(logits_per_image, dim=1)).sum(1).mean()
    loss_t = -(targets * F.log_softmax(logits_per_text,  dim=1)).sum(1).mean()
    return 0.5 * (loss_i + loss_t)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def retrieval_metrics(logits_per_image: torch.Tensor) -> dict:
    bs      = logits_per_image.size(0)
    targets = torch.arange(bs, device=logits_per_image.device)

    def r_at_k(logits, k):
        topk = logits.topk(min(k, bs), dim=1).indices
        return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    return {
        "i2t@1": r_at_k(logits_per_image,    1),
        "t2i@1": r_at_k(logits_per_image.t(), 1),
        "i2t@5": r_at_k(logits_per_image,    5),
        "t2i@5": r_at_k(logits_per_image.t(), 5),
    }


@torch.no_grad()
def full_retrieval_eval(model, loader, device, desc: str = "Full-eval") -> dict:
    """
    Full-corpus N×N retrieval — accurate R@k with resolution 1/N.
    Also computes modality-retrieval accuracy as a proxy metric.
    """
    model.eval()
    all_img, all_txt, all_mod = [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        images         = batch.image.to(device,          non_blocking=True)
        input_ids      = batch.input_ids.to(device,      non_blocking=True)
        attention_mask = batch.attention_mask.to(device,  non_blocking=True)
        img_emb, txt_emb, _, _ = model(images, input_ids, attention_mask)
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
        all_mod.extend(batch.modalities)

    img_emb = F.normalize(torch.cat(all_img, dim=0), dim=-1)
    txt_emb = F.normalize(torch.cat(all_txt, dim=0), dim=-1)
    sim     = img_emb @ txt_emb.t()
    N       = sim.size(0)
    targets = torch.arange(N)

    def r_at_k(logits, k):
        topk = logits.topk(min(k, N), dim=1).indices
        return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    # Modality retrieval: does top-1 retrieved text have same modality?
    mod2id  = {m: i for i, m in enumerate(sorted(set(all_mod)))}
    mod_ids = torch.tensor([mod2id[m] for m in all_mod])
    top1_txt_ids = sim.argmax(dim=1)
    mod_acc = (mod_ids[top1_txt_ids] == mod_ids).float().mean().item()

    return {
        "i2t@1":    r_at_k(sim,    1),
        "i2t@5":    r_at_k(sim,    5),
        "i2t@10":   r_at_k(sim,   10),
        "t2i@1":    r_at_k(sim.t(), 1),
        "t2i@5":    r_at_k(sim.t(), 5),
        "mod_acc":  mod_acc,   # modality-level retrieval accuracy
    }


# ---------------------------------------------------------------------------
# CLI entry point — verify parser
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="Run path parser verification and exit")
    args = p.parse_args()
    if args.verify:
        verify_parser()