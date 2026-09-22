"""
train_classification.py
=======================
Downstream classification using the frozen multi-modal CLIP image encoder.

Task 1 — Modality classification (7 classes):
    Aorta, Cine, Flow2d, LGE, Mapping, T2w, Tagging

Task 2 — Cine view classification (4 classes):
    sax, 4ch, 3ch, 2ch

Training strategy
-----------------
Stage 1 (linear probe):
    Encoder frozen, only the classification head is trained.
    Fast, good baseline.

Stage 2 (fine-tune, optional --finetune):
    Encoder unfrozen with small lr, head with larger lr.
    Usually gives +2–5% over linear probe.

Usage
-----
# Task 1
python train_classification.py \
    --task      modality \
    --train_csv /path/all_train.parquet \
    --test_csv  /path/all_test.parquet \
    --ckpt_path /path/cmr_multimodal/best.pt \
    --out_dir   /path/runs/cls_modality \
    --amp

# Task 2
python train_classification.py \
    --task      cine_view \
    --train_csv /path/all_train.parquet \
    --test_csv  /path/all_test.parquet \
    --ckpt_path /path/cmr_multimodal/best.pt \
    --out_dir   /path/runs/cls_cine_view \
    --amp

# Task 2 with fine-tuning
python train_classification.py \
    --task      cine_view \
    --finetune \
    --lr_backbone 1e-6 \
    ...
"""

import os
import random
import argparse
import tempfile
import shutil
import threading
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Task 1: modality classification
# Keep modalities with sufficient training data (≥64 sequences)
MOD_CLASSES = ["Aorta", "Cine", "Flow2d", "LGE", "Mapping", "T2w", "Tagging"]

# Task 2: Cine view classification
# mat_view (filename stem) → label name
CINE_VIEW_MAP = {
    "cine_sax":     "sax",
    "cine_lax_2ch": "2ch",
    "cine_lax_3ch": "3ch",
    "cine_lax_4ch": "4ch",
}
CINE_VIEW_CLASSES = ["2ch", "3ch", "4ch", "sax"]  # sorted for consistency


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Image I/O & normalisation
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
        p_noise: float           = 0.2,
        noise_std_max: float     = 0.04,
        p_cutout: float          = 0.2,
        cutout_frac: float       = 0.25,
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
        self.cutout_frac       = cutout_frac

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
            ch = int(H * self.cutout_frac)
            cw = int(W * self.cutout_frac)
            y0 = random.randint(0, H - ch)
            x0 = random.randint(0, W - cw)
            x[:, y0:y0 + ch, x0:x0 + cw] = 0.0

        return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    """
    Sequence-level classification dataset.

    For each sequence (mat_path), samples num_slices × num_frames npy files
    and returns a stacked tensor [T, H, W] with a class label.

    Handles both tasks through the `task_cfg` dict:
        task_cfg["classes"]   : list of class names (defines label mapping)
        task_cfg["filter"]    : callable(row) → bool, row is a pandas Series
        task_cfg["label"]     : callable(row) → str, returns class name
    """

    REQUIRED = ["npy_path", "mat_path", "modality", "slice_idx", "frame_idx"]

    def __init__(
        self,
        csv_path: str,
        task_cfg: dict,
        train: bool        = True,
        image_size: int    = 224,
        normalize_mode: str = "zscore",
        num_slices: int    = 3,
        num_frames: int    = 4,
    ):
        if str(csv_path).endswith(".parquet"):
            df = pd.read_parquet(csv_path)
        else:
            try:
                df = pd.read_csv(csv_path, engine="pyarrow")
            except Exception:
                df = pd.read_csv(csv_path)

        df = df.dropna(subset=["npy_path", "mat_path", "modality"])

        # Add mat_view column (filename stem) for view classification
        df["mat_view"] = df["mat_path"].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )

        self.classes    = task_cfg["classes"]
        self.class2idx  = {c: i for i, c in enumerate(self.classes)}
        self.train      = train
        self.normalize  = normalize_mode
        self.num_slices = num_slices
        self.num_frames = num_frames

        self.transform = SequenceTransform(
            image_size=image_size, train=train,
            p_affine=0.5 if train else 0.0,
            p_noise=0.2 if train else 0.0,
            p_cutout=0.2 if train else 0.0,
        )

        # Build per-sequence sample list
        self.samples: list[dict] = []
        label_counts: dict[str, int] = defaultdict(int)

        for mat_path, g in df.groupby("mat_path"):
            g = g.copy()
            row = g.iloc[0]

            # Apply task filter
            if not task_cfg["filter"](row):
                continue

            # Get label
            label_name = task_cfg["label"](row)
            if label_name not in self.class2idx:
                continue

            label_idx = self.class2idx[label_name]

            # Build frame_map: (slice_idx, frame_idx) → npy_path
            frame_map: dict[tuple[int, int], str] = {}
            fallback = ""
            for r in g.itertuples(index=False):
                key = (int(r.slice_idx), int(r.frame_idx))
                frame_map[key] = r.npy_path
                if not fallback:
                    fallback = r.npy_path

            self.samples.append({
                "mat_path":    str(mat_path),
                "frame_map":   frame_map,
                "fallback":    fallback,
                "label_idx":   label_idx,
                "label_name":  label_name,
                "slice_values": sorted({k[0] for k in frame_map}),
                "frame_values": sorted({k[1] for k in frame_map}),
            })
            label_counts[label_name] += 1

        print(f"  {'train' if train else 'test ':5s} {csv_path.split('/')[-1]}: "
              f"{len(self.samples)} sequences")
        for cls in self.classes:
            print(f"    {cls:12s}: {label_counts.get(cls, 0)}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> list[int]:
        """Return all label indices (for WeightedRandomSampler)."""
        return [s["label_idx"] for s in self.samples]

    def _choose(self, values: list, k: int) -> list:
        n = len(values)
        if not self.train:
            idxs = np.clip(np.round(np.linspace(0, n - 1, k)).astype(int), 0, n - 1)
            return [values[i] for i in idxs]
        if n <= k:
            return random.choices(values, k=k)
        bucket = n / k
        return [
            values[random.randint(
                int(b * bucket),
                min(max(int(b * bucket) + 1, int((b + 1) * bucket)), n) - 1
            )]
            for b in range(k)
        ]

    def __getitem__(self, idx: int) -> dict:
        s         = self.samples[idx]
        frame_map = s["frame_map"]

        chosen_slices = self._choose(s["slice_values"], self.num_slices)
        chosen_frames = self._choose(s["frame_values"], self.num_frames)

        frames = []
        for sl in chosen_slices:
            for fr in chosen_frames:
                path = frame_map.get((sl, fr), s["fallback"])
                img  = load_npy_image(path)
                img  = normalize_image(img, mode=self.normalize)
                frames.append(img)

        x = torch.from_numpy(np.stack(frames).astype(np.float32))
        x = self.transform(x)

        return {"image": x, "label": s["label_idx"], "label_name": s["label_name"]}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image":      torch.stack([b["image"] for b in batch]),
        "label":      torch.tensor([b["label"] for b in batch]),
        "label_name": [b["label_name"] for b in batch],
    }


def make_balanced_sampler(dataset: ClassificationDataset) -> WeightedRandomSampler:
    """Class-balanced sampler to handle class imbalance."""
    labels  = dataset.get_labels()
    counts  = np.bincount(labels, minlength=len(dataset.classes))
    weights = 1.0 / (counts + 1e-6)
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# Image encoder (loaded from pre-trained CLIP checkpoint)
# ---------------------------------------------------------------------------

_BACKBONE_CFGS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1,  512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class ImageEncoder(nn.Module):
    """Frame-level CNN backbone + temporal mean-pool + MLP projection."""

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1,
                 backbone_name: str = "resnet50"):
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


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------

class ClassificationModel(nn.Module):
    """Frozen (or fine-tunable) image encoder + linear classification head."""

    def __init__(self, num_classes: int, embed_dim: int = 256,
                 dropout: float = 0.1, backbone_name: str = "resnet50"):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim, dropout, backbone_name)
        self.head    = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.encoder(x)
        logits = self.head(feat)
        return logits

    def load_pretrained_encoder(self, ckpt_path: str | None, device):
        """
        Load image encoder weights from a pre-training checkpoint.

        Supports three checkpoint formats (auto-detected):
          - CLIP : keys start with 'image_encoder.'
                   (from train_cmr_multimodal.py)
          - MIM  : keys start with 'encoder.'
                   (from train_mim.py, after convert_mim_ckpt.py)
          - None : no checkpoint — keep ImageNet weights as-is

        Parameters
        ----------
        ckpt_path : str or None
            Path to checkpoint file. Pass None to use ImageNet init only.
        """
        if ckpt_path is None:
            print("  No checkpoint — using ImageNet pre-trained weights only.")
            return self

        ckpt       = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"]

        # ---- Auto-detect format ----
        has_clip = any(k.startswith("image_encoder.") for k in state_dict)
        has_mim  = any(k.startswith("encoder.") for k in state_dict)

        if has_clip:
            print("  Detected CLIP checkpoint")
            encoder_state = {
                k.replace("image_encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("image_encoder.")
            }
            missing, unexpected = self.encoder.load_state_dict(
                encoder_state, strict=True)
            if missing:
                print(f"  Missing keys: {missing}")

        elif has_mim:
            print("  Detected MIM checkpoint")
            # MIM checkpoint stores ResNet50-compatible keys under 'encoder.*'
            # Map to ImageEncoder.backbone.*
            mim_state = {
                k.replace("encoder.", "backbone.", 1): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            missing, unexpected = self.encoder.load_state_dict(
                mim_state, strict=False)
            print(f"  Encoder load: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected")

        else:
            raise ValueError(
                f"Unknown checkpoint format in {ckpt_path}.\n"
                "Expected keys starting with 'image_encoder.' (CLIP) "
                "or 'encoder.' (MIM)."
            )

        print(f"  Loaded encoder from {ckpt_path}")
        return self

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, amp=True, accum_steps=1):
    model.train()
    scaler   = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    total_loss, correct, count = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch}")
    for step, batch in enumerate(pbar):
        x      = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss   = F.cross_entropy(logits, labels) / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs           = x.size(0)
        total_loss  += loss.item() * accum_steps * bs
        correct     += (logits.argmax(1) == labels).sum().item()
        count       += bs
        pbar.set_postfix({
            "loss": f"{total_loss / count:.4f}",
            "acc":  f"{correct / count:.4f}",
        })

    return {"loss": total_loss / max(count, 1), "acc": correct / max(count, 1)}


@torch.no_grad()
def evaluate(model, loader, device, classes: list[str]) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        x      = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report  = classification_report(
        all_labels, all_preds,
        target_names=classes, digits=4, zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {"acc": acc, "report": report, "cm": cm,
            "preds": all_preds, "labels": all_labels}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc, "args": vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Task config ----
    if args.task == "modality":
        task_cfg = {
            "classes": MOD_CLASSES,
            "filter":  lambda row: row["modality"] in set(MOD_CLASSES),
            "label":   lambda row: row["modality"],
        }
    elif args.task == "cine_view":
        task_cfg = {
            "classes": CINE_VIEW_CLASSES,
            "filter":  lambda row: (
                row["modality"] == "Cine" and
                os.path.splitext(os.path.basename(str(row["mat_path"])))[0]
                in CINE_VIEW_MAP
            ),
            "label":   lambda row: CINE_VIEW_MAP.get(
                os.path.splitext(os.path.basename(str(row["mat_path"])))[0], ""
            ),
        }
    else:
        raise ValueError(f"Unknown task: {args.task}")

    classes = task_cfg["classes"]
    print(f"Task: {args.task}  |  Classes ({len(classes)}): {classes}")

    # ---- Datasets ----
    ds_kwargs = dict(
        task_cfg=task_cfg,
        image_size=args.image_size,
        normalize_mode="zscore",
        num_slices=args.num_slices,
        num_frames=args.num_frames,
    )
    print("\nTrain dataset:")
    train_ds = ClassificationDataset(args.train_csv, train=True,  **ds_kwargs)
    print("Test dataset:")
    test_ds  = ClassificationDataset(args.test_csv,  train=False, **ds_kwargs)

    # Balanced sampler for class imbalance (especially sax vs 2ch/3ch/4ch)
    sampler     = make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False,
    )

    print(f"\nTrain batches/epoch: {len(train_loader)}")

    # ---- Model ----
    model = ClassificationModel(
        num_classes=len(classes),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        backbone_name=args.backbone,
    ).to(device)

    model.load_pretrained_encoder(args.ckpt_path, device)
    model.freeze_encoder()
    ckpt_label = args.ckpt_path if args.ckpt_path else "ImageNet only"
    print(f"Encoder frozen for linear probe stage.  [{ckpt_label}]")

    # ---- Stage 1: Linear probe ----
    head_params = list(model.head.parameters())
    optimizer   = torch.optim.AdamW(head_params, lr=args.lr_head,
                                    weight_decay=args.weight_decay)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.probe_epochs, eta_min=1e-6,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    local_ckpt  = tempfile.mkdtemp(prefix=f"cls_{args.task}_")
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

    best_acc   = 0.0
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"Stage 1: Linear probe ({args.probe_epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, args.probe_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device,
                                  epoch, amp=args.amp)
        eval_m  = evaluate(model, test_loader, device, classes)
        scheduler.step()

        is_best = eval_m["acc"] > best_acc
        if is_best:
            best_acc   = eval_m["acc"]
            no_improve = 0
        else:
            no_improve += 1

        print(f"[Probe {epoch:3d}] train loss={train_m['loss']:.4f} "
              f"acc={train_m['acc']:.4f} | "
              f"test acc={eval_m['acc']:.4f}"
              + (" ← best" if is_best else f"  (no improve {no_improve}/{args.patience})"))

        local_last = os.path.join(local_ckpt, "probe_last.pt")
        save_checkpoint(local_last, model, optimizer, scheduler, epoch, best_acc, args)
        copy_threads.append(_async_copy(
            local_last, os.path.join(args.out_dir, "probe_last.pt")))

        if is_best:
            local_best = os.path.join(local_ckpt, "probe_best.pt")
            shutil.copy2(local_last, local_best)
            copy_threads.append(_async_copy(
                local_best, os.path.join(args.out_dir, "probe_best.pt")))

        if no_improve >= args.patience:
            print(f"Early stopping at probe epoch {epoch}.")
            break

    # ---- Stage 1 final evaluation ----
    best_ckpt = torch.load(os.path.join(local_ckpt, "probe_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    probe_eval = evaluate(model, test_loader, device, classes)

    print(f"\n{'='*60}")
    print(f"Linear probe best test acc: {probe_eval['acc']:.4f}")
    print(f"{'='*60}")
    print(probe_eval["report"])
    print("Confusion matrix:")
    cm_df = pd.DataFrame(probe_eval["cm"], index=classes, columns=classes)
    print(cm_df.to_string())

    # ---- Stage 2: Fine-tuning (optional) ----
    if args.finetune:
        print(f"\n{'='*60}")
        print(f"Stage 2: Fine-tuning ({args.finetune_epochs} epochs)")
        print(f"{'='*60}")

        model.unfreeze_encoder()
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(),    "lr": args.lr_head},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.finetune_epochs, eta_min=1e-7,
        )

        best_ft_acc = probe_eval["acc"]
        no_improve  = 0

        for epoch in range(1, args.finetune_epochs + 1):
            train_m = train_one_epoch(model, train_loader, optimizer, device,
                                      epoch, amp=args.amp)
            eval_m  = evaluate(model, test_loader, device, classes)
            scheduler.step()

            is_best = eval_m["acc"] > best_ft_acc
            if is_best:
                best_ft_acc = eval_m["acc"]
                no_improve  = 0
            else:
                no_improve += 1

            print(f"[FT {epoch:3d}] train loss={train_m['loss']:.4f} "
                  f"acc={train_m['acc']:.4f} | "
                  f"test acc={eval_m['acc']:.4f}"
                  + (" ← best" if is_best else
                     f"  (no improve {no_improve}/{args.patience})"))

            local_last = os.path.join(local_ckpt, "ft_last.pt")
            save_checkpoint(local_last, model, optimizer, scheduler, epoch, best_ft_acc, args)
            copy_threads.append(_async_copy(
                local_last, os.path.join(args.out_dir, "ft_last.pt")))

            if is_best:
                local_best = os.path.join(local_ckpt, "ft_best.pt")
                shutil.copy2(local_last, local_best)
                copy_threads.append(_async_copy(
                    local_best, os.path.join(args.out_dir, "ft_best.pt")))

            if no_improve >= args.patience:
                print(f"Early stopping at fine-tune epoch {epoch}.")
                break

        best_ckpt  = torch.load(os.path.join(local_ckpt, "ft_best.pt"), map_location=device)
        model.load_state_dict(best_ckpt["model"])
        ft_eval = evaluate(model, test_loader, device, classes)

        print(f"\n{'='*60}")
        print(f"Fine-tune best test acc: {ft_eval['acc']:.4f}  "
              f"(probe: {probe_eval['acc']:.4f}  "
              f"Δ={ft_eval['acc'] - probe_eval['acc']:+.4f})")
        print(f"{'='*60}")
        print(ft_eval["report"])
        print("Confusion matrix:")
        cm_df = pd.DataFrame(ft_eval["cm"], index=classes, columns=classes)
        print(cm_df.to_string())

    # ---- Wait for NAS sync ----
    print("\nWaiting for NAS sync ...")
    for t in copy_threads:
        t.join()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Downstream classification: modality or Cine view"
    )

    # Task
    p.add_argument("--task",      required=True, choices=["modality", "cine_view"])
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv",  required=True)
    p.add_argument("--ckpt_path", default=None,
                   help="Pre-trained checkpoint (best.pt). "
                        "Omit to use ImageNet weights only (no CMR pre-training).")
    p.add_argument("--out_dir",   required=True)

    # Model
    p.add_argument("--backbone",  default="resnet50",
                   choices=["resnet18", "resnet50"])
    p.add_argument("--embed_dim", type=int,   default=256)
    p.add_argument("--dropout",   type=float, default=0.1)

    # Image
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--num_slices",  type=int, default=3)
    p.add_argument("--num_frames",  type=int, default=4)

    # Stage 1: Linear probe
    p.add_argument("--probe_epochs", type=int,   default=20)
    p.add_argument("--lr_head",      type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=7)

    # Stage 2: Fine-tuning
    p.add_argument("--finetune",         action="store_true",
                   help="Enable Stage 2 fine-tuning after linear probe")
    p.add_argument("--finetune_epochs",  type=int,   default=15)
    p.add_argument("--lr_backbone",      type=float, default=1e-6,
                   help="Encoder lr during fine-tuning (much smaller than head)")

    # Training
    p.add_argument("--batch_size",   type=int,  default=32)
    p.add_argument("--num_workers",  type=int,  default=8)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--amp",          action="store_true")

    args = p.parse_args()
    main(args)