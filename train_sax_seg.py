"""
train_sax_seg.py
================
Cine SAX cardiac segmentation using a U-Net with pretrained ResNet50 encoder.

Supports two experimental settings:
  --dataset acdc   : single-centre (ACDC only, 100 train patients)
  --dataset mms    : multi-centre  (M&Ms only, 150 train cases, 4 vendors)
  --dataset both   : combined training on ACDC + M&Ms

Label map (both datasets):
  0 = background
  1 = right ventricle (RV)
  2 = myocardium (MYO)
  3 = left ventricle (LV)

Architecture:
  Encoder : pretrained ResNet50 (from CLIP pre-training checkpoint)
            conv1 adapted to 1-channel grayscale input
  Decoder : lightweight FPN-style decoder with skip connections
  Head    : 1x1 conv → 4 classes

Metrics (reported per epoch):
  Dice score per class (RV / MYO / LV) and mean Dice
  Hausdorff distance (optional, requires scipy)

Usage
-----
# Single-centre (ACDC)
python train_sax_seg.py \
    --dataset     acdc \
    --index_csv   /path/sax_seg_npy/index.parquet \
    --ckpt_path   /path/cmr_multimodal/best.pt \
    --out_dir     /path/runs/sax_seg_acdc \
    --amp

# Multi-centre (M&Ms)
python train_sax_seg.py \
    --dataset     mms \
    --index_csv   /path/sax_seg_npy/index.parquet \
    --ckpt_path   /path/cmr_multimodal/best.pt \
    --out_dir     /path/runs/sax_seg_mms \
    --amp
"""

import os
import math
import random
import argparse
import tempfile
import shutil
import threading
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES  = 4          # 0=BG, 1=RV, 2=MYO, 3=LV
CLASS_NAMES  = ["BG", "RV", "MYO", "LV"]
FG_CLASSES   = [1, 2, 3]  # classes used for Dice/metric (exclude BG)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SegTransform:
    """
    Joint augmentation for image + mask.
    Image : float32 [0,1]  shape (H, W)
    Mask  : int64          shape (H, W)
    Both are resized to image_size × image_size.
    """

    def __init__(
        self,
        image_size: int          = 224,
        train: bool              = True,
        random_rotate_deg: float = 15.0,
        translate: float         = 0.05,
        scale_min: float         = 0.9,
        scale_max: float         = 1.1,
        p_affine: float          = 0.5,
        p_hflip: float           = 0.3,
        p_noise: float           = 0.2,
        noise_std: float         = 0.03,
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
        self.noise_std         = noise_std

    def __call__(
        self,
        img: torch.Tensor,   # (1, H, W) float
        mask: torch.Tensor,  # (1, H, W) long
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Resize
        img  = F.interpolate(img.unsqueeze(0),  (self.image_size, self.image_size),
                             mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), (self.image_size, self.image_size),
                             mode="nearest").squeeze(0).long()

        if not self.train:
            return img.clamp(0, 1), mask

        # Shared affine
        if random.random() < self.p_affine:
            angle  = random.uniform(-self.random_rotate_deg, self.random_rotate_deg)
            max_d  = self.translate * self.image_size
            transl = (int(round(random.uniform(-max_d, max_d))),
                      int(round(random.uniform(-max_d, max_d))))
            scale  = random.uniform(self.scale_min, self.scale_max)
            img  = TF.affine(img,  angle, transl, scale, [0.0],
                             interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask.float(), angle, transl, scale, [0.0],
                             interpolation=TF.InterpolationMode.NEAREST).long()

        # Horizontal flip
        if random.random() < self.p_hflip:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # Gaussian noise (image only)
        if random.random() < self.p_noise:
            img = img + torch.randn_like(img) * self.noise_std

        return img.clamp(0, 1), mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SaxSegDataset(Dataset):
    """
    2D slice-level segmentation dataset.

    Each sample:  image (1, H, W) float32,  mask (H, W) int64
    """

    def __init__(
        self,
        index_path: str,
        dataset:    str   = "acdc",   # "acdc", "mms", "both"
        split:      str   = "train",  # "train", "val", "test"
        image_size: int   = 224,
        train:      bool  = True,
        max_cases:  int | None = None,  # limit number of training cases (few-shot)
    ):
        if index_path.endswith(".parquet"):
            df = pd.read_parquet(index_path)
        else:
            df = pd.read_csv(index_path)

        # Filter by dataset
        if dataset == "acdc":
            df = df[df["dataset"] == "ACDC"]
        elif dataset == "mms":
            df = df[df["dataset"] == "MMs"]
        elif dataset == "both":
            pass  # keep all
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Filter by split
        df = df[df["split"] == split]
        df = df[df["has_gt"] == True].reset_index(drop=True)

        # Few-shot: randomly sample max_cases cases (stratified by pathology if available)
        if max_cases is not None and split == "train":
            all_cases = df["case_id"].unique().tolist()
            if len(all_cases) > max_cases:
                import random as _random
                _random.seed(42)
                # Stratified sampling by pathology if column exists
                if "pathology" in df.columns:
                    groups = df.drop_duplicates("case_id").groupby("pathology")["case_id"].apply(list)
                    n_per_group = max(1, max_cases // len(groups))
                    selected = []
                    for g_cases in groups:
                        selected.extend(_random.sample(g_cases, min(n_per_group, len(g_cases))))
                    # Fill up to max_cases if needed
                    remaining = [c for c in all_cases if c not in selected]
                    _random.shuffle(remaining)
                    selected = selected[:max_cases] + remaining[:max(0, max_cases - len(selected))]
                    selected = selected[:max_cases]
                else:
                    selected = _random.sample(all_cases, max_cases)
                df = df[df["case_id"].isin(selected)].reset_index(drop=True)

        self.samples   = df.to_dict(orient="records")
        self.transform = SegTransform(image_size=image_size, train=train)
        self.train     = train

        # Summary
        n_cases = df["case_id"].nunique()
        suffix  = f" [max_cases={max_cases}]" if max_cases and split == "train" else ""
        print(f"  SaxSegDataset [{dataset}/{split}]{suffix}: "
              f"{len(self.samples)} slices from {n_cases} cases")
        if "pathology" in df.columns:
            pc = df["case_id"].drop_duplicates().map(
                df.drop_duplicates("case_id").set_index("case_id")["pathology"]
            ).value_counts()
            for path, cnt in pc.items():
                print(f"    {path:8s}: {cnt} cases")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img  = np.load(s["image_path"]).astype(np.float32)  # (H, W)
        mask = np.load(s["mask_path"]).astype(np.int64)      # (H, W)

        img_t  = torch.from_numpy(img).unsqueeze(0)          # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)

        img_t, mask_t = self.transform(img_t, mask_t)

        return {
            "image":      img_t,             # (1, H, W) float
            "mask":       mask_t.squeeze(0), # (H, W)    long
            "case_id":    s["case_id"],
            "frame_type": s.get("frame_type", ""),
            "slice_idx":  s.get("slice_idx", 0),
            "vendor":     s.get("vendor", ""),
            "pathology":  s.get("pathology", ""),
        }


def seg_collate(batch: list[dict]) -> dict:
    return {
        "image":      torch.stack([b["image"]  for b in batch]),
        "mask":       torch.stack([b["mask"]   for b in batch]),
        "case_id":    [b["case_id"]    for b in batch],
        "frame_type": [b["frame_type"] for b in batch],
        "slice_idx":  [b["slice_idx"]  for b in batch],
        "vendor":     [b["vendor"]     for b in batch],
        "pathology":  [b["pathology"]  for b in batch],
    }


# ---------------------------------------------------------------------------
# Model: ResNet50 encoder + FPN decoder
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample + skip connection + 2× ConvBnRelu."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Pad if sizes don't match exactly
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet50UNet(nn.Module):
    """
    U-Net with ResNet50 encoder.

    Encoder produces feature maps at 4 scales:
        layer1 : stride 4   → 256 ch
        layer2 : stride 8   → 512 ch
        layer3 : stride 16  → 1024 ch
        layer4 : stride 32  → 2048 ch  (bottleneck)

    Decoder upsamples back to input resolution with skip connections.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()

        # ---- Encoder (ResNet50) ----
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Adapt conv1 for 1-channel grayscale
        old = backbone.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                   backbone.relu)          # /2,  64 ch
        self.pool  = backbone.maxpool                       # /4
        self.enc1  = backbone.layer1                        # /4,  256 ch
        self.enc2  = backbone.layer2                        # /8,  512 ch
        self.enc3  = backbone.layer3                        # /16, 1024 ch
        self.enc4  = backbone.layer4                        # /32, 2048 ch

        # ---- Decoder ----
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512,  512,  256)
        self.dec2 = DecoderBlock(256,  256,  128)
        self.dec1 = DecoderBlock(128,  64,   64)
        self.dec0 = DecoderBlock(64,   0,    32)   # no skip at this level

        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)       # (B, 64,  H/2,  W/2)
        e1 = self.enc1(self.pool(e0))  # (B, 256,  H/4,  W/4)
        e2 = self.enc2(e1)       # (B, 512,  H/8,  W/8)
        e3 = self.enc3(e2)       # (B, 1024, H/16, W/16)
        e4 = self.enc4(e3)       # (B, 2048, H/32, W/32)

        # Decoder with skip connections
        d4 = self.dec4(e4, e3)   # (B, 512,  H/16, W/16)
        d3 = self.dec3(d4, e2)   # (B, 256,  H/8,  W/8)
        d2 = self.dec2(d3, e1)   # (B, 128,  H/4,  W/4)
        d1 = self.dec1(d2, e0)   # (B, 64,   H/2,  W/2)
        d0 = self.dec0(d1)       # (B, 32,   H,    W)

        return self.head(self.drop(d0))   # (B, num_classes, H, W)

    def load_pretrained_encoder(self, ckpt_path: str, device):
        """
        Load image encoder weights from a pre-training checkpoint.
        Supports two checkpoint formats automatically:
          - CLIP format : keys start with 'image_encoder.backbone.'
          - MIM  format : keys start with 'encoder.'
        """
        ckpt       = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"]

        # ---- Detect checkpoint format ----
        has_clip = any(k.startswith("image_encoder.backbone.") for k in state_dict)
        has_mim  = any(k.startswith("encoder.") for k in state_dict)

        if has_clip:
            print("  Detected CLIP checkpoint")
            enc_state = {
                k.replace("image_encoder.backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("image_encoder.backbone.")
            }
        elif has_mim:
            print("  Detected MIM checkpoint")
            enc_state = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
        else:
            raise ValueError(
                f"Unknown checkpoint format in {ckpt_path}. "
                "Expected keys starting with 'image_encoder.backbone.' (CLIP) "
                "or 'encoder.' (MIM)."
            )

        # Build a temporary ResNet50 to load state then extract layers
        tmp = models.resnet50()
        old_conv = tmp.conv1
        new_conv = nn.Conv2d(1, old_conv.out_channels, old_conv.kernel_size,
                             old_conv.stride, old_conv.padding, bias=False)
        tmp.conv1 = new_conv
        missing, unexpected = tmp.load_state_dict(enc_state, strict=False)
        print(f"  Encoder load: {len(missing)} missing, {len(unexpected)} unexpected")

        # Copy weights into our encoder modules
        with torch.no_grad():
            self.enc0[0].weight.copy_(tmp.conv1.weight)
            self.enc0[1].weight.copy_(tmp.bn1.weight)
            self.enc0[1].bias.copy_(tmp.bn1.bias)
            self.enc0[1].running_mean.copy_(tmp.bn1.running_mean)
            self.enc0[1].running_var.copy_(tmp.bn1.running_var)
            for name in ["enc1", "enc2", "enc3", "enc4"]:
                layer_name = name.replace("enc", "layer")
                getattr(self, name).load_state_dict(
                    getattr(tmp, layer_name).state_dict()
                )
        print(f"  Pretrained encoder loaded from {ckpt_path}")
        return self

    def freeze_encoder(self):
        for mod in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        for mod in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in mod.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Loss: Dice + Cross-Entropy
# ---------------------------------------------------------------------------

def dice_loss(
    logits: torch.Tensor,   # (B, C, H, W)
    targets: torch.Tensor,  # (B, H, W) long
    smooth: float = 1.0,
    ignore_bg: bool = True,
) -> torch.Tensor:
    probs   = F.softmax(logits, dim=1)
    C       = logits.size(1)
    start_c = 1 if ignore_bg else 0
    losses  = []
    for c in range(start_c, C):
        p = probs[:, c]               # (B, H, W)
        t = (targets == c).float()    # (B, H, W)
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        losses.append(1.0 - (2 * inter + smooth) / (union + smooth))
    return torch.stack(losses).mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float   = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    ce   = F.cross_entropy(logits, targets)
    dice = dice_loss(logits, targets)
    return ce_weight * ce + dice_weight * dice


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dice(
    logits: torch.Tensor,    # (B, C, H, W)
    targets: torch.Tensor,   # (B, H, W) long
) -> dict[str, float]:
    preds = logits.argmax(dim=1)   # (B, H, W)
    dices = {}
    for c in FG_CLASSES:
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dices[CLASS_NAMES[c]] = (2 * inter / (union + 1e-6)).item()
    dices["mean"] = sum(dices[CLASS_NAMES[c]] for c in FG_CLASSES) / len(FG_CLASSES)
    return dices


# ---------------------------------------------------------------------------
# Case-level Dice (aggregate predictions per case before computing Dice)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_by_case(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    return_per_case: bool = False,   # ← 新增参数
) -> dict:
    model.eval()

    case_preds:   dict[str, list] = defaultdict(list)
    case_targets: dict[str, list] = defaultdict(list)
    running_loss  = 0.0
    count         = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device,  non_blocking=True)
        case_keys = batch["case_id"]

        logits = model(images)
        loss   = combined_loss(logits, targets)
        running_loss += loss.item() * images.size(0)
        count        += images.size(0)

        preds = logits.argmax(dim=1).cpu()
        targets_cpu = targets.cpu()
        for i, ck in enumerate(case_keys):
            case_preds[ck].append(preds[i])
            case_targets[ck].append(targets_cpu[i])

    per_class_dices: dict[str, list] = defaultdict(list)
    per_case_records: list = []   # ← 新增：逐 case 记录

    for ck in case_preds:
        pred_vol   = torch.stack(case_preds[ck],   dim=0)
        target_vol = torch.stack(case_targets[ck], dim=0)
        case_row = {"case_id": ck}
        for c in FG_CLASSES:
            p = (pred_vol   == c).float()
            t = (target_vol == c).float()
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            d = (2 * inter / (union + 1e-6)).item()
            per_class_dices[CLASS_NAMES[c]].append(d)
            case_row[f"dice_{CLASS_NAMES[c]}"] = d
        case_row["dice_mean"] = sum(
            case_row[f"dice_{CLASS_NAMES[c]}"] for c in FG_CLASSES
        ) / len(FG_CLASSES)
        per_case_records.append(case_row)   # ← 新增

    result = {"loss": running_loss / max(count, 1)}
    for c in FG_CLASSES:
        name = CLASS_NAMES[c]
        vals = per_class_dices[name]
        result[f"dice_{name}"] = float(np.mean(vals)) if vals else 0.0
    result["dice_mean"] = float(np.mean([
        result[f"dice_{CLASS_NAMES[c]}"] for c in FG_CLASSES
    ]))

    if return_per_case:
        result["per_case"] = per_case_records   # ← 新增

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    epoch:       int,
    amp:         bool = True,
    accum_steps: int  = 1,
) -> dict:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    running_loss  = 0.0
    running_dices = defaultdict(float)
    count = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch}")
    for step, batch in enumerate(pbar):
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device,  non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(images)
            loss   = combined_loss(logits, targets) / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs            = images.size(0)
        dices         = compute_dice(logits.detach(), targets)
        running_loss += loss.item() * accum_steps * bs
        for k, v in dices.items():
            running_dices[k] += v * bs
        count += bs

        pbar.set_postfix({
            "loss":      f"{running_loss / count:.4f}",
            "dice_mean": f"{running_dices['mean'] / count:.4f}",
        })

    result = {"loss": running_loss / max(count, 1)}
    for k in running_dices:
        result[f"dice_{k}"] = running_dices[k] / max(count, 1)
    return result


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_dice": best_dice, "args": vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Dataset: {args.dataset}")

    # ---- Datasets ----
    # For M&Ms: train on Labeled, validate on Validation split
    # For ACDC : use official train/test split (no validation split in ACDC)
    val_split = "val" if args.dataset == "mms" else "test"

    print("\nTrain dataset:")
    train_ds = SaxSegDataset(
        args.index_csv, dataset=args.dataset, split="train",
        image_size=args.image_size, train=True,
        max_cases=args.max_cases,
    )
    print(f"\n{'Val' if val_split == 'val' else 'Test'} dataset:")
    val_ds = SaxSegDataset(
        args.index_csv, dataset=args.dataset, split=val_split,
        image_size=args.image_size, train=False,
    )
    print(f"\nTest dataset:")
    test_ds = SaxSegDataset(
        args.index_csv, dataset=args.dataset, split="test",
        image_size=args.image_size, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=seg_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=seg_collate, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=seg_collate, drop_last=False,
    )

    print(f"\nTrain batches: {len(train_loader)}")

    # ---- Model ----
    model = ResNet50UNet(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)

    if args.ckpt_path:
        model.load_pretrained_encoder(args.ckpt_path, device)
    else:
        print("  No pretrained checkpoint — training from scratch (ImageNet init)")

    # Stage 1: freeze encoder, train decoder only
    model.freeze_encoder()
    print(f"Encoder frozen for {args.probe_epochs} probe epoch(s).")

    decoder_params = [
        p for name, p in model.named_parameters()
        if not any(name.startswith(e) for e in ["enc0", "enc1", "enc2", "enc3", "enc4"])
    ]
    optimizer = torch.optim.AdamW(
        decoder_params, lr=args.lr_head, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.probe_epochs, eta_min=1e-6,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    local_ckpt    = tempfile.mkdtemp(prefix=f"sax_seg_{args.dataset}_")
    copy_threads: list = []

    def _async_copy(src, dst):
        def _do():
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[warn] NAS copy: {e}")
        t = threading.Thread(target=_do, daemon=True)
        t.start()
        return t

    best_dice  = 0.0
    no_improve = 0

    def _run_stage(stage_name, n_epochs, patience):
        nonlocal best_dice, no_improve, optimizer, scheduler
        no_improve = 0

        print(f"\n{'='*60}")
        print(f"{stage_name}  ({n_epochs} epochs)")
        print(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            train_m = train_one_epoch(
                model, train_loader, optimizer, device, epoch,
                amp=args.amp, accum_steps=args.accum_steps,
            )
            val_m   = evaluate_by_case(model, val_loader, device)
            scheduler.step()

            is_best = val_m["dice_mean"] > best_dice
            if is_best:
                best_dice  = val_m["dice_mean"]
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"[{stage_name} {epoch:3d}] "
                f"train loss={train_m['loss']:.4f} "
                f"dice={train_m.get('dice_mean', 0):.4f} | "
                f"val loss={val_m['loss']:.4f} "
                f"RV={val_m['dice_RV']:.4f} "
                f"MYO={val_m['dice_MYO']:.4f} "
                f"LV={val_m['dice_LV']:.4f} "
                f"mean={val_m['dice_mean']:.4f}"
                + (" ← best" if is_best
                   else f"  (no improve {no_improve}/{patience})")
            )

            local_last = os.path.join(local_ckpt, "last.pt")
            save_checkpoint(local_last, model, optimizer, scheduler,
                            epoch, best_dice, args)
            copy_threads.append(
                _async_copy(local_last, os.path.join(args.out_dir, "last.pt"))
            )
            if is_best:
                local_best = os.path.join(local_ckpt, "best.pt")
                shutil.copy2(local_last, local_best)
                copy_threads.append(
                    _async_copy(local_best, os.path.join(args.out_dir, "best.pt"))
                )

            if no_improve >= patience:
                print(f"Early stopping at {stage_name} epoch {epoch}.")
                break

    # ---- Stage 1: Linear probe (decoder only) ----
    _run_stage("Probe", args.probe_epochs, args.patience)

    # ---- Stage 2: Fine-tune (full model) ----
    print("\nUnfreezing encoder for fine-tuning ...")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if any(n.startswith(e) for e in ["enc0","enc1","enc2","enc3","enc4"])],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if not any(n.startswith(e) for e in ["enc0","enc1","enc2","enc3","enc4"])],
         "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-7,
    )
    _run_stage("FT", args.finetune_epochs, args.patience)

    # ---- Wait for NAS sync ----
    print("\nWaiting for NAS sync ...")
    for t in copy_threads:
        t.join()

    # ---- Final test evaluation ----
    print("\nFinal evaluation on test set (best checkpoint):")
    best_ckpt = torch.load(os.path.join(local_ckpt, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_m = evaluate_by_case(model, test_loader, device)
    print(
        f"Test  RV={test_m['dice_RV']:.4f}  "
        f"MYO={test_m['dice_MYO']:.4f}  "
        f"LV={test_m['dice_LV']:.4f}  "
        f"mean={test_m['dice_mean']:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Cine SAX segmentation: ACDC (single-centre) or M&Ms (multi-centre)"
    )

    p.add_argument("--dataset",    required=True, choices=["acdc", "mms", "both"],
                   help="acdc = single-centre, mms = multi-centre, both = combined")
    p.add_argument("--index_csv",  required=True,
                   help="Path to index CSV/parquet from preprocess_sax_seg.py")
    p.add_argument("--ckpt_path",  default=None,
                   help="Pre-trained CLIP checkpoint (best.pt). "
                        "If not set, uses ImageNet init only.")
    p.add_argument("--out_dir",    required=True)

    # Model
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--image_size", type=int,   default=224)

    # Stage 1: probe
    p.add_argument("--probe_epochs",  type=int,   default=10)
    p.add_argument("--lr_head",       type=float, default=1e-3)

    # Stage 2: fine-tune
    p.add_argument("--finetune_epochs", type=int,   default=20)
    p.add_argument("--lr_backbone",     type=float, default=1e-5)

    # Shared
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=7)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--accum_steps",  type=int,   default=1)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--max_cases",    type=int,  default=None,
                   help="Limit training to N cases (few-shot). "
                        "Stratified by pathology if available. Default: use all cases.")

    args = p.parse_args()
    main(args)