"""
train_emidec.py
===============
Two downstream tasks on EMIDEC LGE data:

Task A — LGE SAX segmentation (--task seg):
    Input : LGE SAX slice (1, H, W)
    Output: 4-class mask — 0=BG, 1=LV, 2=normal myo, 3=scar
    Model : ResNet50 U-Net (encoder from pre-trained CLIP)
    Eval  : case-level 3D Dice per class

Task B — LGE detection (--task det):
    Input : LGE SAX slice (1, H, W)
    Output: binary label — 0=Normal, 1=MI patient
    Model : ResNet50 encoder + classification head
    Eval  : case-level accuracy (majority vote over slices)
            + AUC, sensitivity, specificity

Both tasks:
    - Stage 1: freeze encoder, train head/decoder (linear probe)
    - Stage 2: unfreeze encoder, fine-tune end-to-end

Usage
-----
# Segmentation
python train_emidec.py \
    --task      seg \
    --index_csv /path/emidec_npy/index_seg.parquet \
    --ckpt_path /path/cmr_multimodal/best.pt \
    --out_dir   /path/runs/emidec_seg \
    --amp

# Detection
python train_emidec.py \
    --task      det \
    --index_csv /path/emidec_npy/index_det.parquet \
    --ckpt_path /path/cmr_multimodal/best.pt \
    --out_dir   /path/runs/emidec_det \
    --amp
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
from sklearn.metrics import roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Segmentation
SEG_NUM_CLASSES = 4
SEG_CLASS_NAMES = ["BG", "LV", "MYO", "scar"]
SEG_FG_CLASSES  = [1, 2, 3]

# Scar class weight boost (compensate for 0.3% voxel frequency)
SEG_CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 1.0, 5.0])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_zscore(img: np.ndarray) -> np.ndarray:
    img  = img.astype(np.float32)
    mean, std = float(img.mean()), float(img.std())
    img  = (img - mean) / std if std > 1e-6 else img - mean
    return np.clip((img + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SegTransform:
    """Joint image + mask augmentation. Shared geometry, image-only noise."""

    def __init__(
        self,
        image_size: int = 224,
        train: bool     = True,
        rotate_deg: float = 15.0,
        translate: float  = 0.05,
        scale_range: tuple = (0.9, 1.1),
        p_affine: float   = 0.5,
        p_hflip: float    = 0.3,
        p_noise: float    = 0.2,
        noise_std: float  = 0.03,
    ):
        self.image_size  = image_size
        self.train       = train
        self.rotate_deg  = rotate_deg
        self.translate   = translate
        self.scale_range = scale_range
        self.p_affine    = p_affine
        self.p_hflip     = p_hflip
        self.p_noise     = p_noise
        self.noise_std   = noise_std

    def __call__(self, img, mask=None):
        # Resize
        img = F.interpolate(img.unsqueeze(0), (self.image_size, self.image_size),
                            mode="bilinear", align_corners=False).squeeze(0)
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(0).float(),
                                 (self.image_size, self.image_size),
                                 mode="nearest").squeeze(0).long()
        if not self.train:
            return (img.clamp(0, 1), mask) if mask is not None else img.clamp(0, 1)

        if random.random() < self.p_affine:
            angle  = random.uniform(-self.rotate_deg, self.rotate_deg)
            max_d  = self.translate * self.image_size
            transl = (int(round(random.uniform(-max_d, max_d))),
                      int(round(random.uniform(-max_d, max_d))))
            scale  = random.uniform(*self.scale_range)
            img = TF.affine(img, angle, transl, scale, [0.0],
                            interpolation=TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.affine(mask.float(), angle, transl, scale, [0.0],
                                 interpolation=TF.InterpolationMode.NEAREST).long()

        if random.random() < self.p_hflip:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)

        if random.random() < self.p_noise:
            img = img + torch.randn_like(img) * self.noise_std

        img = img.clamp(0, 1)
        return (img, mask) if mask is not None else img


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class EmidecSegDataset(Dataset):
    """LGE SAX segmentation — P cases only."""

    def __init__(self, index_path: str, split: str = "train",
                 image_size: int = 224, train: bool = True,
                 max_cases: int | None = None):
        df = pd.read_parquet(index_path) if index_path.endswith(".parquet") \
             else pd.read_csv(index_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_cases is not None and split == "train":
            import random as _r; _r.seed(42)
            cases = df["case_id"].unique().tolist()
            if len(cases) > max_cases:
                sel = _r.sample(cases, max_cases)
                df  = df[df["case_id"].isin(sel)].reset_index(drop=True)
        self.samples   = df.to_dict(orient="records")
        self.transform = SegTransform(image_size=image_size, train=train)
        n_cases = df["case_id"].nunique()
        suf = f" [max_cases={max_cases}]" if max_cases and split=="train" else ""
        print(f"  EmidecSeg [{split}]{suf}: {len(self.samples)} slices, {n_cases} cases")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img  = torch.from_numpy(np.load(s["image_path"]).astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.load(s["mask_path"]).astype(np.int64)).unsqueeze(0)
        img, mask = self.transform(img, mask)
        return {"image": img, "mask": mask.squeeze(0),
                "case_id": s["case_id"], "slice_idx": s["slice_idx"]}


class EmidecDetDataset(Dataset):
    """LGE detection — N (label=0) vs P (label=1), all cases."""

    def __init__(self, index_path: str, split: str = "train",
                 image_size: int = 224, train: bool = True):
        df = pd.read_parquet(index_path) if index_path.endswith(".parquet") \
             else pd.read_csv(index_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.samples   = df.to_dict(orient="records")
        self.transform = SegTransform(image_size=image_size, train=train)
        n_cases = df["case_id"].nunique()
        label_dist = df["detection_label"].value_counts().to_dict()
        print(f"  EmidecDet [{split}]: {len(self.samples)} slices, "
              f"{n_cases} cases  labels={label_dist}")

    def __len__(self): return len(self.samples)

    def get_labels(self): return [s["detection_label"] for s in self.samples]

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = torch.from_numpy(np.load(s["image_path"]).astype(np.float32)).unsqueeze(0)
        img = self.transform(img)
        return {"image": img, "label": int(s["detection_label"]),
                "case_id": s["case_id"], "slice_idx": s["slice_idx"]}


def seg_collate(batch):
    return {"image":     torch.stack([b["image"] for b in batch]),
            "mask":      torch.stack([b["mask"]  for b in batch]),
            "case_id":   [b["case_id"]   for b in batch],
            "slice_idx": [b["slice_idx"] for b in batch]}

def det_collate(batch):
    return {"image":     torch.stack([b["image"] for b in batch]),
            "label":     torch.tensor([b["label"] for b in batch]),
            "case_id":   [b["case_id"]   for b in batch],
            "slice_idx": [b["slice_idx"] for b in batch]}


def make_balanced_sampler(dataset: EmidecDetDataset) -> WeightedRandomSampler:
    labels  = dataset.get_labels()
    counts  = np.bincount(labels, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    w       = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(ConvBnRelu(in_ch + skip_ch, out_ch),
                                  ConvBnRelu(out_ch, out_ch))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _build_resnet50_backbone():
    """Return ResNet50 with 1-channel grayscale conv1."""
    bb  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    old = bb.conv1
    new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    bb.conv1 = new
    return bb


class ResNet50UNet(nn.Module):
    """ResNet50 encoder + FPN decoder for segmentation."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        bb         = _build_resnet50_backbone()
        self.enc0  = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool  = bb.maxpool
        self.enc1  = bb.layer1
        self.enc2  = bb.layer2
        self.enc3  = bb.layer3
        self.enc4  = bb.layer4
        self.dec4  = DecoderBlock(2048, 1024, 512)
        self.dec3  = DecoderBlock(512,   512, 256)
        self.dec2  = DecoderBlock(256,   256, 128)
        self.dec1  = DecoderBlock(128,    64,  64)
        self.dec0  = DecoderBlock( 64,     0,  32)
        self.drop  = nn.Dropout2d(dropout)
        self.head  = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d  = self.dec0(self.dec1(self.dec2(self.dec3(self.dec4(e4, e3), e2), e1), e0))
        return self.head(self.drop(d))

    def encoder_modules(self):
        return [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]

    def freeze_encoder(self):
        for m in self.encoder_modules():
            for p in m.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for m in self.encoder_modules():
            for p in m.parameters(): p.requires_grad = True

    def load_pretrained(self, ckpt_path, device):
        ckpt     = torch.load(ckpt_path, map_location=device)
        full_sd  = ckpt["model"]
        has_clip = any(k.startswith("image_encoder.backbone.") for k in full_sd)
        has_mim  = any(k.startswith("encoder.") for k in full_sd)
        if has_clip:
            print("  Detected CLIP checkpoint")
            sd = {k.replace("image_encoder.backbone.", ""): v
                  for k, v in full_sd.items()
                  if k.startswith("image_encoder.backbone.")}
        elif has_mim:
            print("  Detected MIM checkpoint")
            sd = {k.replace("encoder.", ""): v
                  for k, v in full_sd.items()
                  if k.startswith("encoder.")}
        else:
            raise ValueError(f"Unknown checkpoint format in {ckpt_path}")
        tmp  = _build_resnet50_backbone()
        miss, unex = tmp.load_state_dict(sd, strict=False)
        print(f"  Encoder load: {len(miss)} missing, {len(unex)} unexpected")
        with torch.no_grad():
            self.enc0[0].weight.copy_(tmp.conv1.weight)
            self.enc0[1].weight.copy_(tmp.bn1.weight)
            self.enc0[1].bias.copy_(tmp.bn1.bias)
            self.enc0[1].running_mean.copy_(tmp.bn1.running_mean)
            self.enc0[1].running_var.copy_(tmp.bn1.running_var)
            for name in ["enc1","enc2","enc3","enc4"]:
                getattr(self, name).load_state_dict(
                    getattr(tmp, name.replace("enc","layer")).state_dict())
        print(f"  Pretrained encoder loaded from {ckpt_path}")


class ResNet50Classifier(nn.Module):
    """ResNet50 encoder + global pool + MLP head for detection."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        bb         = _build_resnet50_backbone()
        self.enc0  = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool  = bb.maxpool
        self.enc1  = bb.layer1
        self.enc2  = bb.layer2
        self.enc3  = bb.layer3
        self.enc4  = bb.layer4
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        e = self.enc4(self.enc3(self.enc2(self.enc1(self.pool(self.enc0(x))))))
        return self.head(self.gap(e))

    def encoder_modules(self):
        return [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]

    def freeze_encoder(self):
        for m in self.encoder_modules():
            for p in m.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for m in self.encoder_modules():
            for p in m.parameters(): p.requires_grad = True

    def load_pretrained(self, ckpt_path, device):
        ckpt     = torch.load(ckpt_path, map_location=device)
        full_sd  = ckpt["model"]
        has_clip = any(k.startswith("image_encoder.backbone.") for k in full_sd)
        has_mim  = any(k.startswith("encoder.") for k in full_sd)
        if has_clip:
            print("  Detected CLIP checkpoint")
            sd = {k.replace("image_encoder.backbone.", ""): v
                  for k, v in full_sd.items()
                  if k.startswith("image_encoder.backbone.")}
        elif has_mim:
            print("  Detected MIM checkpoint")
            sd = {k.replace("encoder.", ""): v
                  for k, v in full_sd.items()
                  if k.startswith("encoder.")}
        else:
            raise ValueError(f"Unknown checkpoint format in {ckpt_path}")
        tmp  = _build_resnet50_backbone()
        miss, unex = tmp.load_state_dict(sd, strict=False)
        print(f"  Encoder load: {len(miss)} missing, {len(unex)} unexpected")
        with torch.no_grad():
            self.enc0[0].weight.copy_(tmp.conv1.weight)
            self.enc0[1].weight.copy_(tmp.bn1.weight)
            self.enc0[1].bias.copy_(tmp.bn1.bias)
            self.enc0[1].running_mean.copy_(tmp.bn1.running_mean)
            self.enc0[1].running_var.copy_(tmp.bn1.running_var)
            for name in ["enc1","enc2","enc3","enc4"]:
                getattr(self, name).load_state_dict(
                    getattr(tmp, name.replace("enc","layer")).state_dict())
        print(f"  Pretrained encoder loaded from {ckpt_path}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(logits, targets, class_weights=None, smooth=1.0):
    probs  = F.softmax(logits, dim=1)
    C      = logits.size(1)
    losses = []
    for c in range(1, C):
        p     = probs[:, c]
        t     = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        d     = 1.0 - (2 * inter + smooth) / (union + smooth)
        w     = class_weights[c].item() if class_weights is not None else 1.0
        losses.append(d * w)
    return torch.stack(losses).mean()


def seg_loss(logits, targets, class_weights, device):
    cw = class_weights.to(device)
    ce = F.cross_entropy(logits, targets, weight=cw)
    dc = dice_loss(logits, targets, class_weights=cw)
    return 0.5 * ce + 0.5 * dc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_seg_dice(logits, targets):
    preds  = logits.argmax(dim=1)
    result = {}
    for c in SEG_FG_CLASSES:
        p = (preds == c).float()
        t = (targets == c).float()
        result[SEG_CLASS_NAMES[c]] = (
            2 * (p * t).sum() / (p.sum() + t.sum() + 1e-6)
        ).item()
    result["mean"] = np.mean(list(result.values()))
    return result


@torch.no_grad()
def evaluate_seg(model, loader, device):
    """Case-level 3D Dice aggregation."""
    model.eval()
    case_preds, case_targets = defaultdict(list), defaultdict(list)
    total_loss, count = 0.0, 0

    for batch in tqdm(loader, desc="Eval seg", leave=False):
        imgs    = batch["image"].to(device)
        targets = batch["mask"].to(device)
        logits  = model(imgs)
        loss    = seg_loss(logits, targets, SEG_CLASS_WEIGHTS, device)
        total_loss += loss.item() * imgs.size(0)
        count      += imgs.size(0)
        preds = logits.argmax(dim=1).cpu()
        for i, ck in enumerate(batch["case_id"]):
            case_preds[ck].append(preds[i])
            case_targets[ck].append(targets[i].cpu())

    per_class = defaultdict(list)
    for ck in case_preds:
        pv = torch.stack(case_preds[ck])
        tv = torch.stack(case_targets[ck])
        for c in SEG_FG_CLASSES:
            p = (pv == c).float(); t = (tv == c).float()
            d = (2 * (p * t).sum() / (p.sum() + t.sum() + 1e-6)).item()
            per_class[SEG_CLASS_NAMES[c]].append(d)

    result = {"loss": total_loss / max(count, 1)}
    for c in SEG_FG_CLASSES:
        result[f"dice_{SEG_CLASS_NAMES[c]}"] = float(np.mean(per_class[SEG_CLASS_NAMES[c]]))
    result["dice_mean"] = float(np.mean([result[f"dice_{SEG_CLASS_NAMES[c]}"]
                                          for c in SEG_FG_CLASSES]))
    return result


@torch.no_grad()
def evaluate_det(model, loader, device):
    """
    Slice-level forward pass → case-level majority vote.
    Reports: accuracy, AUC, sensitivity, specificity.
    """
    model.eval()
    case_probs:  dict[str, list] = defaultdict(list)
    case_labels: dict[str, int]  = {}
    total_loss, count = 0.0, 0

    for batch in tqdm(loader, desc="Eval det", leave=False):
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        loss   = F.cross_entropy(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        count      += imgs.size(0)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()
        for i, ck in enumerate(batch["case_id"]):
            case_probs[ck].append(probs[i])
            case_labels[ck] = batch["label"][i].item()

    # Case-level aggregation: mean probability → threshold 0.5
    case_ids = sorted(case_probs.keys())
    y_true   = [case_labels[ck]            for ck in case_ids]
    y_prob   = [float(np.mean(case_probs[ck])) for ck in case_ids]
    y_pred   = [1 if p >= 0.5 else 0      for p in y_prob]

    acc  = float(np.mean([p == t for p, t in zip(y_pred, y_true)]))
    auc  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / max(tp + fn, 1)   # recall for MI class
    spec = tn / max(tn + fp, 1)

    return {
        "loss": total_loss / max(count, 1),
        "acc":  acc, "auc": auc,
        "sens": sens, "spec": spec,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch_seg(model, loader, optimizer, device, epoch, amp, accum):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    running_loss, running_dice, count = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for step, batch in enumerate(pbar):
        imgs    = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device,  non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(imgs)
            loss   = seg_loss(logits, targets, SEG_CLASS_WEIGHTS, device) / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        bs           = imgs.size(0)
        d            = compute_seg_dice(logits.detach(), targets)
        running_loss += loss.item() * accum * bs
        running_dice += d["mean"] * bs
        count        += bs
        pbar.set_postfix({"loss": f"{running_loss/count:.4f}",
                          "dice": f"{running_dice/count:.4f}"})
    return {"loss": running_loss/count, "dice": running_dice/count}


def train_one_epoch_det(model, loader, optimizer, device, epoch, amp, accum):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    running_loss, correct, count = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for step, batch in enumerate(pbar):
        imgs   = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels) / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        bs           = imgs.size(0)
        running_loss += loss.item() * accum * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        count        += bs
        pbar.set_postfix({"loss": f"{running_loss/count:.4f}",
                          "acc":  f"{correct/count:.4f}"})
    return {"loss": running_loss/count, "acc": correct/count}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_ckpt(path, model, optimizer, scheduler, epoch, best_metric, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_metric": best_metric, "args": vars(args)}, path)


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def run_training(
    model, train_loader, val_loader, test_loader,
    args, device, task,
    eval_fn, train_fn,
    metric_key, higher_is_better=True,
):
    os.makedirs(args.out_dir, exist_ok=True)
    local_ckpt    = tempfile.mkdtemp(prefix=f"emidec_{task}_")
    copy_threads  = []

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

    best_metric = -1e9 if higher_is_better else 1e9
    no_improve  = 0

    def _is_better(current):
        return current > best_metric if higher_is_better else current < best_metric

    def _run_stage(stage_name, n_epochs, optimizer, scheduler, patience):
        nonlocal best_metric, no_improve
        no_improve = 0
        print(f"\n{'='*60}")
        print(f"{stage_name}  ({n_epochs} epochs)")
        print(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            train_m = train_fn(model, train_loader, optimizer, device,
                               epoch, args.amp, args.accum_steps)
            val_m   = eval_fn(model, val_loader, device)
            if scheduler: scheduler.step()

            val_key = val_m[metric_key]
            is_best = _is_better(val_key)
            if is_best:
                best_metric = val_key
                no_improve  = 0
            else:
                no_improve += 1

            # Print
            if task == "seg":
                print(f"[{stage_name} {epoch:3d}] "
                      f"train loss={train_m['loss']:.4f} dice={train_m['dice']:.4f} | "
                      f"val loss={val_m['loss']:.4f} "
                      f"LV={val_m['dice_LV']:.4f} "
                      f"MYO={val_m['dice_MYO']:.4f} "
                      f"scar={val_m['dice_scar']:.4f} "
                      f"mean={val_m['dice_mean']:.4f}"
                      + (" ← best" if is_best
                         else f"  (no improve {no_improve}/{patience})"))
            else:
                print(f"[{stage_name} {epoch:3d}] "
                      f"train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} | "
                      f"val loss={val_m['loss']:.4f} "
                      f"acc={val_m['acc']:.4f} "
                      f"auc={val_m['auc']:.4f} "
                      f"sens={val_m['sens']:.4f} "
                      f"spec={val_m['spec']:.4f}"
                      + (" ← best" if is_best
                         else f"  (no improve {no_improve}/{patience})"))

            # Save
            local_last = os.path.join(local_ckpt, "last.pt")
            save_ckpt(local_last, model, optimizer, scheduler,
                      epoch, best_metric, args)
            copy_threads.append(
                _async_copy(local_last, os.path.join(args.out_dir, "last.pt")))
            if is_best:
                local_best = os.path.join(local_ckpt, "best.pt")
                shutil.copy2(local_last, local_best)
                copy_threads.append(
                    _async_copy(local_best, os.path.join(args.out_dir, "best.pt")))

            if no_improve >= patience:
                print(f"Early stopping at {stage_name} epoch {epoch}.")
                break

    # ---- Build optimizer helpers ----
    def _make_optimizer(freeze_enc):
        if freeze_enc:
            params = [p for n, p in model.named_parameters()
                      if not any(n.startswith(e)
                                 for e in ["enc0","enc1","enc2","enc3","enc4"])]
            return torch.optim.AdamW(params, lr=args.lr_head,
                                     weight_decay=args.weight_decay)
        else:
            enc_params  = [p for n, p in model.named_parameters()
                           if any(n.startswith(e)
                                  for e in ["enc0","enc1","enc2","enc3","enc4"])]
            head_params = [p for n, p in model.named_parameters()
                           if not any(n.startswith(e)
                                      for e in ["enc0","enc1","enc2","enc3","enc4"])]
            return torch.optim.AdamW(
                [{"params": enc_params,  "lr": args.lr_backbone},
                 {"params": head_params, "lr": args.lr_head}],
                weight_decay=args.weight_decay)

    # ---- Stage 1: Linear probe ----
    if args.probe_epochs > 0:
        model.freeze_encoder()
        print(f"Encoder frozen for probe stage ({args.probe_epochs} epochs).")
        opt1 = _make_optimizer(freeze_enc=True)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=args.probe_epochs, eta_min=1e-6)
        _run_stage("Probe", args.probe_epochs, opt1, sch1, args.patience)

    # ---- Stage 2: Fine-tune ----
    model.unfreeze_encoder()
    print("\nUnfreezing encoder for fine-tuning ...")
    opt2 = _make_optimizer(freeze_enc=False)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=args.finetune_epochs, eta_min=1e-7)
    _run_stage("FT", args.finetune_epochs, opt2, sch2, args.patience)

    # ---- Wait NAS sync ----
    print("\nWaiting for NAS sync ...")
    for t in copy_threads: t.join()

    # ---- Final test evaluation ----
    print(f"\nFinal evaluation on test set (best checkpoint):")
    best_ckpt = torch.load(os.path.join(local_ckpt, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_m = eval_fn(model, test_loader, device)

    if task == "seg":
        print(f"Test  LV={test_m['dice_LV']:.4f}  "
              f"MYO={test_m['dice_MYO']:.4f}  "
              f"scar={test_m['dice_scar']:.4f}  "
              f"mean={test_m['dice_mean']:.4f}")
    else:
        print(f"Test  acc={test_m['acc']:.4f}  auc={test_m['auc']:.4f}  "
              f"sens={test_m['sens']:.4f}  spec={test_m['spec']:.4f}  "
              f"TP={test_m['tp']} TN={test_m['tn']} "
              f"FP={test_m['fp']} FN={test_m['fn']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Task: {args.task}")

    if args.task == "seg":
        print("\nTrain dataset:")
        train_ds = EmidecSegDataset(args.index_csv, "train",
                                    args.image_size, train=True,
                                    max_cases=args.max_cases)
        print("Test dataset:")
        test_ds  = EmidecSegDataset(args.index_csv, "test",
                                    args.image_size, train=False)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=seg_collate, drop_last=True,
        )
        val_loader  = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=seg_collate,
        )
        test_loader = val_loader   # same split used for both val and final test

        model = ResNet50UNet(num_classes=SEG_NUM_CLASSES,
                             dropout=args.dropout).to(device)
        if args.ckpt_path:
            model.load_pretrained(args.ckpt_path, device)

        run_training(
            model, train_loader, val_loader, test_loader,
            args, device, task="seg",
            eval_fn=evaluate_seg,
            train_fn=train_one_epoch_seg,
            metric_key="dice_mean",
        )

    elif args.task == "det":
        print("\nTrain dataset:")
        train_ds = EmidecDetDataset(args.index_csv, "train",
                                    args.image_size, train=True)
        print("Test dataset:")
        test_ds  = EmidecDetDataset(args.index_csv, "test",
                                    args.image_size, train=False)

        # Balanced sampler for N/P imbalance (N=26 train, P=54 train)
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=det_collate, drop_last=True,
        )
        val_loader  = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=det_collate,
        )
        test_loader = val_loader

        model = ResNet50Classifier(num_classes=2, embed_dim=args.embed_dim,
                                   dropout=args.dropout).to(device)
        if args.ckpt_path:
            model.load_pretrained(args.ckpt_path, device)

        run_training(
            model, train_loader, val_loader, test_loader,
            args, device, task="det",
            eval_fn=evaluate_det,
            train_fn=train_one_epoch_det,
            metric_key="auc",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="EMIDEC: LGE SAX segmentation (seg) or LGE detection (det)"
    )
    p.add_argument("--task",       required=True, choices=["seg", "det"])
    p.add_argument("--index_csv",  required=True,
                   help="index_seg.parquet or index_det.parquet")
    p.add_argument("--ckpt_path",  default=None,
                   help="Pre-trained CLIP checkpoint (best.pt)")
    p.add_argument("--out_dir",    required=True)

    p.add_argument("--image_size",  type=int,   default=224)
    p.add_argument("--embed_dim",   type=int,   default=256)
    p.add_argument("--dropout",     type=float, default=0.1)

    p.add_argument("--probe_epochs",    type=int,   default=15)
    p.add_argument("--finetune_epochs", type=int,   default=20)
    p.add_argument("--lr_head",         type=float, default=1e-3)
    p.add_argument("--lr_backbone",     type=float, default=1e-5)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--patience",        type=int,   default=10)

    p.add_argument("--batch_size",   type=int,  default=8)
    p.add_argument("--accum_steps",  type=int,  default=2)
    p.add_argument("--num_workers",  type=int,  default=8)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--max_cases",    type=int,  default=None,
                   help="Limit training to N cases (few-shot).")
    p.add_argument("--amp",          action="store_true")

    args = p.parse_args()
    main(args)