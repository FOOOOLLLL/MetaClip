"""
train_mim.py
============
Masked Image Modeling (MIM) pre-training for CMR images using ResNet50.

This is a pure-vision self-supervised baseline to compare against CMR CLIP.
The goal is to answer: does image-text alignment provide additional value
over pure visual self-supervision?

Method:
    1. Randomly mask 60% of each image using block masking (block_size=32)
    2. ResNet50 encoder processes the masked image
    3. Lightweight CNN decoder reconstructs the original image
    4. MSE loss computed only on masked regions
    5. After pre-training, discard decoder, keep only encoder

The encoder architecture is identical to CMR CLIP for fair comparison:
    - ResNet50 with 1-channel grayscale conv1
    - ImageNet V2 weights initialised, conv1 averaged from RGB->grayscale
    - Same input pipeline (3 slices x 4 frames per sequence)

Usage
-----
python train_mim.py \
    --index_csv  /path/all_train.parquet \
    --val_csv    /path/all_test.parquet \
    --out_dir    /path/runs/mim_pretrain \
    --epochs     20 \
    --amp

After training, use the checkpoint with train_sax_seg.py:
    python train_sax_seg.py --ckpt_path /path/runs/mim_pretrain/best.pt ...
"""

import os
import sys
import math
import random
import argparse
import tempfile
import shutil
import threading

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# Reuse dataset utilities from cmr_multimodal_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cmr_multimodal_dataset import (
        MultiModalCMRDataset,
        load_npy_image, normalize_image,
        parse_mat_path, ALL_MODALITIES,
        set_seed,
    )
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Block Mask Generator
# ---------------------------------------------------------------------------

def make_block_mask(
    H: int,
    W: int,
    mask_ratio: float = 0.60,
    block_size: int   = 32,
) -> torch.Tensor:
    """
    Generate a random block mask.

    Returns a (1, H, W) float tensor:
        1.0 = masked (to be reconstructed)
        0.0 = visible (input to encoder)

    Blocks of size block_size x block_size are randomly selected until
    the target mask_ratio is reached.
    """
    n_blocks_h = max(1, H // block_size)
    n_blocks_w = max(1, W // block_size)
    n_total    = n_blocks_h * n_blocks_w
    n_mask     = max(1, int(n_total * mask_ratio))

    block_idxs = np.random.choice(n_total, size=n_mask, replace=False)
    mask = np.zeros((H, W), dtype=np.float32)
    for idx in block_idxs:
        i = (idx // n_blocks_w) * block_size
        j = (idx % n_blocks_w) * block_size
        i_end = min(i + block_size, H)
        j_end = min(j + block_size, W)
        mask[i:i_end, j:j_end] = 1.0

    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# MIM Dataset  (wraps MultiModalCMRDataset, strips text)
# ---------------------------------------------------------------------------

class MIMDataset(Dataset):
    """
    Wraps MultiModalCMRDataset for MIM pre-training.
    Returns (clean_image, masked_image, mask) instead of (image, text).

    clean_image  : (1, H, W) original normalised image
    masked_image : (1, H, W) with masked regions set to 0
    mask         : (1, H, W) binary mask — 1 = masked pixel
    """

    def __init__(
        self,
        csv_path:   str,
        image_size: int   = 224,
        train:      bool  = True,
        num_slices: int   = 3,
        num_frames: int   = 4,
        mask_ratio: float = 0.60,
        block_size: int   = 32,
    ):
        if not DATASET_AVAILABLE:
            raise RuntimeError(
                "cmr_multimodal_dataset.py not found. "
                "Place it in the same directory as train_mim.py."
            )

        self.base_ds = MultiModalCMRDataset(
            csv_path    = csv_path,
            image_size  = image_size,
            train       = train,
            num_slices  = num_slices,
            num_frames  = num_frames,
            p_noise     = 0.0,   # no noise augmentation for MIM
            p_cutout    = 0.0,   # masking is done explicitly below
        )
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.train      = train

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_ds[idx]
        imgs = item["image"]          # (T, H, W)  T = num_slices * num_frames

        # Pick one random frame from the T frames for MIM
        # (training on a random single frame avoids temporal redundancy)
        t_idx = random.randint(0, imgs.shape[0] - 1) if self.train else 0
        img   = imgs[t_idx].unsqueeze(0)   # (1, H, W)

        # Generate block mask
        mask         = make_block_mask(self.image_size, self.image_size,
                                       self.mask_ratio, self.block_size)
        masked_img   = img * (1.0 - mask)  # zero out masked regions

        return {
            "clean":    img,          # (1, H, W)  original
            "masked":   masked_img,   # (1, H, W)  input to encoder
            "mask":     mask,         # (1, H, W)  1=masked
            "modality": item["modality"],
            "mat_path": item["mat_path"],
        }


def mim_collate(batch: list[dict]) -> dict:
    return {
        "clean":    torch.stack([b["clean"]  for b in batch]),
        "masked":   torch.stack([b["masked"] for b in batch]),
        "mask":     torch.stack([b["mask"]   for b in batch]),
        "modality": [b["modality"] for b in batch],
        "mat_path": [b["mat_path"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Encoder  (identical to CMR CLIP image encoder for fair comparison)
# ---------------------------------------------------------------------------

class ResNet50Encoder(nn.Module):
    """
    ResNet50 encoder with 1-channel grayscale input.
    conv1 weights initialised by averaging RGB channels from ImageNet weights.
    Output: (B, 2048, 7, 7) feature map (before GAP).
    """

    def __init__(self):
        super().__init__()
        bb  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        old = bb.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        bb.conv1 = new

        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool = bb.maxpool
        self.enc1 = bb.layer1   # out: (256,  56, 56)
        self.enc2 = bb.layer2   # out: (512,  28, 28)
        self.enc3 = bb.layer3   # out: (1024, 14, 14)
        self.enc4 = bb.layer4   # out: (2048,  7,  7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc0(x)
        x = self.enc1(self.pool(x))
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x   # (B, 2048, 7, 7)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Decoder  (lightweight CNN, discarded after pre-training)
# ---------------------------------------------------------------------------

class MIMDecoder(nn.Module):
    """
    Lightweight transposed-convolution decoder.
    Input : (B, 2048, 7, 7)  from ResNet50 encoder
    Output: (B, 1, 224, 224) reconstructed image

    5 upsampling stages: 7 -> 14 -> 28 -> 56 -> 112 -> 224
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        def _up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up1 = _up_block(2048, 512)   # 7  -> 14
        self.up2 = _up_block(512,  256)   # 14 -> 28
        self.up3 = _up_block(256,  128)   # 28 -> 56
        self.up4 = _up_block(128,   64)   # 56 -> 112
        self.up5 = _up_block( 64,   32)   # 112 -> 224
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.up1(feat)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(self.drop(x))
        return self.head(x)   # (B, 1, 224, 224)


# ---------------------------------------------------------------------------
# Full MIM Model
# ---------------------------------------------------------------------------

class MIMModel(nn.Module):
    def __init__(self, decoder_dropout: float = 0.1):
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.decoder = MIMDecoder(dropout=decoder_dropout)

    def forward(self, masked_img: torch.Tensor) -> torch.Tensor:
        feat  = self.encoder(masked_img)   # (B, 2048, 7, 7)
        recon = self.decoder(feat)         # (B, 1, 224, 224)
        return recon

    def load_pretrained_encoder(self, ckpt_path: str, device):
        """Load encoder weights from a saved MIM checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device)
        sd   = {k.replace("encoder.", ""): v
                for k, v in ckpt["model"].items()
                if k.startswith("encoder.")}
        miss, unex = self.encoder.load_state_dict(sd, strict=False)
        print(f"  MIM encoder load: {len(miss)} missing, {len(unex)} unexpected")
        print(f"  Loaded from {ckpt_path}")


# ---------------------------------------------------------------------------
# MIM Loss  (MSE on masked pixels only)
# ---------------------------------------------------------------------------

def mim_loss(
    recon: torch.Tensor,
    clean: torch.Tensor,
    mask:  torch.Tensor,
) -> torch.Tensor:
    """
    MSE reconstruction loss, computed only on masked pixels.

    recon : (B, 1, H, W)  model output
    clean : (B, 1, H, W)  original image
    mask  : (B, 1, H, W)  1 = masked pixel, 0 = visible
    """
    diff      = (recon - clean) ** 2     # (B, 1, H, W)
    loss_map  = diff * mask               # only masked positions
    n_masked  = mask.sum().clamp(min=1)
    return loss_map.sum() / n_masked


# ---------------------------------------------------------------------------
# Modality accuracy  (proxy metric — does encoder separate modalities?)
# ---------------------------------------------------------------------------

@torch.no_grad()
def modality_accuracy(model, loader, device) -> float:
    """
    Linear-probe modality accuracy using GAP features from the encoder.
    Fit a nearest-centroid classifier on train embeddings, evaluate on val.
    Used as a proxy to compare with CLIP's mod_acc metric.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder

    model.eval()
    feats, mods = [], []
    for batch in tqdm(loader, desc="Mod-acc", leave=False):
        imgs = batch["masked"].to(device)
        with torch.no_grad():
            feat = model.encoder(imgs)            # (B, 2048, 7, 7)
            feat = feat.mean(dim=[2, 3])          # (B, 2048) GAP
        feats.append(feat.cpu().numpy())
        mods.extend(batch["modality"])

    X  = np.concatenate(feats, axis=0)
    le = LabelEncoder()
    y  = le.fit_transform(mods)

    # 5-NN modality classification (same as CLIP eval)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X, y)
    preds = knn.predict(X)
    acc   = float((preds == y).mean())
    return acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     MIMModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    amp:       bool,
    accum:     int,
) -> dict:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    total_loss, count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        clean  = batch["clean"].to(device,  non_blocking=True)
        masked = batch["masked"].to(device, non_blocking=True)
        mask   = batch["mask"].to(device,   non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            recon = model(masked)
            loss  = mim_loss(recon, clean, mask) / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs         = clean.size(0)
        total_loss += loss.item() * accum * bs
        count      += bs
        pbar.set_postfix({"loss": f"{total_loss / count:.4f}"})

    return {"loss": total_loss / max(count, 1)}


@torch.no_grad()
def evaluate(
    model:  MIMModel,
    loader: DataLoader,
    device: torch.device,
    amp:    bool,
) -> dict:
    model.eval()
    total_loss, count = 0.0, 0

    for batch in tqdm(loader, desc="Val", leave=False):
        clean  = batch["clean"].to(device)
        masked = batch["masked"].to(device)
        mask   = batch["mask"].to(device)
        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            recon = model(masked)
            loss  = mim_loss(recon, clean, mask)
        bs         = clean.size(0)
        total_loss += loss.item() * bs
        count      += bs

    return {"loss": total_loss / max(count, 1)}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_ckpt(path, model, optimizer, scheduler, epoch, best_loss, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "args":      vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mask ratio: {args.mask_ratio:.0%}  Block size: {args.block_size}px")

    # ---- Datasets ----
    print("\nTrain dataset:")
    train_ds = MIMDataset(
        csv_path   = args.index_csv,
        image_size = args.image_size,
        train      = True,
        num_slices = args.num_slices,
        num_frames = args.num_frames,
        mask_ratio = args.mask_ratio,
        block_size = args.block_size,
    )
    print(f"\nVal dataset:")
    val_ds = MIMDataset(
        csv_path   = args.val_csv,
        image_size = args.image_size,
        train      = False,
        num_slices = args.num_slices,
        num_frames = args.num_frames,
        mask_ratio = args.mask_ratio,
        block_size = args.block_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        collate_fn  = mim_collate,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        collate_fn  = mim_collate,
    )

    print(f"\nTrain batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ---- Model ----
    model = MIMModel(decoder_dropout=args.dropout).to(device)

    n_enc  = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    n_dec  = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"\nEncoder params: {n_enc:.1f}M  Decoder params: {n_dec:.1f}M")

    # ---- Optimiser & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    # Cosine annealing with linear warmup
    warmup_epochs = max(1, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Checkpoint helpers ----
    os.makedirs(args.out_dir, exist_ok=True)
    local_ckpt   = tempfile.mkdtemp(prefix="mim_pretrain_")
    copy_threads = []

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

    # ---- Training loop ----
    best_loss  = float("inf")
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"MIM Pre-training  ({args.epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, args.amp, args.accum_steps,
        )
        val_m   = evaluate(model, val_loader, device, args.amp)
        scheduler.step()

        is_best = val_m["loss"] < best_loss
        if is_best:
            best_loss  = val_m["loss"]
            no_improve = 0
        else:
            no_improve += 1

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[Epoch {epoch:3d}] "
            f"train loss={train_m['loss']:.4f} | "
            f"val loss={val_m['loss']:.4f}  "
            f"lr={lr_now:.2e}"
            + (" ← best" if is_best
               else f"  (no improve {no_improve}/{args.patience})")
        )

        # Save checkpoints
        local_last = os.path.join(local_ckpt, "last.pt")
        save_ckpt(local_last, model, optimizer, scheduler, epoch, best_loss, args)
        copy_threads.append(
            _async_copy(local_last, os.path.join(args.out_dir, "last.pt"))
        )
        if is_best:
            local_best = os.path.join(local_ckpt, "best.pt")
            shutil.copy2(local_last, local_best)
            copy_threads.append(
                _async_copy(local_best, os.path.join(args.out_dir, "best.pt"))
            )

        # Modality accuracy every 5 epochs (expensive)
        if epoch % 5 == 0 or epoch == args.epochs:
            mod_acc = modality_accuracy(model, val_loader, device)
            print(f"  5-NN modality acc (val): {mod_acc:.4f}  "
                  f"[CLIP baseline: 0.777]")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ---- Wait for NAS sync ----
    print("\nWaiting for NAS sync ...")
    for t in copy_threads:
        t.join()

    print(f"\nDone. Best val loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {args.out_dir}/best.pt")
    print(f"\nTo use for downstream tasks:")
    print(f"  python train_sax_seg.py --ckpt_path {args.out_dir}/best.pt ...")
    print(f"  (the script will automatically detect MIM vs CLIP checkpoint format)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MIM pre-training for CMR images (pure-vision baseline vs CMR CLIP)"
    )

    # Data
    p.add_argument("--index_csv",   required=True,
                   help="Train CSV/parquet (e.g. all_train.parquet)")
    p.add_argument("--val_csv",     required=True,
                   help="Val CSV/parquet (e.g. all_test.parquet)")
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--num_slices",  type=int, default=3,
                   help="Slices sampled per sequence (same as CLIP)")
    p.add_argument("--num_frames",  type=int, default=4,
                   help="Frames sampled per slice (same as CLIP)")

    # Masking
    p.add_argument("--mask_ratio",  type=float, default=0.60,
                   help="Fraction of image to mask (default: 0.60)")
    p.add_argument("--block_size",  type=int,   default=32,
                   help="Block size for random block masking (pixels)")

    # Training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--dropout",      type=float, default=0.1)

    # Infrastructure
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--amp",         action="store_true")

    main(p.parse_args())