"""
train_acdc_disease.py  (v2 — multi-slice input)
================================================
ACDC 5-class disease classification using multi-slice 3D input.

Key change from v1:
    Each case is a single sample (N_FRAMES=20, H, W):
        - ED frame: 10 slices (uniformly sampled / zero-padded)
        - ES frame: 10 slices (uniformly sampled / zero-padded)
    The model sees the full 3D cardiac structure in one forward pass.

Classes: DCM=0, HCM=1, MINF=2, NOR=3, RV=4

Usage
-----
python train_acdc_disease.py \
    --acdc_root  /path/ACDC/database \
    --ckpt_path  /path/cmr_multimodal/best.pt \
    --out_dir    /path/runs/acdc_disease_v2 \
    --amp
"""

import os, random, argparse, tempfile, shutil, threading
import numpy as np, nibabel as nib, pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

CLASSES     = ["DCM", "HCM", "MINF", "NOR", "RV"]
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = 5
N_SLICES    = 10   # per frame
N_FRAMES    = 20   # ED*10 + ES*10

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def normalize_zscore(img):
    img = img.astype(np.float32)
    m, s = float(img.mean()), float(img.std())
    img = (img - m) / s if s > 1e-6 else img - m
    return np.clip((img + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)

def sample_slices(vol, n):
    H, W, S = vol.shape
    if S == n:
        return vol.transpose(2, 0, 1).astype(np.float32)
    elif S > n:
        idxs = np.round(np.linspace(0, S-1, n)).astype(int)
        return vol[:, :, idxs].transpose(2, 0, 1).astype(np.float32)
    else:
        out = np.zeros((n, H, W), dtype=np.float32)
        out[:S] = vol.transpose(2, 0, 1)
        return out

def read_acdc_info(cfg_path):
    info = {}
    with open(cfg_path) as f:
        for line in f:
            if ":" in line:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
    return info

def load_case(patient_dir, patient_id, ed_frame, es_frame, n_slices=N_SLICES):
    def _load(fi):
        path = os.path.join(patient_dir, f"{patient_id}_frame{fi:02d}.nii.gz")
        vol  = nib.load(path).get_fdata().astype(np.float32)
        return sample_slices(normalize_zscore(vol), n_slices)
    ed = _load(ed_frame)
    es = _load(es_frame)
    return np.concatenate([ed, es], axis=0)  # (N_FRAMES, H, W)


class MultiSliceTransform:
    def __init__(self, image_size=224, train=True,
                 rotate_deg=15.0, translate=0.05, scale_range=(0.9, 1.1),
                 p_affine=0.5, p_hflip=0.3, p_noise=0.2, noise_std=0.03):
        self.sz = image_size; self.train = train
        self.rd = rotate_deg; self.tr = translate
        self.sc = scale_range; self.pa = p_affine
        self.ph = p_hflip; self.pn = p_noise; self.ns = noise_std

    def __call__(self, x):
        x = F.interpolate(x.unsqueeze(0), (self.sz, self.sz),
                          mode="bilinear", align_corners=False).squeeze(0)
        if not self.train: return x.clamp(0, 1)
        if random.random() < self.pa:
            angle = random.uniform(-self.rd, self.rd)
            md = self.tr * self.sz
            tr = (int(round(random.uniform(-md, md))), int(round(random.uniform(-md, md))))
            sc = random.uniform(*self.sc)
            x  = torch.cat([TF.affine(x[i:i+1], angle, tr, sc, [0.],
                             interpolation=TF.InterpolationMode.BILINEAR)
                            for i in range(x.shape[0])], dim=0)
        if random.random() < self.ph:
            x = torch.cat([TF.hflip(x[i:i+1]) for i in range(x.shape[0])], dim=0)
        if random.random() < self.pn:
            x = x + torch.randn_like(x) * self.ns
        return x.clamp(0, 1)


class AcdcDiseaseDataset(Dataset):
    def __init__(self, acdc_root, split="train", image_size=224,
                 train=True, n_slices=N_SLICES):
        acdc_split = "training" if split == "train" else "testing"
        split_dir  = os.path.join(acdc_root, acdc_split)
        self.samples = []
        self.n_slices = n_slices
        for p in sorted(os.listdir(split_dir)):
            if not p.startswith("patient"): continue
            if not os.path.isdir(os.path.join(split_dir, p)): continue
            cfg = os.path.join(split_dir, p, "Info.cfg")
            if not os.path.exists(cfg): continue
            info  = read_acdc_info(cfg)
            group = info.get("Group", "")
            if group not in CLASS2IDX: continue
            self.samples.append({
                "patient_dir": os.path.join(split_dir, p),
                "patient_id":  p,
                "group": group, "label": CLASS2IDX[group],
                "ed_frame": int(info.get("ED", 1)),
                "es_frame": int(info.get("ES", 1)),
            })
        self.transform = MultiSliceTransform(image_size=image_size, train=train)
        cnt = Counter(s["group"] for s in self.samples)
        print(f"  AcdcDiseaseDataset [{split}]: {len(self.samples)} cases")
        for cls in CLASSES:
            print(f"    {cls}: {cnt.get(cls, 0)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        vol = load_case(s["patient_dir"], s["patient_id"],
                        s["ed_frame"], s["es_frame"], self.n_slices)
        x   = self.transform(torch.from_numpy(vol))
        return {"image": x, "label": s["label"],
                "patient_id": s["patient_id"], "group": s["group"]}

def collate_fn(batch):
    return {"image":  torch.stack([b["image"]  for b in batch]),
            "label":  torch.tensor([b["label"] for b in batch]),
            "patient_id": [b["patient_id"] for b in batch],
            "group":      [b["group"]      for b in batch]}


class ResNet50MultiSlice(nn.Module):
    def __init__(self, num_classes=5, n_frames=N_FRAMES,
                 embed_dim=256, dropout=0.1):
        super().__init__()
        self.n_frames = n_frames
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        old = bb.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        bb.conv1 = new; bb.fc = nn.Identity()
        self.backbone = bb
        self.head = nn.Sequential(
            nn.Linear(2048, embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        B, T, H, W = x.shape
        feat = self.backbone(x.view(B*T, 1, H, W))  # (B*T, 2048)
        feat = feat.view(B, T, -1).mean(dim=1)       # (B, 2048)
        return self.head(feat)

    def freeze_encoder(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def load_pretrained(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd   = {k.replace("image_encoder.backbone.", ""): v
                for k, v in ckpt["model"].items()
                if k.startswith("image_encoder.backbone.")}
        miss, unex = self.backbone.load_state_dict(sd, strict=False)
        print(f"  Encoder: {len(miss)} missing, {len(unex)} unexpected")
        print(f"  Loaded from {ckpt_path}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    all_true, all_pred = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        loss_sum += F.cross_entropy(logits, labels).item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        count   += imgs.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    acc    = correct / max(count, 1)
    report = classification_report(all_true, all_pred,
                                   target_names=CLASSES, digits=4, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    return {"loss": loss_sum/max(count,1), "acc": acc, "report": report, "cm": cm}


def train_one_epoch(model, loader, optimizer, device, epoch, amp, accum):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type=="cuda"))
    loss_sum, correct, count = 0., 0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for step, batch in enumerate(pbar):
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.amp.autocast("cuda", enabled=(amp and device.type=="cuda")):
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels) / accum
        scaler.scale(loss).backward()
        if (step+1) % accum == 0 or (step+1) == len(loader):
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        bs = imgs.size(0)
        loss_sum += loss.item() * accum * bs
        correct  += (logits.argmax(1) == labels).sum().item()
        count    += bs
        pbar.set_postfix({"loss": f"{loss_sum/count:.4f}",
                          "acc":  f"{correct/count:.4f}"})
    return {"loss": loss_sum/count, "acc": correct/count}


def save_ckpt(path, model, opt, sch, epoch, best_acc, args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sch.state_dict() if sch else None,
                "best_acc": best_acc, "args": vars(args)}, path)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Input: {N_FRAMES} channels (ED×{N_SLICES} + ES×{N_SLICES})")

    print("\nTrain:"); train_ds = AcdcDiseaseDataset(args.acdc_root, "train", args.image_size, True)
    print("\nTest:");  test_ds  = AcdcDiseaseDataset(args.acdc_root, "test",  args.image_size, False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    print(f"\nTrain batches: {len(train_loader)}")

    model = ResNet50MultiSlice(NUM_CLASSES, N_FRAMES, args.embed_dim, args.dropout).to(device)
    if args.ckpt_path: model.load_pretrained(args.ckpt_path, device)

    os.makedirs(args.out_dir, exist_ok=True)
    local_ckpt = tempfile.mkdtemp(prefix="acdc_disease_v2_")
    threads: list = []

    def _copy(src, dst):
        def _do():
            try: os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
            except Exception as e: print(f"[warn] {e}")
        t = threading.Thread(target=_do, daemon=True); t.start(); return t

    best_acc = 0.; no_improve = 0

    def _make_opt(freeze):
        if freeze:
            return torch.optim.AdamW(
                list(model.head.parameters()),
                lr=args.lr_head, weight_decay=args.weight_decay)
        return torch.optim.AdamW([
            {"params": list(model.backbone.parameters()), "lr": args.lr_backbone},
            {"params": list(model.head.parameters()),     "lr": args.lr_head},
        ], weight_decay=args.weight_decay)

    def _stage(name, n_ep, freeze, patience):
        nonlocal best_acc, no_improve
        no_improve = 0
        if freeze: model.freeze_encoder()
        else:      model.unfreeze_encoder()
        opt = _make_opt(freeze)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_ep, eta_min=1e-6 if freeze else 1e-7)
        print(f"\n{'='*60}\n{name} ({n_ep} ep, "
              f"{'frozen' if freeze else 'unfrozen'})\n{'='*60}")
        for epoch in range(1, n_ep+1):
            tr = train_one_epoch(model, train_loader, opt, device,
                                 epoch, args.amp, args.accum_steps)
            va = evaluate(model, test_loader, device)
            sch.step()
            is_best = va["acc"] > best_acc
            if is_best: best_acc = va["acc"]; no_improve = 0
            else: no_improve += 1
            print(f"[{name} {epoch:3d}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} "
                  f"| test acc={va['acc']:.4f}"
                  + (" ← best" if is_best else f"  (no improve {no_improve}/{patience})"))
            ll = os.path.join(local_ckpt, "last.pt")
            save_ckpt(ll, model, opt, sch, epoch, best_acc, args)
            threads.append(_copy(ll, os.path.join(args.out_dir, "last.pt")))
            if is_best:
                lb = os.path.join(local_ckpt, "best.pt")
                shutil.copy2(ll, lb)
                threads.append(_copy(lb, os.path.join(args.out_dir, "best.pt")))
            if no_improve >= patience:
                print(f"Early stopping at {name} epoch {epoch}."); break

    _stage("Probe", args.probe_epochs,    freeze=True,  patience=args.patience)
    _stage("FT",    args.finetune_epochs, freeze=False, patience=args.patience)

    print("\nWaiting for NAS sync ...")
    for t in threads: t.join()

    print(f"\n{'='*60}\nFinal evaluation (best checkpoint)\n{'='*60}")
    model.load_state_dict(torch.load(os.path.join(local_ckpt, "best.pt"),
                                     map_location=device)["model"])
    fm = evaluate(model, test_loader, device)
    print(f"Test accuracy: {fm['acc']:.4f}\n{fm['report']}")
    print("Confusion matrix:")
    print(pd.DataFrame(fm["cm"], index=CLASSES, columns=CLASSES).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acdc_root",       required=True)
    p.add_argument("--ckpt_path",       default=None)
    p.add_argument("--out_dir",         required=True)
    p.add_argument("--image_size",      type=int,   default=224)
    p.add_argument("--embed_dim",       type=int,   default=256)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--probe_epochs",    type=int,   default=20)
    p.add_argument("--finetune_epochs", type=int,   default=20)
    p.add_argument("--lr_head",         type=float, default=1e-3)
    p.add_argument("--lr_backbone",     type=float, default=1e-5)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--patience",        type=int,   default=7)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--accum_steps",     type=int,   default=2)
    p.add_argument("--num_workers",     type=int,   default=8)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--amp",             action="store_true")
    main(p.parse_args())