"""
eval_seg_checkpoint.py
=======================
Inference-only evaluation of an already-trained segmentation checkpoint.

No training is performed here. This script loads a `best.pt` produced by
train_sax_seg.py, runs it once over the test split, and dumps per-case Dice
scores (RV / MYO / LV / mean) to a CSV file. The resulting CSVs are the
input for computing mean ± std and paired significance tests (Friedman +
Wilcoxon signed-rank, with Holm correction for multiple comparisons) across
ImageNet / masked-reconstruction / MetaCLIP-CMR initialisations.

Usage
-----
python eval_seg_checkpoint.py \
    --dataset    acdc \
    --index_csv  /path/sax_seg_npy/index.parquet \
    --ckpt_path  /path/runs/ft100_acdc_clip/best.pt \
    --out_csv    ./results/acdc_100_clip_percase.csv
"""

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_sax_seg import (
    NUM_CLASSES,
    ResNet50UNet,
    SaxSegDataset,
    evaluate_by_case,
    seg_collate,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Test dataset (fixed, no augmentation) ----
    test_ds = SaxSegDataset(
        args.index_csv,
        dataset=args.dataset,
        split="test",
        image_size=args.image_size,
        train=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=seg_collate,
        drop_last=False,
    )

    # ---- Model: load the FULL trained model state_dict directly ----
    # best.pt was written by save_checkpoint() during training and contains
    # the complete ResNet50UNet state_dict (encoder + decoder + head), so we
    # load it directly rather than going through load_pretrained_encoder()
    # (which is only used to initialise the encoder before training starts).
    model = ResNet50UNet(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path}")
    if "epoch" in ckpt:
        print(f"  (epoch {ckpt['epoch']}, best_dice={ckpt.get('best_dice', 'n/a')})")

    # ---- Inference over the full test set, per-case Dice ----
    result = evaluate_by_case(model, test_loader, device, return_per_case=True)

    df = pd.DataFrame(result["per_case"])
    df.to_csv(args.out_csv, index=False)

    print(f"Saved {len(df)} cases to {args.out_csv}")
    print(
        f"Mean Dice: RV={result['dice_RV']:.4f}  "
        f"MYO={result['dice_MYO']:.4f}  "
        f"LV={result['dice_LV']:.4f}  "
        f"mean={result['dice_mean']:.4f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inference-only per-case Dice evaluation of an existing "
                     "segmentation checkpoint (no training)."
    )
    p.add_argument("--dataset", required=True, choices=["acdc", "mms"])
    p.add_argument("--index_csv", required=True,
                    help="Path to index CSV/parquet from preprocess_sax_seg.py")
    p.add_argument("--ckpt_path", required=True,
                    help="Path to a best.pt checkpoint from train_sax_seg.py")
    p.add_argument("--out_csv", required=True,
                    help="Output CSV path for per-case Dice records")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()
    main(args)