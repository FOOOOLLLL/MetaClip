"""
eval_cls_checkpoint.py
=======================
Inference-only evaluation of an already-trained classification checkpoint.

No training is performed here. This script loads a probe_best.pt or
ft_best.pt produced by train_classification.py, runs it once over the test
split, and dumps per-sample predictions (true label, predicted label,
correct/incorrect) to a CSV file. The resulting CSVs are the input for:
  - Bootstrap resampling to estimate a confidence interval / SD for
    accuracy (in place of a per-case SD, since classification accuracy is
    a single aggregate statistic rather than a per-case value like Dice).
  - Cochran's Q test (paired, non-parametric, 3+ groups) and pairwise
    McNemar's tests with Holm correction, comparing ImageNet / MIM /
    MetaCLIP-CMR on the same test samples.

Row order is deterministic and identical across methods: ClassificationDataset
builds its sample list via `df.groupby("mat_path")`, which iterates in sorted
key order, and the test loader uses shuffle=False. As long as the same
--test_csv is used for all three methods (which it is, per train_classification.py's
usage), row i in one method's output CSV corresponds to the same underlying
sequence as row i in another method's output CSV. The mat_path column is
included as an explicit identifier so this can be verified rather than assumed.

Usage
-----
python eval_cls_checkpoint.py \
    --task       modality \
    --test_csv   /path/all_test.parquet \
    --ckpt_path  /path/runs/cls_mod_clip/ft_best.pt \
    --out_csv    ./results_cls/mod_clip_ft_persample.csv
"""

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_classification import (
    CINE_VIEW_CLASSES,
    CINE_VIEW_MAP,
    MOD_CLASSES,
    ClassificationDataset,
    ClassificationModel,
    collate_fn,
    evaluate,
)


def build_task_cfg(task: str) -> dict:
    if task == "modality":
        return {
            "classes": MOD_CLASSES,
            "filter":  lambda row: row["modality"] in set(MOD_CLASSES),
            "label":   lambda row: row["modality"],
        }
    elif task == "cine_view":
        return {
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
        raise ValueError(f"Unknown task: {task}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    task_cfg = build_task_cfg(args.task)
    classes = task_cfg["classes"]
    print(f"Task: {args.task}  |  Classes ({len(classes)}): {classes}")

    # ---- Test dataset (fixed, no augmentation, deterministic order) ----
    test_ds = ClassificationDataset(
        args.test_csv,
        task_cfg=task_cfg,
        train=False,
        image_size=args.image_size,
        num_slices=args.num_slices,
        num_frames=args.num_frames,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- Model: load the FULL trained model state_dict directly ----
    # probe_best.pt / ft_best.pt were written by save_checkpoint() during
    # training and contain the complete ClassificationModel state_dict
    # (encoder + head), so we load it directly rather than going through
    # load_pretrained_encoder() (which is only used to initialise the
    # encoder from a pre-training checkpoint before downstream training
    # starts).
    model = ClassificationModel(
        num_classes=len(classes),
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        backbone_name=args.backbone,
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path}")
    if "epoch" in ckpt:
        print(f"  (epoch {ckpt['epoch']}, best_acc={ckpt.get('best_acc', 'n/a')})")

    # ---- Inference over the full test set, per-sample predictions ----
    result = evaluate(model, test_loader, device, classes)

    n = len(result["labels"])
    mat_paths = [test_ds.samples[i]["mat_path"] for i in range(n)]
    label_names = [test_ds.samples[i]["label_name"] for i in range(n)]

    df = pd.DataFrame({
        "sample_idx": range(n),
        "mat_path": mat_paths,
        "true_label_idx": result["labels"],
        "true_label_name": label_names,
        "pred_label_idx": result["preds"],
        "pred_label_name": [classes[p] for p in result["preds"]],
        "correct": [int(p == l) for p, l in zip(result["preds"], result["labels"])],
    })
    df.to_csv(args.out_csv, index=False)

    print(f"Saved {len(df)} samples to {args.out_csv}")
    print(f"Accuracy: {result['acc']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inference-only per-sample evaluation of an existing "
                     "classification checkpoint (no training)."
    )
    p.add_argument("--task", required=True, choices=["modality", "cine_view"])
    p.add_argument("--test_csv", required=True,
                    help="Same held-out test CSV/parquet used for all three methods")
    p.add_argument("--ckpt_path", required=True,
                    help="Path to probe_best.pt or ft_best.pt from train_classification.py")
    p.add_argument("--out_csv", required=True,
                    help="Output CSV path for per-sample predictions")
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_slices", type=int, default=3)
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    # Default 0: avoids the CUDA+fork hang-on-exit seen with num_workers>0
    # on the segmentation inference script.
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()
    main(args)