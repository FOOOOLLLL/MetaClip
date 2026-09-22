"""
preprocess_emidec.py
====================
Preprocess the EMIDEC dataset for two downstream tasks:

Task A — LGE SAX segmentation (P cases only, 4-class):
    0 = background
    1 = LV cavity
    2 = normal myocardium
    3 = scar (infarcted myocardium label 3 + MVO label 4 merged)

Task B — LGE detection / classification (all 100 cases, binary):
    0 = Normal (Case_N*)
    1 = MI patient (Case_P*)

Directory structure:
    emidec_root/
        Case_N001/
            Images/Case_N001.nii.gz
            Contours/Case_N001.nii.gz
        Case N001.txt   ← clinical info (space in filename)
        ...

Output structure:
    npy_root/
        seg/
            train/Case_P001_s00.npy, Case_P001_s00_gt.npy, ...
            test/ ...
        det/
            train/Case_P001_s00.npy, Case_N001_s00.npy, ...
            test/ ...
    index_seg.parquet   ← for segmentation task
    index_det.parquet   ← for detection task

Train/test split (no official split provided):
    Stratified 80/20 split by case type (N/P), random seed=42.
    N: 26 train / 7 test
    P: 54 train / 13 test
"""

import os
import re
import argparse
import random
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Segmentation label remapping
# Original: 0=BG, 1=LV, 2=normal myo, 3=infarct, 4=MVO
# Output:   0=BG, 1=LV, 2=normal myo, 3=scar (3+4 merged)
SEG_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
SEG_CLASS_NAMES = ["background", "LV cavity", "normal myocardium", "scar"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def normalize_zscore(img: np.ndarray) -> np.ndarray:
    img  = img.astype(np.float32)
    mean = float(img.mean())
    std  = float(img.std())
    img  = (img - mean) / std if std > 1e-6 else img - mean
    return np.clip((img + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)


def remap_labels(gt: np.ndarray) -> np.ndarray:
    """Apply SEG_LABEL_MAP to a GT array."""
    out = np.zeros_like(gt, dtype=np.uint8)
    for src, dst in SEG_LABEL_MAP.items():
        out[gt == src] = dst
    return out


def parse_txt_info(txt_path: str) -> dict:
    """
    Parse a EMIDEC clinical info txt file.
    Returns dict with keys: sex, age, fevg (EF), troponin, etc.
    """
    info = {}
    if not os.path.exists(txt_path):
        return info
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, _, v = line.partition(":")
                info[k.strip().lower().replace(" ", "_")] = v.strip()
    return info


def scan_cases(emidec_root: str) -> tuple[list[str], list[str]]:
    """Return sorted lists of normal and patient case IDs (directory names)."""
    all_dirs = sorted([
        d for d in os.listdir(emidec_root)
        if os.path.isdir(os.path.join(emidec_root, d)) and "_" in d
    ])
    normal  = [d for d in all_dirs if d.startswith("Case_N")]
    patient = [d for d in all_dirs if d.startswith("Case_P")]
    return normal, patient


def stratified_split(
    normal: list[str],
    patient: list[str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Stratified 80/20 train/test split.
    Returns (train_cases, test_cases) where each is a list of case IDs
    with their label (case_id, label) tuples.
    """
    rng = random.Random(seed)

    n_copy = normal[:]
    p_copy = patient[:]
    rng.shuffle(n_copy)
    rng.shuffle(p_copy)

    n_test = max(1, round(len(n_copy) * test_ratio))
    p_test = max(1, round(len(p_copy) * test_ratio))

    n_test_cases = n_copy[:n_test]
    n_train_cases = n_copy[n_test:]
    p_test_cases = p_copy[:p_test]
    p_train_cases = p_copy[p_test:]

    train_cases = [(c, 0) for c in n_train_cases] + [(c, 1) for c in p_train_cases]
    test_cases  = [(c, 0) for c in n_test_cases]  + [(c, 1) for c in p_test_cases]

    rng.shuffle(train_cases)
    rng.shuffle(test_cases)
    return train_cases, test_cases


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def process_case(
    case_id:     str,
    label:       int,   # 0=Normal, 1=MI
    split:       str,   # "train" or "test"
    emidec_root: str,
    npy_root:    str,
    task:        str,   # "seg" or "det"
) -> list[dict]:
    """
    Process one case for one task.
    Returns list of row dicts for CSV.
    """
    img_path = os.path.join(emidec_root, case_id, "Images",   f"{case_id}.nii.gz")
    gt_path  = os.path.join(emidec_root, case_id, "Contours", f"{case_id}.nii.gz")

    if not os.path.exists(img_path):
        print(f"  [warn] Image not found: {img_path}")
        return []

    nii_img = nib.load(img_path)
    img_arr = nii_img.get_fdata().astype(np.float32)   # (H, W, S)
    zooms   = nii_img.header.get_zooms()

    has_gt  = os.path.exists(gt_path)
    gt_raw  = nib.load(gt_path).get_fdata().astype(np.uint8) if has_gt \
              else np.zeros_like(img_arr, dtype=np.uint8)

    # Remap labels for segmentation task
    gt_arr = remap_labels(gt_raw) if task == "seg" else gt_raw

    H, W, S = img_arr.shape

    # Parse clinical txt info
    txt_path = os.path.join(
        emidec_root,
        case_id.replace("_", " ") + ".txt"   # e.g. "Case N001.txt"
    )
    info = parse_txt_info(txt_path)

    out_dir = os.path.join(npy_root, task, split)
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []

    for s in range(S):
        img_slice = normalize_zscore(img_arr[:, :, s])
        gt_slice  = gt_arr[:, :, s]

        # For seg task: skip slices with no foreground labels
        if task == "seg" and split == "train" and has_gt and gt_slice.max() == 0:
            continue

        stem    = f"{case_id}_s{s:02d}"
        img_npy = os.path.join(out_dir, f"{stem}.npy")
        gt_npy  = os.path.join(out_dir, f"{stem}_gt.npy")

        np.save(img_npy, img_slice)
        np.save(gt_npy,  gt_slice)

        rows.append({
            "task":             task,
            "split":            split,
            "case_id":          case_id,
            "case_type":        "N" if label == 0 else "P",
            "detection_label":  label,          # 0=Normal, 1=MI
            "slice_idx":        s,
            "image_path":       img_npy,
            "mask_path":        gt_npy,
            "has_gt":           has_gt,
            "H":                H,
            "W":                W,
            "n_slices":         S,
            "pixel_spacing_x":  float(zooms[0]),
            "pixel_spacing_y":  float(zooms[1]),
            "slice_thickness":  float(zooms[2]) if len(zooms) > 2 else 10.0,
            "sex":              info.get("sex", ""),
            "age":              info.get("age", ""),
            "fevg":             info.get("fevg", ""),
            "troponin":         info.get("troponin", ""),
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(42)

    emidec_root = args.emidec_root
    npy_root    = args.npy_root
    os.makedirs(npy_root, exist_ok=True)

    # ---- Scan cases ----
    normal, patient = scan_cases(emidec_root)
    print(f"Found {len(normal)} Normal cases, {len(patient)} MI cases")

    # ---- Train/test split ----
    train_cases, test_cases = stratified_split(normal, patient,
                                               test_ratio=0.2, seed=42)
    train_n = sum(1 for _, l in train_cases if l == 0)
    train_p = sum(1 for _, l in train_cases if l == 1)
    test_n  = sum(1 for _, l in test_cases  if l == 0)
    test_p  = sum(1 for _, l in test_cases  if l == 1)
    print(f"Train: {len(train_cases)} cases  (N={train_n}, P={train_p})")
    print(f"Test : {len(test_cases)}  cases  (N={test_n},  P={test_p})")

    all_rows: dict[str, list] = {"seg": [], "det": []}

    # ---- Process all cases ----
    for split_name, case_list in [("train", train_cases), ("test", test_cases)]:
        print(f"\nProcessing {split_name} ...")
        for case_id, label in tqdm(case_list, desc=split_name):

            # Segmentation: only P cases have scar labels
            # N cases have LV + normal myo → still useful for seg
            # but only run seg on P cases
            if label == 1:  # P case
                rows = process_case(
                    case_id, label, split_name,
                    emidec_root, npy_root, task="seg",
                )
                all_rows["seg"].extend(rows)

            # Detection: all cases
            rows = process_case(
                case_id, label, split_name,
                emidec_root, npy_root, task="det",
            )
            all_rows["det"].extend(rows)

    # ---- Save CSVs ----
    for task in ["seg", "det"]:
        df = pd.DataFrame(all_rows[task])
        if df.empty:
            print(f"[warn] No rows for task={task}")
            continue

        df = df.sort_values(["split", "case_id", "slice_idx"]).reset_index(drop=True)

        csv_path = os.path.join(npy_root, f"index_{task}.csv")
        parquet_path = csv_path.replace(".csv", ".parquet")
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        print(f"\n{'='*55}")
        print(f"Task: {task.upper()}")
        print(f"  Total slices : {len(df)}")
        print(f"  CSV          : {csv_path}")
        print(f"  Parquet      : {parquet_path}")
        print(f"\n  Split breakdown:")
        g = df.groupby(["split", "case_type"]).agg(
            cases=("case_id", "nunique"),
            slices=("image_path", "count"),
        )
        print(g.to_string())

        if task == "seg":
            # Label distribution per split
            print(f"\n  Scar label distribution (P cases only):")
            for sp in ["train", "test"]:
                sp_df = df[df["split"] == sp]
                if sp_df.empty:
                    continue
                total_vox = 0
                label_vox = {0: 0, 1: 0, 2: 0, 3: 0}
                for _, row in sp_df.iterrows():
                    gt = np.load(row["mask_path"])
                    for k in label_vox:
                        label_vox[k] += int((gt == k).sum())
                        total_vox    += int((gt == k).sum())
                print(f"    {sp}:")
                for k, name in enumerate(SEG_CLASS_NAMES):
                    v = label_vox[k]
                    print(f"      {k} ({name:22s}): {v:7d} "
                          f"({100*v/max(total_vox,1):.1f}%)")

        if task == "det":
            print(f"\n  Detection label distribution:")
            print(df.groupby(["split", "detection_label", "case_type"])
                    .size().reset_index(name="slices").to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess EMIDEC for LGE SAX segmentation and LGE detection"
    )
    p.add_argument("--emidec_root", required=True,
                   help="Path to emidec-dataset-1.0.1/ directory")
    p.add_argument("--npy_root",    required=True,
                   help="Output root for npy files and index CSVs")
    args = p.parse_args()
    main(args)