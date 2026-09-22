"""
preprocess_sax_seg.py
=====================
Preprocess ACDC and M&Ms datasets for Cine SAX segmentation.

Outputs per-slice 2D npy files (image + mask) and a unified CSV index.

Label mapping (both datasets consistent):
    0 = background
    1 = right ventricle (RV)
    2 = myocardium
    3 = left ventricle (LV)

ACDC structure:
    training/patientXXX/
        patientXXX_frameYY.nii.gz       ← ED or ES image (H, W, S)
        patientXXX_frameYY_gt.nii.gz    ← segmentation mask
        Info.cfg                         ← ED/ES frame indices, disease group

M&Ms structure:
    Training/Labeled/CASEID/
        CASEID_sa.nii.gz                ← 4D cine (H, W, S, T)
        CASEID_sa_gt.nii.gz             ← 4D mask (H, W, S, T)
    211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
        ← ED/ES frame indices, vendor, centre, pathology

Output structure:
    npy_root/
        ACDC/
            train/patientXXX_ED_s00.npy   ← image slice
            train/patientXXX_ED_s00_gt.npy ← mask slice
            ...
        MMs/
            train/CASEID_ED_s00.npy
            ...

CSV columns:
    dataset, split, case_id, frame_type, slice_idx,
    image_path, mask_path,
    vendor, centre, pathology, ed_frame, es_frame,
    H, W, pixel_spacing_x, pixel_spacing_y, slice_thickness
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Image normalisation
# ---------------------------------------------------------------------------

def normalize_zscore(img: np.ndarray) -> np.ndarray:
    """Z-score normalise then clip to [0,1]."""
    img   = img.astype(np.float32)
    mean  = float(img.mean())
    std   = float(img.std())
    img   = (img - mean) / std if std > 1e-6 else img - mean
    return np.clip((img + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# ACDC preprocessing
# ---------------------------------------------------------------------------

def read_acdc_info(cfg_path: str) -> dict:
    """Parse Info.cfg → {ED, ES, Group, Height, Weight, NbFrame}."""
    info = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def preprocess_acdc(
    acdc_root: str,
    npy_root:  str,
    split:     str = "training",  # "training" or "testing"
) -> list[dict]:
    """
    Process all patients in ACDC split.
    Returns list of row dicts for CSV.
    """
    split_dir = os.path.join(acdc_root, split)
    patients  = sorted([
        p for p in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, p))
    ])

    out_split = "train" if split == "training" else "test"
    out_dir   = os.path.join(npy_root, "ACDC", out_split)
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []

    for patient in tqdm(patients, desc=f"ACDC {split}"):
        patient_dir = os.path.join(split_dir, patient)
        cfg_path    = os.path.join(patient_dir, "Info.cfg")

        if not os.path.exists(cfg_path):
            print(f"  [warn] No Info.cfg for {patient}, skipping")
            continue

        info      = read_acdc_info(cfg_path)
        group     = info.get("Group", "UNK")
        ed_frame  = int(info.get("ED", 1))
        es_frame  = int(info.get("ES", 1))

        # Process ED and ES frames
        for frame_type, frame_idx in [("ED", ed_frame), ("ES", es_frame)]:
            img_path = os.path.join(
                patient_dir, f"{patient}_frame{frame_idx:02d}.nii.gz"
            )
            gt_path  = os.path.join(
                patient_dir, f"{patient}_frame{frame_idx:02d}_gt.nii.gz"
            )

            if not os.path.exists(img_path):
                print(f"  [warn] Missing {img_path}")
                continue

            nii_img = nib.load(img_path)
            img_arr = nii_img.get_fdata().astype(np.float32)   # (H, W, S)
            zooms   = nii_img.header.get_zooms()                # (px, py, pz)

            has_gt  = os.path.exists(gt_path)
            gt_arr  = nib.load(gt_path).get_fdata().astype(np.uint8) \
                      if has_gt else np.zeros_like(img_arr, dtype=np.uint8)

            H, W, S = img_arr.shape

            for s in range(S):
                img_slice = normalize_zscore(img_arr[:, :, s])
                gt_slice  = gt_arr[:, :, s]

                # Skip slices with no foreground in training
                if split == "training" and has_gt and gt_slice.max() == 0:
                    continue

                stem      = f"{patient}_{frame_type}_s{s:02d}"
                img_npy   = os.path.join(out_dir, f"{stem}.npy")
                gt_npy    = os.path.join(out_dir, f"{stem}_gt.npy")

                np.save(img_npy, img_slice)
                np.save(gt_npy,  gt_slice)

                rows.append({
                    "dataset":          "ACDC",
                    "split":            out_split,
                    "case_id":          patient,
                    "frame_type":       frame_type,
                    "slice_idx":        s,
                    "image_path":       img_npy,
                    "mask_path":        gt_npy,
                    "has_gt":           has_gt,
                    "vendor":           "Siemens",   # ACDC is single-vendor
                    "centre":           "1",          # ACDC is single-centre
                    "pathology":        group,
                    "ed_frame":         ed_frame,
                    "es_frame":         es_frame,
                    "H":                H,
                    "W":                W,
                    "n_slices":         S,
                    "pixel_spacing_x":  float(zooms[0]),
                    "pixel_spacing_y":  float(zooms[1]),
                    "slice_thickness":  float(zooms[2]),
                })

    return rows


# ---------------------------------------------------------------------------
# M&Ms preprocessing
# ---------------------------------------------------------------------------

def preprocess_mms(
    mms_root:   str,
    npy_root:   str,
    csv_info:   pd.DataFrame,
) -> list[dict]:
    """
    Process M&Ms Labeled training + Validation + Testing splits.
    Returns list of row dicts for CSV.
    """
    # Build lookup: case_id → {ED, ES, Vendor, Centre, Pathology, ...}
    info_map = {}
    for _, row in csv_info.iterrows():
        case_id = str(row["External code"]).strip()
        info_map[case_id] = {
            "vendor":    str(row["VendorName"]).strip(),
            "centre":    str(row["Centre"]).strip(),
            "pathology": str(row["Pathology"]).strip(),
            "ed_frame":  int(row["ED"]),
            "es_frame":  int(row["ES"]),
        }

    splits = {
        "train": os.path.join(mms_root, "Training", "Labeled"),
        "val":   os.path.join(mms_root, "Validation"),
        "test":  os.path.join(mms_root, "Testing"),
    }

    rows: list[dict] = []

    for out_split, split_dir in splits.items():
        if not os.path.isdir(split_dir):
            print(f"  [warn] Split dir not found: {split_dir}")
            continue

        cases   = sorted([
            c for c in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, c))
        ])
        out_dir = os.path.join(npy_root, "MMs", out_split)
        os.makedirs(out_dir, exist_ok=True)

        for case_id in tqdm(cases, desc=f"M&Ms {out_split}"):
            case_dir = os.path.join(split_dir, case_id)
            img_path = os.path.join(case_dir, f"{case_id}_sa.nii.gz")
            gt_path  = os.path.join(case_dir, f"{case_id}_sa_gt.nii.gz")

            if not os.path.exists(img_path):
                print(f"  [warn] Missing {img_path}")
                continue

            nii_img  = nib.load(img_path)
            img_4d   = nii_img.get_fdata().astype(np.float32)  # (H, W, S, T)
            zooms    = nii_img.header.get_zooms()               # (px, py, pz, pt)
            has_gt   = os.path.exists(gt_path)
            gt_4d    = nib.load(gt_path).get_fdata().astype(np.uint8) \
                       if has_gt else np.zeros_like(img_4d, dtype=np.uint8)

            H, W, S, T = img_4d.shape

            # Get ED/ES frame indices from CSV
            meta     = info_map.get(case_id, {})
            ed_frame = meta.get("ed_frame", 0)
            es_frame = meta.get("es_frame", T // 2)

            # Clamp to valid range
            ed_frame = min(ed_frame, T - 1)
            es_frame = min(es_frame, T - 1)

            for frame_type, frame_idx in [("ED", ed_frame), ("ES", es_frame)]:
                img_3d = img_4d[:, :, :, frame_idx]   # (H, W, S)
                gt_3d  = gt_4d[:, :, :, frame_idx]    # (H, W, S)

                for s in range(S):
                    img_slice = normalize_zscore(img_3d[:, :, s])
                    gt_slice  = gt_3d[:, :, s]

                    # Skip background-only slices in training
                    if out_split == "train" and has_gt and gt_slice.max() == 0:
                        continue

                    stem    = f"{case_id}_{frame_type}_s{s:02d}"
                    img_npy = os.path.join(out_dir, f"{stem}.npy")
                    gt_npy  = os.path.join(out_dir, f"{stem}_gt.npy")

                    np.save(img_npy, img_slice)
                    np.save(gt_npy,  gt_slice)

                    rows.append({
                        "dataset":          "MMs",
                        "split":            out_split,
                        "case_id":          case_id,
                        "frame_type":       frame_type,
                        "slice_idx":        s,
                        "image_path":       img_npy,
                        "mask_path":        gt_npy,
                        "has_gt":           has_gt,
                        "vendor":           meta.get("vendor", ""),
                        "centre":           meta.get("centre", ""),
                        "pathology":        meta.get("pathology", ""),
                        "ed_frame":         ed_frame,
                        "es_frame":         es_frame,
                        "H":                H,
                        "W":                W,
                        "n_slices":         S,
                        "pixel_spacing_x":  float(zooms[0]),
                        "pixel_spacing_y":  float(zooms[1]),
                        "slice_thickness":  float(zooms[2]) if len(zooms) > 2 else 0.0,
                    })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.npy_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)

    all_rows: list[dict] = []

    # ---- ACDC ----
    if args.acdc_root:
        print("\n=== Processing ACDC ===")
        for split in ["training", "testing"]:
            split_dir = os.path.join(args.acdc_root, split)
            if os.path.isdir(split_dir):
                rows = preprocess_acdc(args.acdc_root, args.npy_root, split)
                all_rows.extend(rows)
                print(f"  {split}: {len(rows)} slices")

    # ---- M&Ms ----
    if args.mms_root:
        print("\n=== Processing M&Ms ===")
        csv_files = list(Path(args.mms_root).glob("*.csv"))
        if not csv_files:
            print("  [warn] No CSV info file found in mms_root")
            csv_info = pd.DataFrame()
        else:
            csv_info = pd.read_csv(csv_files[0])
            print(f"  Loaded info CSV: {csv_files[0].name}  ({len(csv_info)} rows)")

        rows = preprocess_mms(args.mms_root, args.npy_root, csv_info)
        all_rows.extend(rows)
        print(f"  Total M&Ms slices: {len(rows)}")

    # ---- Save CSV ----
    df = pd.DataFrame(all_rows)
    df.to_csv(args.csv_out, index=False)
    df.to_parquet(args.csv_out.replace(".csv", ".parquet"), index=False)

    print(f"\n{'='*60}")
    print(f"Total slices: {len(df)}")
    print(f"CSV  → {args.csv_out}")
    print(f"Parquet → {args.csv_out.replace('.csv', '.parquet')}")

    print("\nBreakdown:")
    summary = df.groupby(["dataset", "split"]).size().reset_index(name="n_slices")
    print(summary.to_string(index=False))

    print("\nPathology distribution:")
    print(df.groupby(["dataset", "pathology"]).size().reset_index(name="n_slices").to_string(index=False))

    if "vendor" in df.columns:
        print("\nVendor distribution (M&Ms):")
        mms = df[df["dataset"] == "MMs"]
        if not mms.empty:
            print(mms.groupby("vendor").size().reset_index(name="n_slices").to_string(index=False))

    # Quick sanity check
    print("\nSanity check (first 3 rows):")
    cols = ["dataset", "split", "case_id", "frame_type", "slice_idx",
            "pathology", "vendor", "H", "W"]
    print(df[cols].head(3).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess ACDC and M&Ms for Cine SAX segmentation"
    )
    p.add_argument("--acdc_root", type=str, default=None,
                   help="Path to ACDC database root (contains training/ and testing/)")
    p.add_argument("--mms_root",  type=str, default=None,
                   help="Path to M&Ms OpenDataset root (contains Training/, Validation/, Testing/, *.csv)")
    p.add_argument("--npy_root",  type=str, required=True,
                   help="Output root for npy files")
    p.add_argument("--csv_out",   type=str, required=True,
                   help="Output CSV path (.parquet also saved alongside)")
    args = p.parse_args()

    if args.acdc_root is None and args.mms_root is None:
        p.error("At least one of --acdc_root or --mms_root must be specified")

    main(args)