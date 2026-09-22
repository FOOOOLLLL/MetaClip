"""
convert_mat_to_npy.py
=====================
Batch-converts all CMR mat files to per-slice-per-frame npy files,
then writes a unified CSV index for all modalities including Cine.

Cine is handled specially: its npy files already exist under a separate
root (--cine_npy_root). The script scans those npy files, matches them
to the Cine mat files for metadata, and adds them to the same CSV without
re-converting anything.

Unified mat structure
---------------------
All modalities store data under the key 'gtsosimage' with shape:
    (H, W)           — 2D single image
    (H, W, S)        — 3D: S slices, 1 frame
    (H, W, S, T)     — 4D: S slices, T frames
    (H, W, S, 2) for LGE    → complex magnitude: sqrt(r²+i²)
    (H, W, S, 2) for Flow2d → two channels saved as _M and _P sequences

CSV columns (no 'text' column — text is generated at runtime from mat_path)
-----------
mat_path, npy_path, modality, split, center,
vendor, field, scanner, sequence, view, slice_idx, frame_idx,
H, W, n_slices, n_frames
"""

import os
import re
import sys
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODALITIES = [
    "Cine", "LGE", "Mapping", "Flow2d", "Aorta",
    "Perfusion", "T1w", "T2w", "Tagging", "T1rho",
]

VENDOR_FULL = {
    "siemens": "Siemens",
    "uih":     "United Imaging",
    "philips": "Philips",
    "ge":      "GE",
    "canon":   "Canon",
}

FIELD_NORM = {
    "30t": "3.0T", "3t": "3.0T",
    "15t": "1.5T", "1t": "1.5T",
    "055t": "0.55T",
}

COMPLEX_MODALITIES = {"LGE"}          # last dim = [real, imag] → magnitude
FLOW_MODALITIES    = {"Flow2d"}       # last dim = [magnitude, phase] → split
STATIC_MODALITIES  = {"T1w", "T2w", "T1rho"}   # 3D only, no time axis

MAT_KEY = "gtsosimage"

_VENDOR_PREFIXES = tuple(VENDOR_FULL.keys())
_FIELD_RE        = re.compile(r"\d+[.,]?\d*[Tt]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Path parser
# ---------------------------------------------------------------------------

def parse_mat_path(mat_path: str) -> dict:
    parts    = mat_path.replace("\\", "/").split("/")
    filename = os.path.splitext(os.path.basename(mat_path))[0]

    modality = next((p for p in parts if p in ALL_MODALITIES), "unknown")
    split    = next((p for p in parts if p.endswith("Set") and p[0].isupper()), "unknown")
    center   = next((p for p in parts if p.startswith("Center")), "")
    patient  = next((p for p in parts if re.match(r"^P\d+$", p)), "")

    scanner_seg = next(
        (p for p in parts
         if any(p.lower().startswith(v) for v in _VENDOR_PREFIXES)
         and _FIELD_RE.search(p)),
        ""
    )

    vendor, field, scanner = "", "", ""
    if scanner_seg:
        sp = scanner_seg.split("_", 2)
        vendor  = VENDOR_FULL.get(sp[0].lower(), sp[0])
        fr      = sp[1].lower().replace(".", "").replace(",", "") if len(sp) > 1 else ""
        field   = FIELD_NORM.get(fr, sp[1] if len(sp) > 1 else "")
        scanner = sp[2] if len(sp) > 2 else ""

    return dict(
        modality=modality, split=split, center=center,
        vendor=vendor, field=field, scanner=scanner,
        patient=patient, sequence=filename,
    )


# ---------------------------------------------------------------------------
# Image loading and normalisation
# ---------------------------------------------------------------------------

def load_mat_image(mat_path: str) -> np.ndarray:
    """Return float32 array with shape (H, W, S, T)."""
    mat      = sio.loadmat(mat_path)
    data     = mat[MAT_KEY].astype(np.float32)
    modality = parse_mat_path(mat_path)["modality"]

    if data.ndim == 2:
        data = data[:, :, np.newaxis, np.newaxis]
    elif data.ndim == 3:
        data = data[:, :, :, np.newaxis]

    H, W, S, T = data.shape

    if modality in COMPLEX_MODALITIES and T == 2:
        data = np.sqrt(data[:, :, :, 0] ** 2 + data[:, :, :, 1] ** 2)
        data = data[:, :, :, np.newaxis]

    return data   # (H, W, S, T)


def normalize_slice(img: np.ndarray, mode: str = "zscore") -> np.ndarray:
    img = img.astype(np.float32)
    if mode == "zscore":
        mean = float(img.mean())
        std  = float(img.std())
        img  = (img - mean) / std if std >= 1e-6 else img - mean
        img  = np.clip((img + 3.0) / 6.0, 0.0, 1.0)
    elif mode == "minmax":
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn) if mx - mn >= 1e-6 else np.zeros_like(img)
    return img


# ---------------------------------------------------------------------------
# Single-file conversion (non-Cine)
# ---------------------------------------------------------------------------

def convert_one_mat(
    mat_path:  str,
    npy_root:  str,
    normalize: str = "zscore",
    sample_id: int = 0,
) -> list[dict]:
    meta     = parse_mat_path(mat_path)
    modality = meta["modality"]
    split    = meta["split"]
    sequence = meta["sequence"]

    try:
        data = load_mat_image(mat_path)
    except Exception as e:
        print(f"[ERROR] load failed: {mat_path}\n  {e}", file=sys.stderr)
        return []

    H, W, S, T = data.shape
    rows: list[dict] = []

    def _save(arr2d, s, f, suffix=""):
        arr2d    = normalize_slice(arr2d, mode=normalize)
        seq_name = sequence + suffix
        out_dir  = os.path.join(npy_root, modality, split, seq_name)
        os.makedirs(out_dir, exist_ok=True)
        npy_path = os.path.join(out_dir, f"{sample_id:06d}_s{s:03d}_f{f:03d}.npy")
        np.save(npy_path, arr2d)
        return npy_path, seq_name

    def _row(npy_path, s, f, seq_name):
        return {
            "mat_path":  mat_path,
            "npy_path":  npy_path,
            "modality":  modality,
            "split":     split,
            "center":    meta["center"],
            "vendor":    meta["vendor"],
            "field":     meta["field"],
            "scanner":   meta["scanner"],
            "sequence":  seq_name,
            "view":      seq_name,
            "slice_idx": s,
            "frame_idx": f,
            "H": H, "W": W, "n_slices": S, "n_frames": T,
        }

    # Flow: split into magnitude (_M) and phase (_P)
    if modality in FLOW_MODALITIES and T == 2:
        for suffix, ch in [("_M", 0), ("_P", 1)]:
            for s in range(S):
                npy_path, seq_name = _save(data[:, :, s, ch], s, 0, suffix)
                rows.append(_row(npy_path, s, 0, seq_name))
        return rows

    # All other modalities
    for s in range(S):
        for f in range(T):
            npy_path, seq_name = _save(data[:, :, s, f], s, f)
            rows.append(_row(npy_path, s, f, seq_name))

    return rows


# ---------------------------------------------------------------------------
# Cine: read existing CSV index (npy already exist, no conversion needed)
# ---------------------------------------------------------------------------

def index_cine_npy(
    cine_mat_root:  str,
    cine_npy_root:  str,
    cine_csv_paths: list[str] | None = None,
) -> list[dict]:
    """
    Cine npy files already exist and are indexed in existing CSV/parquet files.

    Strategy (in priority order):
      1. If cine_csv_paths provided: read those CSV/parquet files directly.
         This is the fastest and most accurate method.
      2. Otherwise: scan cine_npy_root for npy files and build the index
         by matching against Cine mat files.

    Actual npy directory structure (confirmed):
        cine_npy_root/{split}/Cine/{view}/{sample_id}_s{s:03d}_f{f:03d}.npy
        e.g. npy_slices/TrainingSet/Cine/3ch/036389_s000_f000.npy

    view names in npy dirs: sax, 2ch, 3ch, 4ch, lax, lvot, ...
    view names in mat files: cine_sax, cine_lax_2ch, cine_lax_3ch, ...
    """
    print("Indexing existing Cine npy files ...")

    # ---- Method 1: read existing CSV/parquet ----
    if cine_csv_paths:
        dfs = []
        for p in cine_csv_paths:
            if not os.path.exists(p):
                print(f"  WARNING: CSV not found: {p}")
                continue
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            dfs.append(df)
            print(f"  Read {len(df)} rows from {os.path.basename(p)}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            # Drop old 'text' column — text is now generated at runtime
            df = df.drop(columns=["text"], errors="ignore")

            # Normalise column names to match our schema
            rename = {"set_name": "split"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

            # Ensure required columns exist
            for col in ["center", "vendor", "field", "scanner", "sequence",
                        "view", "H", "W", "n_slices", "n_frames"]:
                if col not in df.columns:
                    df[col] = None

            if "sequence" not in df.columns or df["sequence"].isna().all():
                df["sequence"] = df.get("view", "")

            df["modality"] = "Cine"

            # Ensure split values are standardised
            split_map = {
                "train": "TrainingSet", "TrainingSet": "TrainingSet",
                "val":   "TrainingSet",  # merge val into train
                "test":  "TestSet",     "TestSet":     "TestSet",
            }
            if "split" in df.columns:
                df["split"] = df["split"].map(lambda x: split_map.get(str(x), str(x)))

            COLS = [
                "mat_path", "npy_path", "modality", "split", "center",
                "vendor", "field", "scanner", "sequence", "view",
                "slice_idx", "frame_idx", "H", "W", "n_slices", "n_frames",
            ]
            for c in COLS:
                if c not in df.columns:
                    df[c] = None

            df = df.loc[:, ~df.columns.duplicated()]  # drop duplicate cols
            rows = df[COLS].to_dict(orient="records")
            print(f"  Indexed {len(rows)} Cine npy rows from CSV")
            return rows

    # ---- Method 2: scan npy files + match mat metadata ----
    print("  No CSV provided — scanning npy files ...")

    # Build view-name mapping: npy dir name → mat sequence stem
    # npy dir: "sax"    → mat stem: "cine_sax"
    # npy dir: "3ch"    → mat stem: "cine_lax_3ch"
    # npy dir: "4ch"    → mat stem: "cine_lax_4ch"
    # npy dir: "2ch"    → mat stem: "cine_lax_2ch"
    # npy dir: "lax"    → mat stem: "cine_lax"
    # npy dir: "lvot"   → mat stem: "cine_lvot"
    VIEW_TO_SEQ = {
        "sax":  "cine_sax",
        "lax":  "cine_lax",
        "2ch":  "cine_lax_2ch",
        "3ch":  "cine_lax_3ch",
        "4ch":  "cine_lax_4ch",
        "lvot": "cine_lvot",
        "rvot": "cine_rvot",
        "ot":   "cine_ot",
    }

    # Build mat metadata index: sequence_stem → list of mat metadata dicts
    # (one entry per patient, we only need per-view metadata like vendor/field)
    seq_meta: dict[str, dict] = {}
    cine_mat_dir = os.path.join(cine_mat_root, "Cine")
    if os.path.isdir(cine_mat_dir):
        for root, _, files in os.walk(cine_mat_dir):
            for f in files:
                if not f.endswith(".mat"):
                    continue
                mat_path = os.path.join(root, f)
                meta     = parse_mat_path(mat_path)
                seq      = meta["sequence"]   # e.g. "cine_sax"
                if seq not in seq_meta:
                    seq_meta[seq] = meta      # keep first occurrence per sequence type
    print(f"  Found metadata for {len(seq_meta)} Cine sequence types")

    rows: list[dict] = []
    _SF_RE = re.compile(r"_s(\d+)_f(\d+)$")

    for npy_path in sorted(Path(cine_npy_root).rglob("*.npy")):
        stem  = npy_path.stem
        m     = _SF_RE.search(stem)
        if not m:
            continue
        s_idx    = int(m.group(1))
        f_idx    = int(m.group(2))
        view_dir = npy_path.parent.name    # e.g. "sax", "3ch"
        parts    = str(npy_path).replace("\\", "/").split("/")
        split    = next(
            (p for p in parts if p.endswith("Set") and p[0].isupper()),
            "TrainingSet",
        )

        seq_stem = VIEW_TO_SEQ.get(view_dir, f"cine_{view_dir}")
        meta     = seq_meta.get(seq_stem, {})

        rows.append({
            "mat_path":  meta.get("mat_path", ""),
            "npy_path":  str(npy_path),
            "modality":  "Cine",
            "split":     split,
            "center":    meta.get("center", ""),
            "vendor":    meta.get("vendor", ""),
            "field":     meta.get("field", ""),
            "scanner":   meta.get("scanner", ""),
            "sequence":  seq_stem,
            "view":      view_dir,
            "slice_idx": s_idx,
            "frame_idx": f_idx,
            "H":         None,
            "W":         None,
            "n_slices":  None,
            "n_frames":  None,
        })

    print(f"  Indexed {len(rows)} Cine npy slices")
    return rows


# ---------------------------------------------------------------------------
# Batch conversion — all modalities
# ---------------------------------------------------------------------------

def collect_mat_files(
    data_root:  str,
    modalities: list[str] | None = None,
    splits:     list[str] | None = None,
) -> list[str]:
    allowed_mod   = set(modalities) if modalities else set(ALL_MODALITIES)
    allowed_split = set(splits)     if splits     else None

    mat_files = []
    for mod in allowed_mod:
        mod_dir = os.path.join(data_root, mod)
        if not os.path.isdir(mod_dir):
            continue
        for root, _, files in os.walk(mod_dir):
            if allowed_split:
                parts = root.replace("\\", "/").split("/")
                split = next((p for p in parts if p.endswith("Set")), "")
                if split not in allowed_split:
                    continue
            for f in files:
                if f.endswith(".mat"):
                    mat_files.append(os.path.join(root, f))

    return sorted(mat_files)


def convert_all(
    data_root:      str,
    npy_root:       str,
    csv_out:        str,
    cine_npy_root:  str | None       = None,
    cine_csv_paths: list[str] | None = None,
    modalities:     list[str] | None = None,
    splits:         list[str] | None = None,
    normalize:      str              = "zscore",
    num_workers:    int              = 4,
) -> pd.DataFrame:
    """
    Convert all non-Cine mat files and index existing Cine npy files.
    Produces a single unified CSV covering all modalities.

    Parameters
    ----------
    data_root      : root dir with Cine/, LGE/, Mapping/, ... sub-dirs
    npy_root       : where to write new npy files (non-Cine)
    csv_out        : output CSV path (.parquet also saved alongside)
    cine_npy_root  : path to existing Cine npy root (used if no cine_csv_paths)
    cine_csv_paths : list of existing Cine CSV/parquet paths — fastest method
    modalities     : modalities to convert (None = all non-Cine)
    splits         : TrainingSet / TestSet filter (None = all)
    normalize      : zscore or minmax
    num_workers    : parallel workers for non-Cine conversion (use 1 on NAS)
    """
    all_rows: list[dict] = []

    # ---- Cine: read from existing CSV (fastest) or scan npy dir ----
    if cine_csv_paths or (cine_npy_root and os.path.isdir(cine_npy_root)):
        cine_rows = index_cine_npy(data_root, cine_npy_root or "", cine_csv_paths)
        all_rows.extend(cine_rows)
    elif modalities and "Cine" in modalities:
        print("Warning: Cine in --modalities but no --cine_csv or --cine_npy_root set.")

    # ---- Non-Cine: convert mat → npy ----
    # Exclude Cine from mat conversion if we already indexed its npy
    non_cine_mods = modalities or [m for m in ALL_MODALITIES if m != "Cine"]
    if cine_npy_root and "Cine" in non_cine_mods:
        non_cine_mods = [m for m in non_cine_mods if m != "Cine"]

    mat_files = collect_mat_files(data_root, non_cine_mods, splits)
    print(f"\nFound {len(mat_files)} non-Cine mat files to convert")

    if mat_files:
        if num_workers <= 1:
            for i, mat_path in enumerate(tqdm(mat_files, desc="Converting")):
                rows = convert_one_mat(mat_path, npy_root, normalize, sample_id=i)
                all_rows.extend(rows)
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                for i, mat_path in enumerate(mat_files):
                    fut = pool.submit(convert_one_mat, mat_path, npy_root, normalize, i)
                    futures[fut] = mat_path
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                    try:
                        all_rows.extend(fut.result())
                    except Exception as e:
                        print(f"[ERROR] {futures[fut]}: {e}", file=sys.stderr)

    if not all_rows:
        print("Warning: no rows generated.")
        return pd.DataFrame()

    # ---- Build and save DataFrame ----
    COLS = [
        "mat_path", "npy_path", "modality", "split", "center",
        "vendor", "field", "scanner", "sequence", "view",
        "slice_idx", "frame_idx", "H", "W", "n_slices", "n_frames",
    ]
    df = pd.DataFrame(all_rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS]
    df = df.sort_values(["modality", "mat_path", "slice_idx", "frame_idx"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    df.to_csv(csv_out, index=False)
    df.to_parquet(csv_out.replace(".csv", ".parquet"), index=False)

    print(f"\nDone. Total rows: {len(df)}")
    print(f"CSV     : {csv_out}")
    print(f"Parquet : {csv_out.replace('.csv', '.parquet')}")
    print("\nPer-modality summary:")
    grp = df.groupby("modality")
    for mod, g in sorted(grp):
        n_seq = g["mat_path"].nunique() if g["mat_path"].notna().any() else 0
        print(f"  {mod:12s}: {n_seq:5d} sequences  {len(g):7d} npy rows")

    return df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def quick_test(data_root: str, npy_root: str, normalize: str = "zscore"):
    test_paths = {
        "LGE":      "LGE/TestSet/GTSOS/Center001/UIH_30T_umr780/P012/lge_lax_2ch.mat",
        "Mapping":  "Mapping/TrainingSet/GTSOS/Center001/UIH_30T_umr780/P001/T1map.mat",
        "Flow2d":   "Flow2d/TestSet/GTSOS/Center006/Siemens_30T_Prisma/P003/flow2d.mat",
        "Aorta":    "Aorta/TrainingSet/GTSOS/Center015/Siemens_30T_Vida/P002/aorta_sag.mat",
        "Tagging":  "Tagging/TrainingSet/GTSOS/Center015/Siemens_30T_Vida/P002/tagging.mat",
        "T1w":      "T1w/TrainingSet/GTSOS/Center003/UIH_30T_umr880/P004/T1w.mat",
        "T2w":      "T2w/TrainingSet/GTSOS/Center001/UIH_30T_umr780/P001/T2w.mat",
        "Perfusion":"Perfusion/TestSet/GTSOS/Center001/UIH_30T_umr780/P012/perfusion.mat",
    }

    print("=" * 60)
    print("Quick test: converting one file per modality")
    print("=" * 60)

    all_rows = []
    for sample_id, (mod, rel_path) in enumerate(test_paths.items()):
        full_path = os.path.join(data_root, rel_path)
        if not os.path.exists(full_path):
            print(f"\n{mod}: SKIP (not found)")
            continue
        print(f"\n{mod}: {rel_path.split('/')[-1]}")
        try:
            data = load_mat_image(full_path)
            print(f"  shape: {data.shape}")
            rows = convert_one_mat(full_path, npy_root, normalize, sample_id)
            print(f"  → {len(rows)} npy files")
            if rows:
                arr = np.load(rows[0]["npy_path"])
                print(f"  npy: {arr.shape}  [{arr.min():.3f}, {arr.max():.3f}]")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if all_rows:
        df = pd.DataFrame(all_rows)
        print("\nCSV preview:")
        print(df[["modality", "sequence", "slice_idx", "frame_idx", "H", "W"]].head(8))
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CMR mat files to npy + unified CSV for all modalities"
    )
    parser.add_argument("--data_root",     type=str, required=True,
                        help="Root dir with Cine/, LGE/, Mapping/, ... sub-dirs")
    parser.add_argument("--npy_root",      type=str, required=True,
                        help="Output root for new npy files (non-Cine)")
    parser.add_argument("--csv_out",       type=str, required=True,
                        help="Output CSV path (.parquet also saved alongside)")
    parser.add_argument("--cine_npy_root", type=str, default=None,
                        help="Path to existing Cine npy root — used if --cine_csv not set")
    parser.add_argument("--cine_csv",      type=str, default=None,
                        help="Comma-separated paths to existing Cine CSV/parquet files "
                             "(fastest — reads npy paths directly from index)")
    parser.add_argument("--modalities",    type=str, default=None,
                        help="Comma-separated modalities (default: all non-Cine)")
    parser.add_argument("--splits",        type=str, default=None,
                        help="TrainingSet,TestSet (default: all)")
    parser.add_argument("--normalize",     type=str, default="zscore",
                        choices=["zscore", "minmax"])
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--test",          action="store_true",
                        help="Quick test: convert one example per modality and exit")

    args = parser.parse_args()

    if args.test:
        quick_test(args.data_root, args.npy_root, args.normalize)
    else:
        modalities = [m.strip() for m in args.modalities.split(",")] if args.modalities else None
        splits     = [s.strip() for s in args.splits.split(",")]     if args.splits     else None
        cine_csv_paths = (
            [p.strip() for p in args.cine_csv.split(",")]
            if args.cine_csv else None
        )
        convert_all(
            data_root=args.data_root,
            npy_root=args.npy_root,
            csv_out=args.csv_out,
            cine_npy_root=args.cine_npy_root,
            cine_csv_paths=cine_csv_paths,
            modalities=modalities,
            splits=splits,
            normalize=args.normalize,
            num_workers=args.num_workers,
        )