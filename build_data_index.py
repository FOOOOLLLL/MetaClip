import re
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd


# ======== 你要改的路径 ========
TXT_PATH = "/media/NAS_R02/USER_PATH/xueyi/data.txt"
OUT_CSV = "/media/NAS_R02/USER_PATH/xueyi/all_samples.csv"


def normalize_vendor_field_scanner(scanner_str: str):
    parts = scanner_str.split("_")
    vendor = parts[0] if len(parts) >= 1 else "Unknown"
    field_raw = parts[1] if len(parts) >= 2 else "Unknown"
    scanner_model = "_".join(parts[2:]) if len(parts) >= 3 else "Unknown"

    field_map = {
        "15T": "1.5T",
        "30T": "3.0T",
        "70T": "7.0T",
    }
    field_strength = field_map.get(field_raw, field_raw)
    return vendor, field_strength, scanner_model


def map_top_modality(modality_dir: str):
    mapping = {
        "Cine": "Cine",
        "LGE": "LGE",
        "Mapping": "Mapping",
        "Flow2d": "Flow2d",
        "Perfusion": "Perfusion",
        "Tagging": "Tagging",
        "Aorta": "Aorta",
        "BlackBlood": "BlackBlood",
        "T1w": "T1w",
        "T1rho": "T1rho",
        "T2w": "T2w",
    }
    return mapping.get(modality_dir, modality_dir)


def infer_view_and_submodality(filename: str, top_modality: str):
    name = Path(filename).stem.lower()

    view = "unknown"
    submodality = name

    if top_modality == "Cine":
        if "2ch" in name:
            view = "2ch"
        elif "3ch" in name:
            view = "3ch"
        elif "4ch" in name:
            view = "4ch"
        elif "sax" in name:
            view = "sax"
        elif "lvot" in name:
            view = "lvot"
        elif "rvot" in name:
            view = "rvot"
        elif "lax" in name:
            view = "lax"
        elif "ot" in name:
            view = "ot"

    elif top_modality == "LGE":
        if "2ch" in name:
            view = "2ch"
        elif "3ch" in name:
            view = "3ch"
        elif "4ch" in name:
            view = "4ch"
        elif "sax" in name:
            view = "sax"
        elif "lax" in name:
            view = "lax"

    elif top_modality == "Flow2d":
        if "inplane" in name:
            view = "inplane"
        elif "throughplane" in name:
            view = "throughplane"

    elif top_modality == "Aorta":
        if "sag" in name:
            view = "aorta_sag"
        elif "tra" in name:
            view = "aorta_tra"

    elif top_modality == "Mapping":
        if "t1mappost" in name:
            submodality = "T1mappost"
        elif "t1map" in name:
            submodality = "T1map"
        elif "t2smap" in name:
            submodality = "T2smap"
        elif "t2map" in name:
            submodality = "T2map"

    return view, submodality


def parse_one_path(p: str) -> Optional[Dict]:
    p = p.strip()
    if not p or not p.endswith(".mat"):
        return None

    path = Path(p)
    parts = path.parts

    # 预期结构:
    # ... / <modality> / <set> / GTSOS / <center> / <scanner> / <Pxxx> / <file.mat>
    if len(parts) < 8:
        return None

    try:
        filename = parts[-1]
        local_patient_id = parts[-2]
        scanner_key = parts[-3]
        center = parts[-4]
        gtsos_tag = parts[-5]
        set_name = parts[-6]
        modality_dir = parts[-7]
    except IndexError:
        return None

    top_modality = map_top_modality(modality_dir)
    vendor, field_strength, scanner_model = normalize_vendor_field_scanner(scanner_key)
    view, submodality = infer_view_and_submodality(filename, top_modality)

    exam_uid = "__".join([set_name, center, scanner_key, local_patient_id])

    return {
        "mat_path": p,
        "filename": filename,
        "set_name": set_name,
        "center": center,
        "scanner_key": scanner_key,
        "vendor": vendor,
        "field_strength": field_strength,
        "scanner_model": scanner_model,
        "local_patient_id": local_patient_id,
        "top_modality": top_modality,
        "submodality": submodality,
        "view": view,
        "exam_uid": exam_uid,
    }


def main():
    rows: List[Dict] = []

    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = parse_one_path(line)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["top_modality", "set_name", "center", "scanner_key", "local_patient_id", "filename"]
    ).reset_index(drop=True)

    print(f"Total valid samples: {len(df)}")
    print(df.head())

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to: {OUT_CSV}")

    print("\nTop modality counts:")
    print(df["top_modality"].value_counts())

    print("\nCenter counts:")
    print(df["center"].value_counts().head(20))


if __name__ == "__main__":
    main()