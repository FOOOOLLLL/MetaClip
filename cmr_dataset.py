from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import scipy.io as sio
import torch.nn.functional as F

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def load_mat_auto(mat_path: str) -> Dict[str, Any]:
    """
    尝试读取 .mat 文件。
    先用 scipy.io.loadmat，失败再尝试 h5py。
    """
    mat_path = str(mat_path)

    try:
        data = sio.loadmat(mat_path)
        return data
    except NotImplementedError:
        if not H5PY_AVAILABLE:
            raise
    except Exception:
        if not H5PY_AVAILABLE:
            raise

    # HDF5 style .mat
    with h5py.File(mat_path, "r") as f:
        data = {}
        for k in f.keys():
            data[k] = np.array(f[k])
        return data


def pick_main_array(mat_dict: Dict[str, Any]) -> np.ndarray:
    candidates = []

    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim < 2:
            continue

        shape = v.shape
        size = v.size

        # 优先保留看起来像图像的候选
        # 至少两个维度 > 32，避免选到小辅助矩阵
        large_dims = sum([d > 32 for d in shape])
        if large_dims >= 2:
            candidates.append((k, v, size))

    if not candidates:
        raise ValueError("No valid image-like ndarray found in .mat")

    # 先按 size 排序
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    # 你也可以在这里优先挑 key 名像图像的
    preferred_keywords = ["gtsos", "sos", "img", "image", "recon"]
    for k, v, _ in candidates:
        lk = k.lower()
        if any(word in lk for word in preferred_keywords):
            return np.asarray(v)

    # fallback: 返回最大候选
    return np.asarray(candidates[0][1])

def squeeze_to_image_or_sequence(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        # [H, W, T] -> [T, H, W]
        if arr.shape[-1] <= 64:
            arr = np.transpose(arr, (2, 0, 1))
        # [T, H, W]
        elif arr.shape[0] <= 64:
            pass
        else:
            # fallback
            arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    if arr.ndim == 4:
        # 常见情况 1: [H, W, 1, T]
        if arr.shape[2] == 1:
            arr = arr[:, :, 0, :]
            arr = np.transpose(arr, (2, 0, 1))
            return arr.astype(np.float32)

        # 常见情况 2: [1, H, W, T]
        if arr.shape[0] == 1:
            arr = arr[0]
            if arr.shape[-1] <= 64:
                arr = np.transpose(arr, (2, 0, 1))
            return arr.astype(np.float32)

        # 常见情况 3: [H, W, S, T]，先取中间 slice
        # 比如 mapping / volume / flow 之类
        if arr.shape[-1] <= 64:
            s = arr.shape[2] // 2
            arr = arr[:, :, s, :]
            arr = np.transpose(arr, (2, 0, 1))
            return arr.astype(np.float32)

        raise ValueError(f"Unsupported 4D shape: {arr.shape}")

    raise ValueError(f"Unsupported array shape after squeeze: {arr.shape}")

def normalize_image(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32)
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    img = np.clip(img, vmin, vmax)
    img = (img - img.min()) / (img.max() - img.min() + eps)
    return img.astype(np.float32)


def to_3ch_tensor(img2d: np.ndarray) -> torch.Tensor:
    """
    [H, W] -> [3, H, W]
    """
    x = torch.from_numpy(img2d).float().unsqueeze(0)  # [1, H, W]
    x = x.repeat(3, 1, 1)
    return x

def resize_tensor_image(x: torch.Tensor, target_size=(224, 224)) -> torch.Tensor:
    """
    x: [3, H, W]
    return: [3, target_h, target_w]
    """
    x = x.unsqueeze(0)  # [1, 3, H, W]
    x = F.interpolate(
        x,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )
    x = x.squeeze(0)
    return x

class CMRSingleDataset(Dataset):
    """
    单样本数据集：
    从 all_samples.csv 读取，每次返回一个样本。
    可选取 2D 中间帧。
    """

    def __init__(
        self,
        csv_path: str,
        modality_filter=None,
        split_filter=None,
        return_metadata: bool = True,
        frame_mode: str = "middle",
        target_size=(224, 224),
    ):
        self.df = pd.read_csv(csv_path)

        if modality_filter is not None:
            self.df = self.df[self.df["top_modality"].isin(modality_filter)].copy()

        if split_filter is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split_filter].copy()

        self.df = self.df.reset_index(drop=True)
        self.return_metadata = return_metadata
        self.frame_mode = frame_mode
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, mat_path: str) -> torch.Tensor:
        mat_dict = load_mat_auto(mat_path)

        # 打印所有 key 和 shape，方便定位
        print(f"\n[DEBUG] Loading: {mat_path}")
        for k, v in mat_dict.items():
            if isinstance(v, np.ndarray):
                print(f"  key={k}, shape={v.shape}, dtype={v.dtype}")

        arr = pick_main_array(mat_dict)
        print(f"[DEBUG] picked main array shape: {arr.shape}")

        arr = squeeze_to_image_or_sequence(arr)

        if arr.ndim == 2:
            img = arr
        elif arr.ndim == 3:
            if self.frame_mode == "middle":
                t = arr.shape[0] // 2
                img = arr[t]
            elif self.frame_mode == "first":
                img = arr[0]
            else:
                t = arr.shape[0] // 2
                img = arr[t]
        else:
            raise ValueError(f"Unexpected processed shape: {arr.shape}")

        img = normalize_image(img)
        x = to_3ch_tensor(img)
        x = resize_tensor_image(x, self.target_size)
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._load_image(row["mat_path"])

        sample = {
            "image": x,
            "path": row["mat_path"],
            "modality": row["top_modality"],
            "view": row["view"],
            "exam_uid": row["exam_uid"],
        }

        if self.return_metadata:
            sample["metadata"] = {
                "set_name": row["set_name"],
                "center": row["center"],
                "scanner_key": row["scanner_key"],
                "vendor": row["vendor"],
                "field_strength": row["field_strength"],
                "scanner_model": row["scanner_model"],
                "local_patient_id": row["local_patient_id"],
                "submodality": row["submodality"],
            }

        return sample


class CMRPairDataset(Dataset):
    """
    配对数据集：
    从 positive_pairs.csv 读取，每次返回一对样本。
    适合 contrastive pretraining.
    """

    def __init__(self, pair_csv_path: str, split_filter: Optional[str] = None, target_size=(224, 224)):
        self.df = pd.read_csv(pair_csv_path)

        if split_filter is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split_filter].copy()

        self.df = self.df.reset_index(drop=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def _load_one(self, mat_path: str) -> torch.Tensor:
        mat_dict = load_mat_auto(mat_path)
        arr = pick_main_array(mat_dict)
        arr = squeeze_to_image_or_sequence(arr)

        if arr.ndim == 2:
            img = arr
        elif arr.ndim == 3:
            img = arr[arr.shape[0] // 2]
        else:
            raise ValueError(f"Unexpected processed shape: {arr.shape}")

        img = normalize_image(img)
        x = to_3ch_tensor(img)
        x = resize_tensor_image(x, self.target_size)
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        xa = self._load_one(row["path_a"])
        xb = self._load_one(row["path_b"])

        return {
            "image_a": xa,
            "image_b": xb,
            "path_a": row["path_a"],
            "path_b": row["path_b"],
            "modality_a": row["modality_a"],
            "modality_b": row["modality_b"],
            "view_a": row["view_a"],
            "view_b": row["view_b"],
            "exam_uid": row["exam_uid"],
            "center": row["center"],
            "scanner_key": row["scanner_key"],
            "vendor": row["vendor"],
            "field_strength": row["field_strength"],
        }