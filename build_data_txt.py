import os
from pathlib import Path

# 👉 改成你的数据根目录（非常重要）
ROOT_DIR = "/media/NAS_R02/USER_PATH/xueyi"

# 输出文件
OUT_TXT = "/media/NAS_R02/USER_PATH/xueyi/data.txt"


def collect_mat_paths(root_dir):
    root = Path(root_dir)
    all_paths = []

    for path in root.rglob("*.mat"):
        all_paths.append(str(path))

    return all_paths


def main():
    print("Scanning .mat files...")

    paths = collect_mat_paths(ROOT_DIR)

    print(f"Total found: {len(paths)}")

    with open(OUT_TXT, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"Saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()