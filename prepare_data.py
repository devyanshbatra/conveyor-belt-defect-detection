"""
prepare_data.py — Unzip train.zip, create 80/20 train/val split, and fix data.yaml.

Usage:
    python prepare_data.py --zip train.zip --out dataset/
"""

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=str, default="train.zip")
    p.add_argument("--out", type=str, default="dataset")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    zip_path = Path(args.zip)
    out_dir  = Path(args.out)

    # ── Unzip ──────────────────────────────────────────────────────────────
    print(f"Extracting {zip_path} → {out_dir}/raw/ ...")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Skip macOS junk
            if "__MACOSX" in member.filename or member.filename.endswith(".DS_Store"):
                continue
            zf.extract(member, raw_dir)

    # Locate images and labels directories
    images_src = raw_dir / "train" / "train" / "images"
    labels_src = raw_dir / "train" / "train" / "labels"

    if not images_src.exists():
        raise RuntimeError(f"Expected images at {images_src}, not found. Check zip structure.")

    all_images = sorted(images_src.glob("*.jpg")) + sorted(images_src.glob("*.png"))
    random.shuffle(all_images)

    n_val   = max(1, int(len(all_images) * args.val_split))
    val_imgs = all_images[:n_val]
    trn_imgs = all_images[n_val:]

    print(f"Total: {len(all_images)} | Train: {len(trn_imgs)} | Val: {len(val_imgs)}")

    # ── Create split directories ───────────────────────────────────────────
    for split, img_list in [("train", trn_imgs), ("val", val_imgs)]:
        img_dst = out_dir / split / "images"
        lbl_dst = out_dir / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl_path = labels_src / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    # ── Write fixed data.yaml ──────────────────────────────────────────────
    yaml_path = out_dir / "data.yaml"
    yaml_content = f"""\
train: {(out_dir / 'train' / 'images').resolve()}
val:   {(out_dir / 'val'   / 'images').resolve()}

nc: 1
names: ['belt_roi']
"""
    yaml_path.write_text(yaml_content)
    print(f"data.yaml written: {yaml_path}")
    print("\nNext step:")
    print(f"  python train.py --data {yaml_path} --export-trt")


if __name__ == "__main__":
    main()
