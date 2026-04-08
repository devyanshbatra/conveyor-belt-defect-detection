"""
anomaly_train.py — Train a PatchCore anomaly detector on normal (undamaged) belt crops.

This script:
  1. Parses YOLO-seg belt_roi labels → extracts belt region from each image
  2. Lets you mark images as 'normal' (no visible damage) for training
  3. Trains a PatchCore model on those normal crops
  4. Saves the trained model to disk for use by pipeline.py

Usage:
    # First run with --inspect to review images and build normal list
    python anomaly_train.py --images train/train/images \
                             --labels train/train/labels \
                             --output anomaly_model/

Dependencies:
    pip install anomalib>=1.0.0 opencv-python pillow tqdm
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchCore on belt crops")
    parser.add_argument("--images",  type=str, required=True,
                        help="Path to YOLO images folder")
    parser.add_argument("--labels",  type=str, required=True,
                        help="Path to YOLO labels folder (belt_roi polygons)")
    parser.add_argument("--output",  type=str, default="anomaly_model",
                        help="Output folder for anomaly model + crops")
    parser.add_argument("--normal-list", type=str, default=None,
                        help="Text file listing filenames to use as 'normal' training set. "
                             "If not given, ALL images are used (assumes mostly normal).")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Resize belt crop to this square size for anomaly model")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2",
                        help="Backbone for PatchCore feature extraction")
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1,
                        help="PatchCore coreset sampling ratio (lower = faster, less memory)")
    parser.add_argument("--device",  type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Belt crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_belt_polygon(label_path: Path, img_w: int, img_h: int):
    """
    Parse a YOLO-seg label file and return pixel-space polygon points.
    Format: class_id x1 y1 x2 y2 ... (normalized)
    """
    with open(label_path) as f:
        line = f.readline().strip().split()
    if not line:
        return None
    # class_id followed by alternating x,y pairs
    coords = list(map(float, line[1:]))
    points = []
    for i in range(0, len(coords) - 1, 2):
        px = int(coords[i] * img_w)
        py = int(coords[i + 1] * img_h)
        points.append([px, py])
    return np.array(points, dtype=np.int32)


def extract_belt_crop(image: np.ndarray, polygon: np.ndarray, crop_size: int):
    """
    Mask image with belt polygon and return a tight axis-aligned crop,
    resized to crop_size x crop_size.
    """
    h, w = image.shape[:2]

    # Create binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Black out non-belt region
    masked = image.copy()
    masked[mask == 0] = 0

    # Crop to bounding rect of belt polygon
    x, y, bw, bh = cv2.boundingRect(polygon)
    crop = masked[y:y + bh, x:x + bw]

    if crop.size == 0:
        return None

    # Resize to square for anomaly model
    crop = cv2.resize(crop, (crop_size, crop_size))
    return crop


def build_crop_dataset(images_dir: Path, labels_dir: Path,
                        out_dir: Path, normal_names: set, crop_size: int):
    """
    Extract belt crops from all images and save to anomalib Folder structure:
        out_dir/
          train/good/        ← normal crops for training
          test/normal/       ← normal crops for validation
          test/anomaly/      ← (empty, filled at inference time)
    """
    train_good = out_dir / "train" / "good"
    test_normal = out_dir / "test" / "normal"
    test_anomaly = out_dir / "test" / "anomaly"

    for d in [train_good, test_normal, test_anomaly]:
        d.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    skipped = 0

    for img_path in tqdm(img_paths, desc="Extracting belt crops"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue

        h, w = image.shape[:2]
        polygon = load_belt_polygon(label_path, w, h)
        if polygon is None:
            skipped += 1
            continue

        crop = extract_belt_crop(image, polygon, crop_size)
        if crop is None:
            skipped += 1
            continue

        stem = img_path.stem
        is_normal = (len(normal_names) == 0) or (img_path.name in normal_names)

        if is_normal:
            # 90% train, 10% val
            import random
            dest = train_good if random.random() < 0.9 else test_normal
        else:
            dest = test_anomaly

        cv2.imwrite(str(dest / f"{stem}.png"), crop)

    print(f"Crops saved to {out_dir}. Skipped {skipped} images.")
    return train_good


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore training
# ─────────────────────────────────────────────────────────────────────────────

def train_patchcore(crop_dataset_dir: Path, output_dir: Path, args):
    """
    Train PatchCore using anomalib 2.x API.
    Saves model checkpoint to output_dir/patchcore_model/
    """
    try:
        from anomalib.data import Folder
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
    except ImportError:
        raise ImportError(
            "anomalib not found. Install with: pip install anomalib>=1.0.0"
        )

    model_dir = output_dir / "patchcore_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # anomalib 2.x Folder datamodule — image_size removed, pass as transform
    datamodule = Folder(
        root=str(crop_dataset_dir.parent),  # parent of train/test folders
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/normal",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,   # 0 workers avoids multiprocessing issues on CPU
        task="detection",
    )

    model = Patchcore(
        backbone=args.backbone,
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=str(model_dir),
        max_epochs=1,   # PatchCore fits in a single pass (memory bank)
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
    )

    print("Fitting PatchCore memory bank (single pass)...")
    engine.fit(model=model, datamodule=datamodule)

    # Save threshold + model state
    engine.test(model=model, datamodule=datamodule)
    print(f"PatchCore model saved to {model_dir}")
    return model_dir


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import random
    random.seed(42)
    np.random.seed(42)

    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normal filenames list if provided
    normal_names = set()
    if args.normal_list and Path(args.normal_list).exists():
        with open(args.normal_list) as f:
            normal_names = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(normal_names)} filenames as 'normal' set.")
    else:
        print("No --normal-list provided. Using ALL images as normal training set.")
        print("NOTE: If some images contain damage, the anomaly model may underperform.")
        print("      Consider creating a normal_list.txt with undamaged image filenames.")

    # Step 1: Extract crops
    crop_dir = output_dir / "crops"
    build_crop_dataset(images_dir, labels_dir, crop_dir, normal_names, args.crop_size)

    # Step 2: Train PatchCore
    train_patchcore(crop_dir, output_dir, args)

    print("\nDone. Use anomaly_model/ in pipeline.py with --anomaly-model flag.")


if __name__ == "__main__":
    main()
