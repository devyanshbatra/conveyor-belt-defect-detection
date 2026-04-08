"""
train.py — Fine-tune YOLO26-seg on belt_roi segmentation labels.

Usage:
    python train.py --data <path_to_data.yaml> [--model yolo26m-seg.pt]
                    [--epochs 150] [--batch 8] [--imgsz 640] [--device 0]
                    [--export-trt]

After training, optionally exports the best checkpoint to a TensorRT engine.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26-seg on belt_roi dataset")
    parser.add_argument("--data",       type=str, required=True,
                        help="Path to data.yaml (YOLO dataset config)")
    parser.add_argument("--model",      type=str, default="yolo26m-seg.pt",
                        help="Base model weights (pretrained). Default: yolo26m-seg.pt")
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--batch",      type=int, default=8,
                        help="Batch size (reduce to 4 if OOM)")
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--device",     type=str, default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--project",    type=str, default="runs/belt_seg")
    parser.add_argument("--name",       type=str, default="yolo26m")
    parser.add_argument("--export-trt", action="store_true",
                        help="Export best.pt to TensorRT FP16 engine after training")
    parser.add_argument("--trt-int8",   action="store_true",
                        help="Use INT8 instead of FP16 for TRT export (needs calibration data)")
    return parser.parse_args()


def train(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ── Augmentation ─────────────────────────────────────────────────────
        # Day/night brightness variation is the hardest challenge in this dataset.
        hsv_h=0.02,       # slight hue shift
        hsv_s=0.5,        # saturation jitter
        hsv_v=0.4,        # brightness jitter (key for day vs night frames)
        degrees=5.0,      # small rotation (belt is mostly straight)
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,       # do NOT flip vertically — belt direction matters
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,   # instance copy-paste, helps for small dataset

        # ── Segmentation ─────────────────────────────────────────────────────
        overlap_mask=True,
        mask_ratio=4,

        # ── Training control ─────────────────────────────────────────────────
        patience=30,      # early stopping — small dataset plateaus fast
        close_mosaic=10,  # disable mosaic last 10 epochs for stable convergence
        save=True,
        save_period=-1,   # only save best/last
        cache=False,      # set True if RAM > 16 GB for speed
        workers=4,
        project=args.project,
        name=args.name,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,  # only belt_roi, treat all as one class
        rect=False,
        cos_lr=True,      # cosine LR schedule
        nbs=64,           # nominal batch size for auto LR scaling
        val=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


def export_tensorrt(weights_path: Path, data_yaml: str, use_int8: bool = False):
    """Export trained .pt to TensorRT engine (FP16 or INT8)."""
    print(f"\nExporting {weights_path} to TensorRT ({'INT8' if use_int8 else 'FP16'})...")
    model = YOLO(str(weights_path))

    export_kwargs = dict(
        format="engine",
        imgsz=640,
        device=0,
        simplify=True,       # ONNX graph simplification before TRT conversion
        workspace=4,         # GB of GPU workspace for TRT engine builder
        batch=1,             # fixed batch=1 for pipeline.py inference
        verbose=False,
    )

    if use_int8:
        # INT8 needs calibration images (subset of training data)
        export_kwargs["int8"] = True
        export_kwargs["data"] = data_yaml
    else:
        export_kwargs["half"] = True   # FP16

    engine_path = model.export(**export_kwargs)
    print(f"TensorRT engine saved: {engine_path}")
    return engine_path


def main():
    args = parse_args()
    best_weights = train(args)

    if args.export_trt:
        if best_weights.exists():
            export_tensorrt(best_weights, args.data, use_int8=args.trt_int8)
        else:
            print(f"WARNING: best.pt not found at {best_weights}, skipping TRT export.")


if __name__ == "__main__":
    main()
