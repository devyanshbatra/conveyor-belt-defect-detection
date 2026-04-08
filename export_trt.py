"""
export_trt.py — Standalone TensorRT export script.

Run this after training to convert best.pt → best.engine (FP16 or INT8).

Usage:
    python export_trt.py --weights runs/belt_seg/yolo26m/weights/best.pt
    python export_trt.py --weights best.pt --int8 --data train/data.yaml
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   type=str, required=True, help="Path to .pt file")
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--batch",     type=int, default=1)
    p.add_argument("--device",    type=int, default=0)
    p.add_argument("--workspace", type=int, default=4,
                   help="TensorRT builder workspace in GB")
    p.add_argument("--int8",      action="store_true",
                   help="Export INT8 (needs --data for calibration)")
    p.add_argument("--data",      type=str, default=None,
                   help="data.yaml for INT8 calibration")
    p.add_argument("--dynamic",   action="store_true",
                   help="Enable dynamic batch/image-size shapes")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    export_kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workspace=args.workspace,
        simplify=True,
        verbose=True,
    )

    if args.dynamic:
        export_kwargs["dynamic"] = True

    if args.int8:
        if not args.data:
            raise ValueError("--data is required for INT8 export (calibration images)")
        export_kwargs["int8"] = True
        export_kwargs["data"] = args.data
        print("Exporting INT8 TensorRT engine (calibration may take a few minutes)...")
    else:
        export_kwargs["half"] = True   # FP16
        print("Exporting FP16 TensorRT engine...")

    engine_path = model.export(**export_kwargs)
    print(f"\nEngine saved: {engine_path}")
    print(f"Use --seg-model {engine_path} in pipeline.py")


if __name__ == "__main__":
    main()
