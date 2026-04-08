"""
pipeline.py — Inference pipeline for conveyor belt defect detection.

Detects 'scratch' and 'edge_damage' defects using a two-stage approach:
  Stage 1: YOLO26-seg TensorRT engine → localize belt ROI
  Stage 2: PatchCore anomaly detector  → detect defects within belt crop
  Post:    Position-based classification → scratch vs edge_damage

Usage:
    python pipeline.py --image_dir <path_to_image_folder> \
                       --output_dir <output_folder> \
                       --seg-model  <best.engine or best.pt> \
                       --anomaly-model <anomaly_model/patchcore_model>

Output per image:
    <output_dir>/<image_name>.jpg     — annotated image with bounding boxes
    <output_dir>/<image_name>.json    — detections in required format
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Belt defect inference pipeline")
    parser.add_argument("--image_dir",     type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir",    type=str, required=True,
                        help="Directory to save annotated images and JSON files")
    parser.add_argument("--seg-model",     type=str, default="best.engine",
                        help="Path to YOLO26-seg weights (.engine for TRT, .pt for PyTorch)")
    parser.add_argument("--anomaly-model", type=str, default="anomaly_model/patchcore_clean",
                        help="Path to PatchCore model directory")
    parser.add_argument("--imgsz",         type=int, default=640,
                        help="YOLO inference image size")
    parser.add_argument("--conf",          type=float, default=0.25,
                        help="YOLO belt segmentation confidence threshold")
    parser.add_argument("--device",        type=str, default="cpu",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--anomaly-threshold", type=float, default=None,
                        help="Anomaly score threshold (auto-calibrated if not set)")
    parser.add_argument("--edge-zone",     type=float, default=0.15,
                        help="Fraction of belt width considered 'edge' (default 0.15 = 15%%)")
    parser.add_argument("--min-defect-area", type=int, default=200,
                        help="Minimum pixel area for a defect bounding box")
    parser.add_argument("--crop-size",     type=int, default=256,
                        help="Belt crop size fed to anomaly model")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Belt segmentation — YOLO26-seg via Ultralytics (supports .pt and .engine)
# ─────────────────────────────────────────────────────────────────────────────

class BeltSegmenter:
    def __init__(self, model_path: str, imgsz: int, conf: float, device: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        print(f"BeltSegmenter loaded: {model_path}")

    def get_belt_polygon(self, image: np.ndarray):
        """
        Run YOLO26-seg on image. Returns (polygon_pts, bbox_xyxy) or (None, None).
        polygon_pts: np.ndarray of shape (N, 2) in pixel coords
        bbox_xyxy:   [x_min, y_min, x_max, y_max] tight bbox of the polygon
        """
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return None, None

        # Take highest-confidence belt detection
        best_idx = int(result.boxes.conf.argmax())
        mask_xy = result.masks.xy[best_idx]          # (N, 2) pixel coords
        polygon = mask_xy.astype(np.int32)

        x_min = int(polygon[:, 0].min())
        y_min = int(polygon[:, 1].min())
        x_max = int(polygon[:, 0].max())
        y_max = int(polygon[:, 1].max())

        return polygon, [x_min, y_min, x_max, y_max]


# ─────────────────────────────────────────────────────────────────────────────
# Belt crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_belt_crop(image: np.ndarray, polygon: np.ndarray,
                      belt_bbox: list, crop_size: int):
    """
    Mask + crop the belt region and resize to crop_size x crop_size.
    Returns (crop_rgb, offset_xy) where offset_xy = (x_min, y_min) of the crop
    in the original image, for mapping detections back.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    masked = image.copy()
    masked[mask == 0] = 0

    x_min, y_min, x_max, y_max = belt_bbox
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    crop = masked[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, None

    crop_resized = cv2.resize(crop, (crop_size, crop_size))
    return crop_resized, (x_min, y_min, x_max - x_min, y_max - y_min)


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection — PatchCore via anomalib
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    def __init__(self, model_dir: str, device: str, threshold: float = None):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.device = device
        self._load_model()

    def _load_model(self):
        try:
            from anomalib.models import Patchcore
        except ImportError:
            raise ImportError("Install anomalib: pip install anomalib>=1.0.0")

        # anomalib 2.x: load Lightning checkpoint directly via Patchcore
        ckpt_candidates = sorted(self.model_dir.rglob("*.ckpt"))
        if not ckpt_candidates:
            raise FileNotFoundError(
                f"No .ckpt found in {self.model_dir}. Run patchcore_fit.py first."
            )
        ckpt_path = ckpt_candidates[0]
        print(f"AnomalyDetector loading: {ckpt_path}")

        # Load checkpoint with weights_only=False (safe — we generated this file ourselves)
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=False,       # weights come from checkpoint
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()

        # Load calibrated raw-distance thresholds (computed from clean normals)
        thr_file = self.model_dir / "thresholds.json"
        if thr_file.exists():
            import json
            thr_data = json.loads(thr_file.read_text())
            self.pixel_threshold = float(thr_data["pixel_threshold"])
            if self.threshold is None:
                self.threshold = float(thr_data["image_threshold"])
            print(f"Calibrated thresholds — image: {self.threshold:.2f}, "
                  f"pixel: {self.pixel_threshold:.2f}  (raw KNN distances)")
        else:
            self.pixel_threshold = None
            if self.threshold is None:
                self.threshold = 24.57   # safe fallback
            print(f"Using fallback anomaly threshold: {self.threshold}")

    def predict(self, crop_bgr: np.ndarray):
        """
        Run anomaly detection on a belt crop (BGR numpy array).
        Returns anomaly_map (H x W float32, raw KNN distances) and overall score.
        """
        import torch
        from torchvision import transforms

        # Preprocess: BGR → RGB → tensor → normalize
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(crop_rgb).unsqueeze(0)  # (1, 3, 256, 256)

        with torch.no_grad():
            # Use inner model directly to get raw KNN distances (not post-processed)
            output = self.model.model(tensor)

        # Extract raw anomaly map (pixel-level KNN distances)
        if hasattr(output, 'anomaly_map') and output.anomaly_map is not None:
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
        else:
            anomaly_map = np.zeros((256, 256), dtype=np.float32)

        if hasattr(output, 'pred_score') and output.pred_score is not None:
            score = float(output.pred_score.squeeze().cpu())
        else:
            score = float(anomaly_map.max())

        return anomaly_map, score


# ─────────────────────────────────────────────────────────────────────────────
# Defect localization + classification
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_map_to_bboxes(anomaly_map: np.ndarray, threshold: float,
                           min_area: int, crop_size: int,
                           belt_rect: tuple, edge_zone: float,
                           pixel_threshold: float = None):
    """
    Threshold the anomaly map, find connected components, return list of:
        {
          "class": "scratch" | "edge_damage",
          "bbox":  [x_min, y_min, x_max, y_max]  (original image coords),
          "score": float
        }

    belt_rect: (orig_x_min, orig_y_min, crop_w_orig, crop_h_orig)
    pixel_threshold: pixel-level threshold from model (preferred over image-level)
    """
    orig_x, orig_y, orig_w, orig_h = belt_rect

    # Resize anomaly map back to original crop size for accurate bbox mapping
    amap_resized = cv2.resize(anomaly_map, (orig_w, orig_h))

    # Use raw KNN distance threshold directly (calibrated from clean training normals)
    # pixel_threshold is the 99th-pct distance of clean normal crops (~22.97)
    if pixel_threshold is not None:
        raw_thr = float(pixel_threshold)
    else:
        # Adaptive fallback: 90th percentile of this image (only reasonable if image is mostly clean)
        raw_thr = float(np.percentile(amap_resized, 90))

    # Normalize map for visualization and score reporting only
    amap_min = float(amap_resized.min())
    amap_max = float(amap_resized.max())
    if amap_max > amap_min:
        amap_norm = (amap_resized - amap_min) / (amap_max - amap_min)
    else:
        amap_norm = amap_resized * 0.0   # uniform → no useful signal

    binary = (amap_resized >= raw_thr).astype(np.uint8) * 255

    # Morphological cleanup to reduce noise and merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Reject components covering >70% of belt area (whole-belt false positive)
    belt_area = orig_w * orig_h
    detections = []
    for i in range(1, num_labels):    # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if area > belt_area * 0.7:     # skip whole-belt false positives
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Skip if bbox itself covers most of the belt width
        if w > orig_w * 0.8 and h > orig_h * 0.8:
            continue

        # Map from crop coords → original image coords
        x_min_orig = orig_x + x
        y_min_orig = orig_y + y
        x_max_orig = orig_x + x + w
        y_max_orig = orig_y + y + h

        # Score = mean anomaly score in component (using normalized map)
        component_mask = (labels == i)
        score = float(amap_norm[component_mask].mean())

        # Classify: edge_damage if bbox center-x is in left/right edge zone
        # OR if the bbox is narrow and touches belt edge
        cx_rel = (x + w / 2) / orig_w   # relative position in belt width
        touches_left  = x < orig_w * edge_zone
        touches_right = (x + w) > orig_w * (1.0 - edge_zone)
        narrow = w < orig_w * 0.3       # narrow stripe = likely edge tear

        if (cx_rel < edge_zone or cx_rel > (1.0 - edge_zone) or
                (narrow and (touches_left or touches_right))):
            defect_class = "edge_damage"
        else:
            defect_class = "scratch"

        detections.append({
            "class": defect_class,
            "bbox":  [int(x_min_orig), int(y_min_orig),
                      int(x_max_orig), int(y_max_orig)],
            "score": round(score, 4),
        })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Drawing + JSON output
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "scratch":      (0, 0, 255),    # Red (BGR)
    "edge_damage":  (0, 165, 255),  # Orange (BGR)
}

def draw_detections(image: np.ndarray, detections: list, belt_polygon=None):
    vis = image.copy()

    # Draw belt polygon outline
    if belt_polygon is not None:
        cv2.polylines(vis, [belt_polygon], isClosed=True,
                      color=(0, 255, 0), thickness=2)

    for det in detections:
        cls   = det["class"]
        bbox  = det["bbox"]
        score = det["score"]
        color = CLASS_COLORS.get(cls, (255, 255, 255))

        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)

        label = f"{cls} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def build_json(detections: list) -> dict:
    """Build output JSON in the required format."""
    out = {}
    for idx, det in enumerate(detections, start=1):
        out[str(idx)] = {
            "bbox_coordinates": det["bbox"],
            "class": det["class"],
            "score": round(det["score"], 4),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline loop
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path: Path, segmenter: BeltSegmenter,
                  anomaly_det: AnomalyDetector, args) -> tuple:
    """
    Full pipeline for a single image.
    Returns (annotated_image, detections_list).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  WARNING: Cannot read {image_path}, skipping.")
        return None, []

    # ── Stage 1: belt segmentation ──────────────────────────────────────────
    polygon, belt_bbox = segmenter.get_belt_polygon(image)

    if polygon is None or belt_bbox is None:
        print(f"  WARNING: No belt detected in {image_path.name}")
        # Return original image, no detections
        return image, []

    # ── Crop belt region ────────────────────────────────────────────────────
    crop, belt_rect = extract_belt_crop(image, polygon, belt_bbox, args.crop_size)
    if crop is None:
        return image, []

    # ── Stage 2: anomaly detection ──────────────────────────────────────────
    anomaly_map, score = anomaly_det.predict(crop)

    # Image-level gate: skip localization for clearly normal images
    if score < anomaly_det.threshold:
        return draw_detections(image, [], belt_polygon=polygon), []

    # ── Localize + classify defects ─────────────────────────────────────────
    detections = anomaly_map_to_bboxes(
        anomaly_map=anomaly_map,
        threshold=anomaly_det.threshold,
        min_area=args.min_defect_area,
        crop_size=args.crop_size,
        belt_rect=belt_rect,
        edge_zone=args.edge_zone,
        pixel_threshold=anomaly_det.pixel_threshold,
    )

    # ── Annotate image ──────────────────────────────────────────────────────
    annotated = draw_detections(image, detections, belt_polygon=polygon)

    return annotated, detections


def main():
    args = parse_args()

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading belt segmentation model...")
    segmenter = BeltSegmenter(
        model_path=args.seg_model,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )

    print("Loading anomaly detection model...")
    anomaly_det = AnomalyDetector(
        model_dir=args.anomaly_model,
        device=args.device,
        threshold=args.anomaly_threshold,
    )

    # ── Gather images ────────────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} images → {output_dir}\n")

    # ── Process loop ─────────────────────────────────────────────────────────
    for img_path in image_paths:
        print(f"  {img_path.name}", end=" ... ", flush=True)

        annotated, detections = process_image(img_path, segmenter, anomaly_det, args)

        stem = img_path.stem

        # Save annotated image
        if annotated is not None:
            out_img_path = output_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_img_path), annotated)

        # Save detections JSON
        det_json = build_json(detections)
        out_json_path = output_dir / f"{stem}.json"
        with open(out_json_path, "w") as f:
            json.dump(det_json, f, indent=2)

        n = len(detections)
        classes = [d["class"] for d in detections]
        print(f"{n} detection(s): {classes}")

    print(f"\nDone. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
