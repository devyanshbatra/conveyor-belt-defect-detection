"""
rebuild_normal_crops.py
Selects low-texture (undamaged) belt crops using Laplacian variance,
rebuilds the PatchCore training folder with only those images.
"""
import cv2, shutil, numpy as np
from pathlib import Path

LAP_THRESHOLD = 30   # images below this are considered "normal"

src   = Path("anomaly_model/crops/train/good")
crops = list(src.glob("*.png"))

scores = {}
for cp in crops:
    img = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    h, w = img.shape
    strip = img[h//4:3*h//4, w//4:3*w//4]
    scores[cp.name] = cv2.Laplacian(strip, cv2.CV_64F).var()

normal = [n for n, v in scores.items() if v < LAP_THRESHOLD]
damaged = [n for n, v in scores.items() if v >= LAP_THRESHOLD]

print(f"Total crops   : {len(crops)}")
print(f"Normal (<{LAP_THRESHOLD}) : {len(normal)}")
print(f"Damaged (≥{LAP_THRESHOLD}): {len(damaged)}")

# --- Rebuild folder structure ---
new_good  = Path("anomaly_model/crops_clean/train/good")
test_norm = Path("anomaly_model/crops_clean/test/normal")
test_anom = Path("anomaly_model/crops_clean/test/anomaly")

for d in [new_good, test_norm, test_anom]:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

import random
random.seed(42)
random.shuffle(normal)
n_val = max(1, int(len(normal) * 0.1))
val_names   = set(normal[:n_val])
train_names = set(normal[n_val:])

for name in train_names:
    shutil.copy2(src / name, new_good / name)
for name in val_names:
    shutil.copy2(src / name, test_norm / name)

# Use a few damaged crops as "anomaly" test set (for threshold calibration)
for name in damaged[:5]:
    shutil.copy2(src / name, test_anom / name)

print(f"\nRebuilt crops_clean/")
print(f"  train/good   : {len(list(new_good.glob('*.png')))} images")
print(f"  test/normal  : {len(list(test_norm.glob('*.png')))} images")
print(f"  test/anomaly : {len(list(test_anom.glob('*.png')))} images  ← REAL damaged samples")
print("\nNext: python patchcore_fit_clean.py")
