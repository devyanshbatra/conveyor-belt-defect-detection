"""
patchcore_fit_clean.py
Retrain PatchCore on ONLY the clean (undamaged) belt crops.
"""
import sys
sys.argv = ['patchcore_fit_clean.py']

from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

crop_dir  = Path("anomaly_model/crops_clean")
model_dir = Path("anomaly_model/patchcore_clean")
model_dir.mkdir(parents=True, exist_ok=True)

n_train = len(list((crop_dir / "train/good").glob("*.png")))
n_val   = len(list((crop_dir / "test/normal").glob("*.png")))
n_anom  = len(list((crop_dir / "test/anomaly").glob("*.png")))
print(f"Train normal : {n_train}")
print(f"Val normal   : {n_val}")
print(f"Test anomaly : {n_anom}  (real damaged crops for threshold calibration)")

datamodule = Folder(
    name="belt_clean",
    root=str(crop_dir.parent),
    normal_dir="crops_clean/train/good",
    abnormal_dir="crops_clean/test/anomaly",
    normal_test_dir="crops_clean/test/normal",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)

engine = Engine(
    default_root_dir=str(model_dir),
    max_epochs=1,
    accelerator="cpu",
    devices=1,
)

print("\nFitting PatchCore on clean normals only...")
engine.fit(model=model, datamodule=datamodule)
print("\nValidating + calibrating threshold on real damaged crops...")
engine.test(model=model, datamodule=datamodule)
print(f"\nDone. Clean model saved to: {model_dir}")
