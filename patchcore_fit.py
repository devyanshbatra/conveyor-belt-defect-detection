"""
Standalone PatchCore training script — anomalib 2.x compatible.
Uses already-extracted belt crops from anomaly_model/crops/
"""
import sys
sys.argv = ['patchcore_fit.py']

from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

crop_dir  = Path("anomaly_model/crops")
model_dir = Path("anomaly_model/patchcore_model")
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Train crops : {len(list((crop_dir / 'train/good').glob('*.png')))} images")
print(f"Val crops   : {len(list((crop_dir / 'test/normal').glob('*.png')))} images")
print(f"Anomaly dir : {len(list((crop_dir / 'test/anomaly').glob('*.png')))} images (empty — unsupervised)")

datamodule = Folder(
    name="belt_crops",
    root=str(crop_dir.parent),
    normal_dir="crops/train/good",
    abnormal_dir="crops/test/anomaly",
    normal_test_dir="crops/test/normal",
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

print("\nFitting PatchCore memory bank (single pass)...")
engine.fit(model=model, datamodule=datamodule)

print("\nRunning validation...")
engine.test(model=model, datamodule=datamodule)

print(f"\nDone. PatchCore model saved to: {model_dir}")
