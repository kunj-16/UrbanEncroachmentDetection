import rasterio
import numpy as np
from pathlib import Path
from manual_negatives import MANUAL_NEGATIVES, PATCH_SIZE


def is_manual_negative(x, y):
    for neg in MANUAL_NEGATIVES:
        # snap negative point to nearest grid location
        gx = (neg["x"] // STRIDE) * STRIDE
        gy = (neg["y"] // STRIDE) * STRIDE

        if x == gx and y == gy:
            print(f"[MANUAL NEGATIVE] snapped ({neg['x']},{neg['y']}) → ({gx},{gy})")
            return True
    return False

BASE_DIR = Path(r"E:/Projects for Resume/terrain_analyzer/src")

DATA_DIR = BASE_DIR / "data"
IMG_PATH = DATA_DIR / "processed" / "urban_encroachment_stack.tif"
MASK_PATH = DATA_DIR / "processed" / "encroachment_labels.tif"

PATCH = 256
STRIDE = 128

IMG_OUT = DATA_DIR / "dataset" / "images"
MASK_OUT = DATA_DIR / "dataset" / "masks"

IMG_OUT.mkdir(parents=True, exist_ok=True)
MASK_OUT.mkdir(parents=True, exist_ok=True)

with rasterio.open(IMG_PATH) as img_src, rasterio.open(MASK_PATH) as mask_src:
    img = img_src.read()
    mask = mask_src.read(1)

H, W = mask.shape
count = 0

for y in range(0, H - PATCH, STRIDE):
    for x in range(0, W - PATCH, STRIDE):
        img_patch = img[:, y:y+PATCH, x:x+PATCH]
        mask_patch = mask[y:y+PATCH, x:x+PATCH]
        if is_manual_negative(x, y):
            mask_patch[:] = 0
            print(f"[MANUAL NEGATIVE] Overriding patch at x={x}, y={y}")

        if mask_patch.sum() < 20:
            continue

        np.save(IMG_OUT / f"img_{count}.npy", img_patch)
        np.save(MASK_OUT / f"mask_{count}.npy", mask_patch)
        count += 1

print("Total patches saved:", count)
