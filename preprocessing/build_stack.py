# import rasterio
# import numpy as np

# before_path = r"E:\Projects for Resume\terrain_analyzer\src\data\raw\before_2019.tiff"
# after_path = r"E:\Projects for Resume\terrain_analyzer\src\data\raw\after_2024.tiff"

# out_path    = r"E:\Projects for Resume\terrain_analyzer\src\data\processed\urban_encroachment_stack.tif"

import rasterio
import numpy as np
from pathlib import Path

BASE = Path(r"E:/Projects for Resume/terrain_analyzer/src")
DATA_DIR = BASE / "data"
before_path = DATA_DIR / "raw" / "before_2019_agra.tiff"
after_path  = DATA_DIR / "raw" / "after_2025_agra.tiff"
out_path    = DATA_DIR / "processed" / "urban_encroachment_stack.tif"

with rasterio.open(before_path) as src:
    b2_b, b3_b, b4_b, b8_b, b11_b = src.read()
    meta = src.meta

with rasterio.open(after_path) as src:
    b2_a, b3_a, b4_a, b8_a, b11_a = src.read()

# ---- NDVI ----
ndvi_b = (b8_b - b4_b) / (b8_b + b4_b + 1e-6)
ndvi_a = (b8_a - b4_a) / (b8_a + b4_a + 1e-6)
ndvi_d = ndvi_a - ndvi_b

# ---- NDBI ----
ndbi_b = (b11_b - b8_b) / (b11_b + b8_b + 1e-6)
ndbi_a = (b11_a - b8_a) / (b11_a + b8_a + 1e-6)
ndbi_d = ndbi_a - ndbi_b

stack = np.stack([
    # before
    b2_b, b3_b, b4_b, b8_b, b11_b, ndvi_b, ndbi_b,
    # after
    b2_a, b3_a, b4_a, b8_a, b11_a, ndvi_a, ndbi_a,
    # diffs
    ndvi_d, ndbi_d
])

meta.update(count=16, dtype="float32")

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(stack)

print("Saved new 16-band stack:", out_path)