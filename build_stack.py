import rasterio
import numpy as np

before_path = "before_2019.tiff"
after_path  = "after_2024.tiff"
out_path    = "urban_encroachment_stack.tif"

with rasterio.open(before_path) as src:
    b2_b, b3_b, b4_b, b8_b = src.read()
    meta = src.meta

with rasterio.open(after_path) as src:
    b2_a, b3_a, b4_a, b8_a = src.read()

ndvi_b = (b8_b - b4_b) / (b8_b + b4_b + 1e-6)
ndvi_a = (b8_a - b4_a) / (b8_a + b4_a + 1e-6)
ndvi_d = ndvi_a - ndvi_b

stack = np.stack([
    b2_b, b3_b, b4_b, b8_b, ndvi_b,
    b2_a, b3_a, b4_a, b8_a, ndvi_a,
    ndvi_d
])

meta.update(count=11, dtype="float32")

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(stack)

print("Saved:", out_path)