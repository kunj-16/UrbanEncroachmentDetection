import rasterio
import numpy as np

stack_path = "urban_encroachment_stack.tif"
out_path   = "encroachment_labels.tif"

with rasterio.open(stack_path) as src:
    data = src.read()
    meta = src.meta

ndvi_diff = data[10]   # Band 11

# Rule: vegetation loss threshold
labels = np.zeros_like(ndvi_diff, dtype=np.uint8)
labels[ndvi_diff < -0.15] = 1   # encroachment candidate

meta.update(count=1, dtype="uint8")

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(labels, 1)

print("Saved label mask:", out_path)