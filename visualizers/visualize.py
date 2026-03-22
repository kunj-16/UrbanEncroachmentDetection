import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
STACK_PATH = BASE / "src" / "data" / "processed" / "urban_encroachment_stack.tif"
CNN_PATH = BASE / "src" / "results" / "cnn_pred_map.npy"

# ----------------------------
# Load data
# ----------------------------
with rasterio.open(STACK_PATH) as src:
    data = src.read()

cnn = np.load(CNN_PATH)

# Band layout
# 0–4  = before (B2,B3,B4,B8,B11)
#---[5]NDVI before  ---[6]NDBI before
#7-11 = after  (B2,B3,B4,B8,B11) 
# 10   = NDVI before
# 11   = NDVI after
# 12   = NDVI diff  ---NDVI after
# 13   = NDBI before  ---NDBI after
# 14   = NDBI after  ---NDVI diff
# 15   = NDBI diff

# ----------------------------
# RGB composites
# ----------------------------
b2_b, b3_b, b4_b = data[0], data[1], data[2]
b2_a, b3_a, b4_a = data[7], data[8], data[9]

def normalize_rgb(img):
    lo = np.percentile(img, 2)
    hi = np.percentile(img, 98)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

rgb_before = normalize_rgb(np.stack([b4_b, b3_b, b2_b], axis=-1))
rgb_after  = normalize_rgb(np.stack([b4_a, b3_a, b2_a], axis=-1))

# ----------------------------
# Change maps
# ----------------------------
ndvi_diff = data[14]
ndbi_diff = data[15]

cnn_norm = (cnn - cnn.min()) / (cnn.max() - cnn.min() + 1e-6)

ndvi_mask = ndvi_diff < -0.05

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)
plt.title("RGB Before (2019)")
plt.imshow(rgb_before)
plt.axis("off")

plt.subplot(3,3,2)
plt.title("RGB After (2024)")
plt.imshow(rgb_after)
plt.axis("off")

plt.subplot(3,3,3)
plt.title("NDVI Difference")
plt.imshow(ndvi_diff, cmap="RdYlGn")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(3,3,4)
plt.title("NDBI Difference")
plt.imshow(ndbi_diff, cmap="BrBG")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(3,3,5)
plt.title("NDVI-Based Change Mask")
plt.imshow(rgb_after)
plt.imshow(ndvi_mask, cmap="Reds", alpha=0.5)
plt.axis("off")

plt.subplot(3,3,6)
plt.title("CNN Probability Map")
plt.imshow(cnn_norm, cmap="hot")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(3,3,7)
plt.title("CNN Overlay")
plt.imshow(rgb_after)
plt.imshow(cnn_norm, cmap="hot", alpha=0.5)
plt.axis("off")

plt.subplot(3,3,8)
plt.title("CNN High Confidence")
plt.imshow(cnn_norm > 0.85, cmap="gray")
plt.axis("off")

plt.subplot(3,3,9)
plt.hist(cnn_norm.flatten(), bins=100)
plt.title("CNN Confidence Distribution")
plt.xlabel("Probability")

plt.tight_layout()
plt.show()