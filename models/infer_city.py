import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Model definition (16 bands)
# ----------------------------
class UNet(torch.nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU()
        )
        self.pool = torch.nn.MaxPool2d(2)
        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 2, stride=2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.dec1(x2)
        return self.out(x3)

# ----------------------------
# Paths
# ----------------------------
BASE = Path(__file__).resolve().parents[2]
STACK_PATH = BASE / "src" / "data" / "processed" / "urban_encroachment_stack.tif"
MODEL_PATH = BASE / "src" / "models" / "encroachment_model.pt"
OUT_PATH   = BASE / "src" / "results" / "cnn_pred_map.npy"

# ----------------------------
# Load model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# Load data
# ----------------------------
with rasterio.open(STACK_PATH) as src:
    data = src.read()  # (16, H, W)

C, H, W = data.shape
assert C == 16, f"Expected 16 bands, got {C}"

# ----------------------------
# Sliding window inference
# ----------------------------
PATCH = 256
STRIDE = 128

pred_map = np.zeros((H, W), dtype=np.float32)
count_map = np.zeros((H, W), dtype=np.float32)

with torch.no_grad():
    for y in range(0, H - PATCH, STRIDE):
        for x in range(0, W - PATCH, STRIDE):
            patch_img = data[:, y:y+PATCH, x:x+PATCH]
            patch_img = torch.tensor(patch_img).unsqueeze(0).to(device)

            pred = torch.sigmoid(model(patch_img)).cpu().numpy()[0, 0]

            pred_map[y:y+PATCH, x:x+PATCH] += pred
            count_map[y:y+PATCH, x:x+PATCH] += 1

# Average overlapping predictions
pred_map /= (count_map + 1e-6)

# ----------------------------
# Normalize + threshold
# ----------------------------
pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-6)
pred_bin = (pred_norm > 0.85).astype(np.uint8)

# ----------------------------
# RGB background (AFTER image)
# Band order reminder:
# 0–6  : before
# 7–13 : after
# ----------------------------
b2_a = data[7]
b3_a = data[8]
b4_a = data[9]

rgb = np.stack([b4_a, b3_a, b2_a], axis=-1)
rgb = (rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) + 1e-6)
rgb = np.clip(rgb, 0, 1)



# ----------------------------
# Visualization (3 panels)
# ----------------------------
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.title("CNN Probability Map")
plt.imshow(pred_norm, cmap="hot")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("CNN Binary Prediction")
plt.imshow(pred_bin, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("CNN Overlay on RGB")
plt.imshow(rgb)
plt.imshow(pred_bin, cmap="Reds", alpha=0.5)
plt.axis("off")

plt.subplot(1, 4, 4)
plt.hist(pred_norm.flatten(), bins=100)
plt.title("Prediction confidence distribution")
plt.xlabel("Probability")
plt.ylabel("Pixel count")
plt.show()

plt.tight_layout()
plt.show()


high = (pred_norm > 0.85).astype(np.uint8)
mid  = ((pred_norm > 0.6) & (pred_norm <= 0.85)).astype(np.uint8)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("High confidence")
plt.imshow(high, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Medium confidence")
plt.imshow(mid, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay (High confidence)")
plt.imshow(rgb)
plt.imshow(high, cmap="Reds", alpha=0.6)
plt.axis("off")

plt.show()


# ----------------------------
# Save output
# ----------------------------
np.save(OUT_PATH, pred_map)
print("Saved CNN prediction map:", OUT_PATH)
