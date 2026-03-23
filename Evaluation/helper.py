import matplotlib.pyplot as plt
import numpy as np

# load what you already have
pred = np.load(r"E:\Projects for Resume\terrain_analyzer\src\results\cnn_pred_map.npy")

# normalize
pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)

plt.figure(figsize=(8,8))
plt.imshow(pred_norm, cmap="hot")
plt.title("Click TOP-LEFT of a clear NON-CONSTRUCTION patch")
plt.colorbar()
pts = plt.ginput(5)   # click 5 points
print("Selected points:", pts)
plt.show()
