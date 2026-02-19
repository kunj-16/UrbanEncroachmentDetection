import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(11,64,3,1,1), torch.nn.ReLU())
        self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(64,128,3,1,1), torch.nn.ReLU())
        self.pool = torch.nn.MaxPool2d(2)
        self.dec1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(128,64,2,2), torch.nn.ReLU())
        self.out = torch.nn.Conv2d(64,1,1)

    def forward(self,x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.dec1(x2)
        return self.out(x3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load("encroachment_model.pt", map_location=device))
model.eval()

with rasterio.open("urban_encroachment_stack.tif") as src:
    data = src.read()

H,W = data.shape[1:]
patch = 256
stride = 128

pred_map = np.zeros((H,W))

with torch.no_grad():
    for y in range(0, H-patch, stride):
        for x in range(0, W-patch, stride):
            patch_img = data[:,y:y+patch,x:x+patch]
            patch_img = torch.tensor(patch_img).unsqueeze(0).to(device)
            pred = torch.sigmoid(model(patch_img)).cpu().numpy()[0,0]
            pred_map[y:y+patch,x:x+patch] += pred

plt.imshow(pred_map, cmap="hot")
plt.title("Predicted Urban Encroachment Map")
plt.colorbar()
plt.axis("off")
plt.show()