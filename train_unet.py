import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from tqdm import tqdm

class EncroachDataset(Dataset):
    def __init__(self):
        self.imgs = sorted(glob.glob("dataset/images/*.npy"))
        self.masks = sorted(glob.glob("dataset/masks/*.npy"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = np.load(self.imgs[idx]).astype(np.float32)
        y = np.load(self.masks[idx]).astype(np.float32)
        x = torch.tensor(x)
        y = torch.tensor(y).unsqueeze(0)
        return x, y

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(11,64,3,1,1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128,64,2,2), nn.ReLU())
        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.dec1(x2)
        return self.out(x3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)

loader = DataLoader(EncroachDataset(), batch_size=4, shuffle=True)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    pbar = tqdm(loader)
    for x,y in pbar:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_description(f"Epoch {epoch+1} Loss {loss.item():.4f}")

torch.save(model.state_dict(),"encroachment_model.pt")
print("Model saved.")