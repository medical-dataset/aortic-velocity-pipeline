# deep_autoencoder_maskedloss.py
# Improved 3D Autoencoder with skip connections and shape alignment

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Dataset ----------------
class VelocityDataset(Dataset):
    def __init__(self, folder_path, patch_size=16):
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
        self.folder = folder_path
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_volume = np.load(os.path.join(self.folder, self.files[idx])).astype(np.float32)  # (30, 192, 192)
        cx, cy = 96, 96
        half = self.patch_size // 2
        padded = np.pad(full_volume, ((0, 0), (half, half), (half, half)), mode="constant")
        patch = padded[:, cy:cy+self.patch_size, cx:cx+self.patch_size] / 250.0
        return torch.tensor(patch).unsqueeze(0)  # (1, 30, H, W)

# ---------------- Improved Autoencoder ----------------
class DeepVelocityAutoencoder(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        self.enc_conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_bottleneck = nn.Sequential(
            nn.Linear(64 * 4 * (patch_size // 8) * (patch_size // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * (patch_size // 8) * (patch_size // 8)),
            nn.ReLU()
        )
        self.unflatten = nn.Unflatten(1, (64, 4, patch_size // 8, patch_size // 8))

        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((30, patch_size, patch_size)),
            nn.Conv3d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        skip1 = self.enc_conv1(x)
        encoded = self.enc_conv2(skip1)
        z = self.flatten(encoded)
        z = self.fc_bottleneck(z)
        z = self.unflatten(z)

        if z.shape != encoded.shape:
            encoded = nn.functional.interpolate(encoded, size=z.shape[2:], mode='trilinear', align_corners=False)
        decoded = self.dec_conv1(z + encoded)

        if decoded.shape != skip1.shape:
            skip1 = nn.functional.interpolate(skip1, size=decoded.shape[2:], mode='trilinear', align_corners=False)
        out = self.dec_conv2(decoded + skip1)
        return out, z

# ---------------- Training ----------------
def train(folder, epochs=30, batch_size=8, lr=1e-3, patch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VelocityDataset(folder, patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = DeepVelocityAutoencoder(patch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\n[Epoch {epoch+1}/{epochs}] Starting...")
        for i, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            output, _ = model(batch)
            mask = (batch != 0).float()
            loss = (torch.abs(output - batch) * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

            if i % 5 == 0:
                print(f"  Batch {i+1}/{len(loader)} - Loss: {loss.item():.5f}")

        print(f"Epoch {epoch+1} completed - Avg Loss: {total_loss / len(dataset):.6f}")

    torch.save(model.state_dict(), f"deep_velocity_autoencoder_patch{patch_size}.pth")
    print(" Model saved.")

if __name__ == "__main__":
    train(r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\inter_subject_BSpline_10")
