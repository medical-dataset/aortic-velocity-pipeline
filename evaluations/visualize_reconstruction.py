# visualize_reconstruction.py
# Load trained autoencoder and visualize reconstruction vs original on velocity mask patches

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\Analysis")
from train_autoencoder import DeepVelocityAutoencoder, VelocityDataset

# -------- CONFIG --------
PATCH_SIZE = 16
CHECKPOINT = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\deep_velocity_autoencoder_patch16.pth"
DATA_FOLDER = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\inter_subject_BSpline_10"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 5

# -------- Load model --------
model = DeepVelocityAutoencoder(patch_size=PATCH_SIZE).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# -------- Load dataset --------
dataset = VelocityDataset(DATA_FOLDER, patch_size=PATCH_SIZE)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# -------- Visualize --------
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if i >= NUM_SAMPLES:
            break
        batch = batch.to(DEVICE)
        output, _ = model(batch)

        input_np = batch.squeeze().cpu().numpy()       # shape: (30, H, W)
        output_np = output.squeeze().cpu().numpy()     # shape: (30, H, W)

        frame_idx = 15
        in_frame = input_np[frame_idx]
        out_frame = output_np[frame_idx]
        diff = np.abs(in_frame - out_frame)

        print(f"Sample {i+1} — Frame {frame_idx}")
        print(f"Input  → mean: {in_frame.mean():.4f}, std: {in_frame.std():.4f}, min: {in_frame.min():.4f}, max: {in_frame.max():.4f}")
        print(f"Output → mean: {out_frame.mean():.4f}, std: {out_frame.std():.4f}, min: {out_frame.min():.4f}, max: {out_frame.max():.4f}")
        print(f"MAE (mean absolute error): {diff.mean():.5f}")


        # Visualize middle frame
        frame_idx = 15
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(input_np[frame_idx], cmap='bwr', vmin=-1, vmax=1)
        axs[0].set_title("Original")
        axs[1].imshow(output_np[frame_idx], cmap='bwr', vmin=-1, vmax=1)
        axs[1].set_title("Reconstructed")
        for ax in axs:
            ax.axis("off")
        plt.suptitle(f"Sample {i+1} - Frame {frame_idx}")
        plt.tight_layout()
        plt.show()