# feature_extraction.py
# ----------------------------------------------
# This script extracts statistical and temporal features from aortic velocity masks.
# Input: A 30-frame velocity volume (shape: 30×192×192), where each frame contains nonzero values only inside the aortic region.
# For each volume, it extracts:
#  - Per-frame mean, std, and skewness of nonzero velocities in central patch → 30 × 3 = 90 features
#  - Derivatives over the mean velocity time series → 4 features (mean/std of 1st and 2nd derivative)
#  - Additional global stats from the central patch: mean, std, min, max, range, pos/neg ratio → 6 features
# Output: 100-dimensional feature vector per sample
# ----------------------------------------------

import numpy as np
import os
from scipy.stats import skew

def extract_velocity_features(volume, center=(96, 96), patch_size=10):
    """
    Extract meaningful statistical and temporal features from a 30-frame velocity mask volume,
    focusing only on the central region (patch) around the aorta.

    Input:
        volume: numpy array of shape (30, 192, 192), with velocity values in aortic region only
        center: tuple (x, y) for patch center (default is image center)
        patch_size: size of square patch (default 10)

    Output:
        feature_vector: 1D numpy array of 100 features
    """
    features = []
    all_values = []
    cx, cy = center
    half = patch_size // 2

    for t in range(volume.shape[0]):
        patch = volume[t, cy - half:cy + half, cx - half:cx + half]
        nonzero_vals = patch[patch != 0]
        all_values.extend(nonzero_vals.tolist())
        if nonzero_vals.size == 0:
            features.extend([0, 0, 0])  # mean, std, skew
        else:
            features.append(np.mean(nonzero_vals))
            features.append(np.std(nonzero_vals))
            features.append(skew(nonzero_vals))

    mean_series = np.array([np.mean(volume[t, cy - half:cy + half, cx - half:cx + half][volume[t, cy - half:cy + half, cx - half:cx + half] != 0]) if np.any(volume[t, cy - half:cy + half, cx - half:cx + half] != 0) else 0 for t in range(volume.shape[0])])

    d1 = np.diff(mean_series)
    d2 = np.diff(d1)
    features.append(np.mean(d1))
    features.append(np.std(d1))
    features.append(np.mean(d2))
    features.append(np.std(d2))

    all_values = np.array(all_values)
    if all_values.size == 0:
        features.extend([0, 0, 0, 0, 0, 0])
    else:
        features.append(np.mean(all_values))
        features.append(np.std(all_values))
        features.append(np.min(all_values))
        features.append(np.max(all_values))
        features.append(np.max(all_values) - np.min(all_values))
        pos = np.count_nonzero(all_values > 0)
        neg = np.count_nonzero(all_values < 0)
        total = all_values.size
        features.append(pos / total if total > 0 else 0)
        features.append(neg / total if total > 0 else 0)

    return np.array(features)


def extract_features_from_folder(folder_path, max_samples=None):
    """
    Process all .npy files in the given folder and extract features for each.
    Returns:
        X: np.ndarray of shape (n_samples, 100)
        ids: list of filenames
    """
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    if max_samples:
        all_files = all_files[:max_samples]

    X = []
    ids = []

    for idx, fname in enumerate(all_files):
        path = os.path.join(folder_path, fname)
        volume = np.load(path)
        feat = extract_velocity_features(volume)
        X.append(feat)
        ids.append(fname)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_files):
            print(f"[INFO] Processed {idx + 1} / {len(all_files)} samples")

    return np.array(X), ids


# Example usage:
if __name__ == "__main__":
    folder = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\inter_subject_BSpline_10"   # Path to folder containing .npy files
    output_feature_file = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output/velocity_features_patch_1000.npy"
    output_id_file = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output/velocity_ids_patch_1000.txt"

    X, ids = extract_features_from_folder(folder_path=folder, max_samples=1000)

    print("Feature matrix shape:", X.shape)
    print("First sample ID:", ids[0])
    print("First sample features:", X[0][:10])

    # Save to output files
    os.makedirs(os.path.dirname(output_feature_file), exist_ok=True)
    np.save(output_feature_file, X)
    with open(output_id_file, "w") as f:
        for name in ids:
            f.write(f"{name}\n")

    print(f"Saved features to {output_feature_file}")
    print(f"Saved sample IDs to {output_id_file}")
