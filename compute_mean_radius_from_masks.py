# """
# compute_mean_radius_from_masks.py

# This script calculates an average (mean) aortic radius across all velocity-encoded (VENC) .npy samples in a given folder.
# Each .npy file is assumed to contain a 3D array with shape (30, H, W), representing 30 velocity frames per subject.

# Steps:
# 1. Converts each velocity frame to a binary mask using a small threshold.
# 2. Computes the area (number of non-zero pixels) of the mask for each frame.
# 3. Aggregates all mask areas across all subjects and frames.
# 4. Computes the mean area and derives the equivalent radius assuming a circular shape.

# This radius can be used to build a synthetic circular mask template for registration or standardization.
# """
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# def estimate_mean_radius_from_masks(folder_path, threshold=1e-6):
#     """Estimates average radius based on the average area of binary masks in .npy velocity files."""
#     files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
#     all_areas = []

#     for f in files:
#         path = os.path.join(folder_path, f)
#         data = np.load(path)  # shape: (30, H, W)
#         binary_masks = (np.abs(data) > threshold).astype(np.uint8)
#         frame_areas = binary_masks.sum(axis=(1, 2))  # sum per frame
#         all_areas.extend(frame_areas)

#     mean_area = np.mean(all_areas)
#     mean_radius = np.sqrt(mean_area / np.pi)
#     return mean_radius

# def generate_circular_mask(size, radius, center=None):
#     """Generates a binary mask with a filled circle."""
#     H, W = size
#     if center is None:
#         center = (H // 2, W // 2)
#     Y, X = np.ogrid[:H, :W]
#     dist = (X - center[1]) ** 2 + (Y - center[0]) ** 2
#     mask = dist <= radius ** 2
#     return mask.astype(np.uint8)

# # ==== Usage ====
# input_folder = r"\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\done"
# radius = estimate_mean_radius_from_masks(input_folder)
# print(f"Estimated mean radius: {radius:.2f} pixels")

# template_mask = generate_circular_mask(size=(192, 192), radius=radius)
# np.save("circle_template.npy", template_mask)

# # Optional visualization
# plt.imshow(template_mask, cmap='gray')
# plt.title("Synthetic Circular Template")
# plt.axis('off')
# plt.show()

"""
compute_mean_radius_from_masks.py

This script calculates the distribution and average (mean) aortic radius across all velocity-encoded (VENC) .npy samples in a given folder.
Each .npy file is assumed to contain a 3D array with shape (30, H, W), representing 30 velocity frames per subject.

Steps:
1. Converts each velocity frame to a binary mask using a small threshold.
2. Computes the area (number of non-zero pixels) of the mask for each frame.
3. Derives radius per frame assuming circular shape (r = sqrt(area / pi)).
4. Aggregates all radii and reports statistics (mean, min, max, std).
5. Logs all frames with radius ≤ 5 or ≥ 13 pixels into a CSV file.
6. Generates a synthetic circular template using mean radius.

Useful for understanding mask scale variability and building a standard registration template.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import csv

def estimate_radius_distribution(folder_path, threshold=1e-6, max_files=None,
                                 outlier_csv_path="radius_outliers.csv",
                                 min_radius_thresh=5, max_radius_thresh=13):
    """Estimates per-frame radii, computes statistics, and logs outliers to CSV."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if max_files:
        files = files[:max_files]
    
    all_radii = []
    outlier_records = []

    for f in files:
        path = os.path.join(folder_path, f)
        data = np.load(path)  # shape: (30, H, W)
        binary_masks = (np.abs(data) > threshold).astype(np.uint8)

        for frame_index in range(binary_masks.shape[0]):
            area = np.sum(binary_masks[frame_index])
            radius = np.sqrt(area / np.pi)
            all_radii.append(radius)

            if radius <= min_radius_thresh or radius >= max_radius_thresh:
                outlier_records.append([f, frame_index, round(radius, 2)])

    # Save outliers to CSV
    with open(outlier_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "FrameIndex", "Radius (pixels)"])
        writer.writerows(outlier_records)

    all_radii = np.array(all_radii)
    print("=" * 60)
    print(f"Aortic Radius Statistics from {len(files)} files:")
    print(f"  Mean radius    : {np.mean(all_radii):.2f}")
    print(f"  Min radius     : {np.min(all_radii):.2f}")
    print(f"  Max radius     : {np.max(all_radii):.2f}")
    print(f"  Std deviation  : {np.std(all_radii):.2f}")
    print("=" * 60)
    print(f"Outlier frames (≤{min_radius_thresh}px or ≥{max_radius_thresh}px): {len(outlier_records)} saved to {outlier_csv_path}")
    
    return all_radii

def generate_circular_mask(size, radius, center=None):
    """Generates a binary mask with a filled circle."""
    H, W = size
    if center is None:
        center = (H // 2, W // 2)
    Y, X = np.ogrid[:H, :W]
    dist = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    mask = dist <= radius ** 2
    return mask.astype(np.uint8)

# ==== Configuration ====
input_folder = r"\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\results2"#\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\done
max_files_to_use = None  # 10000 or You can limit processing to N files (or None for all)

# User-specified output paths:
outlier_log_path = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\TemplateCircular2\circle_outliers_resault2_12000.csv"
template_save_path = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\TemplateCircular2\circle_template_resault2_12000.npy"

# ==== Compute ====
all_radii = estimate_radius_distribution(
    folder_path=input_folder,
    threshold=1e-6,
    max_files=max_files_to_use,
    outlier_csv_path=outlier_log_path,
    min_radius_thresh=5,
    max_radius_thresh=13
)

# Save circular template with mean radius
mean_radius = np.mean(all_radii)
template_mask = generate_circular_mask(size=(192, 192), radius=mean_radius)
np.save(template_save_path, template_mask)

# ==== Plotting ====
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(all_radii, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Radius (pixels)")
plt.ylabel("Frequency")
plt.title("Distribution of Aortic Radii Across All Frames")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.imshow(template_mask, cmap='gray')
plt.title(f"Synthetic Circular Template\nRadius = {mean_radius:.2f} px")
plt.axis('off')

plt.tight_layout()
plt.show()
