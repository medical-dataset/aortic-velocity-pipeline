"""
velocity_alignment_comparison.py

This script loads summary CSV results from three different intra-subject alignment methods 
(e.g., Affine using mask, BSpline using mask, BSpline using velocity values) applied to 
aortic velocity MRI data. It merges the metrics into a single DataFrame and generates 
comparative visualizations including violin plots, scatter comparisons, and correlation matrices.

The goal is to evaluate and compare alignment quality across methods using metrics like:
- Temporal variance (max, mean)
- Signal-to-noise ratio (SNR)
- Energy preservation
- Temporal correlation
- Structural similarity (SSIM)

Usage:
- Define the paths to the three summary `.csv` files.
- Run the script to generate and save comparison plots and performance summaries.

Author: Majid
Date: 2025-05-08
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------ Set your paths here ------------------
# affine_path = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\output_affine_internalTemplate_mask_1007337/summary_affine_mask.csv"
# bspline_mask_path = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\output_bspline_intra_mask_1007337/summary_metricsBSpline_mask.csv"
# bspline_velocity_path = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\output_bspline_intra_raw_velocity_1007337/summary_metricsBSpline_velocity.csv"
affine_path = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\inter_subject_affine_10/summary_affine_mask.csv"
bspline_mask_path = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\inter_subject_BSpline_10/summary_metricsBSpline_mask.csv"

# ------------------ Load and Merge ------------------
df_affine = pd.read_csv(affine_path)
df_mask = pd.read_csv(bspline_mask_path)
#df_velocity = pd.read_csv(bspline_velocity_path)

#df = pd.concat([df_affine, df_mask, df_velocity], ignore_index=True)
df = pd.concat([df_affine, df_mask], ignore_index=True)

# Optional: clean method names (in case some rows are missing 'method')
df['method'] = df['method'].fillna('Unknown')

# ------------------ Plotting Settings ------------------
sns.set(style="whitegrid")
palette = {"Affine_mask": "#1f77b4", "BSpline_mask": "#2ca02c", "BSpline_velocity": "#d62728"}

metrics = [
    ("max_variance", "Max Temporal Variance"),
    ("mean_variance", "Mean Temporal Variance"),
    ("snr", "SNR (In/Out Energy Ratio)"),
    ("energy_preservation", "Energy Preservation"),
    ("mean_temporal_correlation", "Mean Temporal Correlation"),
    ("mean_ssim", "Mean SSIM vs. Mean Frame")
]

output_dir = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\comparison_plots10"
os.makedirs(output_dir, exist_ok=True)

# ------------------ Violin Plots ------------------
for metric, label in metrics:
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="method", y=metric, palette=palette, cut=0, inner="box")
    plt.title(f"Distribution of {label}")
    plt.ylabel(label)
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violin_{metric}.png"), dpi=150)
    plt.close()

# ------------------ Scatter Comparisons ------------------
scatter_pairs = [
    ("mean_variance", "snr"),
    ("mean_variance", "energy_preservation"),
    ("mean_variance", "mean_temporal_correlation"),
    ("mean_variance", "mean_ssim")
]

for x_col, y_col in scatter_pairs:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="method", alpha=0.7, palette=palette)
    plt.title(f"{x_col} vs {y_col}")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{x_col}_vs_{y_col}.png"), dpi=150)
    plt.close()

# ------------------ Summary Stats ------------------
print("=== Summary Statistics by Method ===")
summary_stats = df.groupby("method")[
    ['max_variance', 'mean_variance', 'snr', 'energy_preservation',
     'mean_temporal_correlation', 'mean_ssim']
].agg(['mean', 'std', 'median', 'min', 'max'])

# Show in console
print(summary_stats)

# Save to CSV
summary_stats.to_csv(os.path.join(output_dir, "summary_statistics_by_method.csv"))

print(f"\n All plots and summaries saved in: {output_dir}")
