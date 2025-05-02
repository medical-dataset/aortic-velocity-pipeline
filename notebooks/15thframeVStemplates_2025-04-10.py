# -----------------------------------------------
# Register Frame 15 to All 30 Templates (Fixed BSpline MeshSize)
# -----------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from IPython.display import display
from PIL import Image as PILImage

# ------------------ Paths ------------------
input_path = r"P:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/data/20213_2_0_masked.npy"
template_dir = r"P:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/templates/Template10allsimNonrigid"
template_prefix = "template_phase"
template_suffix = "_similarity_10.npy"
save_dir = r"P:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/notebooks/analysis_output/test3"
os.makedirs(save_dir, exist_ok=True)

# ------------------ Load Test Frame ------------------
full_data = np.load(input_path)
frame_index = 15
test_frame = (np.abs(full_data[frame_index]) > 1e-6).astype(np.uint8)

# ------------------ Utility Functions ------------------
def compute_stats(mask):
    area = int(np.sum(mask))
    com_y, com_x = center_of_mass(mask) if area > 0 else (np.nan, np.nan)
    return area, com_y, com_x

def register_to_template_debug(
    moving_mask,
    template,
    bspline_mesh_size=[5, 5],
    interpolator=sitk.sitkNearestNeighbor,
    metric='mattes',
    n_iterations=100,
    frame_idx=None,
    debug_save=False,
    debug_dir="./debug_outputs"
):
    fixed = sitk.GetImageFromArray(template.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_mask.astype(np.float32))

    # Initial rigid transform
    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # BSpline deformable
    bspline_tx = sitk.BSplineTransformInitializer(
        image1=fixed,
        transformDomainMeshSize=bspline_mesh_size,
        order=3
    )

    # Composite transform
    composite_tx = sitk.CompositeTransform(2)
    composite_tx.AddTransform(initial_tx)
    composite_tx.AddTransform(bspline_tx)

    # Registration setup
    registration = sitk.ImageRegistrationMethod()
    if metric.lower() == 'mattes':
        registration.SetMetricAsMattesMutualInformation(32)
    elif metric.lower() == 'meansquares':
        registration.SetMetricAsMeanSquares()
    elif metric.lower() == 'correlation':
        registration.SetMetricAsCorrelation()
    else:
        raise ValueError("Unsupported metric")

    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=n_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000
    )

    registration.SetInitialTransform(composite_tx, inPlace=False)
    registration.SetInterpolator(interpolator)

    final_tx = registration.Execute(fixed, moving)

    # Debug info
    if frame_idx is not None:
        print(f"[Frame {frame_idx}] Transform Params: {np.round(final_tx.GetParameters(), 3)}")

    # Apply transform
    aligned_image = sitk.Resample(moving, fixed, final_tx, interpolator, 0.0)
    aligned_array = sitk.GetArrayFromImage(aligned_image).astype(np.uint8)

    if np.array_equal(aligned_array, moving_mask):
        print(f"[Frame {frame_idx}] ⚠️ Aligned output is identical to input!")
    else:
        print(f"[Frame {frame_idx}] ✅ Aligned output is different from input.")

    # Optional save
    if debug_save and frame_idx is not None:
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, f"aligned_{frame_idx:02d}.npy"), aligned_array)
        np.save(os.path.join(debug_dir, f"template_{frame_idx:02d}.npy"), template)

    return aligned_array

# ------------------ Loop Over All Templates ------------------
rows = []
aligned_masks = []

for i in range(30):
    path = os.path.join(template_dir, f"{template_prefix}_{i:02d}{template_suffix}")
    if not os.path.exists(path):
        print("Missing template:", path)
        continue

    template = np.load(path)
    template = (template > 0.5).astype(np.uint8)

    aligned = register_to_template_debug(
        test_frame, template,
        bspline_mesh_size=[5, 5],
        interpolator=sitk.sitkNearestNeighbor,
        metric='mattes',
        n_iterations=200,
        frame_idx=i
    )

    a_tpl, y_tpl, x_tpl = compute_stats(template)
    a_in, y_in, x_in = compute_stats(test_frame)
    a_out, y_out, x_out = compute_stats(aligned)

    rows.append({
        'Frame': i,
        'Area_template': a_tpl,
        'COM_template_y': y_tpl,
        'COM_template_x': x_tpl,
        'Area_input': a_in,
        'COM_input_y': y_in,
        'COM_input_x': x_in,
        'Area_aligned': a_out,
        'COM_aligned_y': y_out,
        'COM_aligned_x': x_out,
        'Delta_A_after': a_out - a_tpl,
        'Delta_COM_y_after': y_out - y_tpl,
        'Delta_COM_x_after': x_out - x_tpl,
        'Delta_A_input_aligned': a_out - a_in,
        'Delta_COM_y_input_aligned': y_out - y_in,
        'Delta_COM_x_input_aligned': x_out - x_in,
    })

    aligned_masks.append(aligned)

# ------------------ Save Table ------------------
df = pd.DataFrame(rows)
csv_path = os.path.join(save_dir, "frame15_bspline_to_all_templates.csv")
df.to_csv(csv_path, index=False)
print("Saved table to:", csv_path)

# ------------------ Save and Show Image Grid ------------------
aligned_masks = np.array(aligned_masks)
fig, axes = plt.subplots(30, 3, figsize=(12, 60))

for i in range(30):
    axes[i, 0].imshow(test_frame, cmap='gray')
    axes[i, 0].set_title(f"Input Frame {frame_index}", fontsize=8)
    axes[i, 0].axis('off')

    tpl_path = os.path.join(template_dir, f"{template_prefix}_{i:02d}{template_suffix}")
    tpl = np.load(tpl_path)
    tpl = (tpl > 0.5).astype(np.uint8)
    axes[i, 1].imshow(tpl, cmap='gray')
    axes[i, 1].set_title(f"Template {i}", fontsize=8)
    axes[i, 1].axis('off')

    axes[i, 2].imshow(aligned_masks[i], cmap='gray')
    axes[i, 2].set_title(f"Aligned {i}", fontsize=8)
    axes[i, 2].axis('off')

plt.tight_layout()
img_path = os.path.join(save_dir, "frame15_bspline_aligned_to_all_templates.png")
plt.savefig(img_path, dpi=300)
plt.close()
print("Saved visualization:", img_path)

# ------------------ Display Results ------------------
display(df.round(2))

img = PILImage.open(img_path)
plt.figure(figsize=(12, 80))
plt.imshow(img)
plt.axis('off')
plt.title("Frame 15: Input vs. Template vs. Aligned (BSpline)", fontsize=14)
plt.show()
