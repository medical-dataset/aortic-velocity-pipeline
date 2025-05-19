# Aortic Velocity Mask Alignment to Circular Template: Four Methods
# Author: Pardis Research (AI-assisted)

import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates, center_of_mass, shift
from tqdm import tqdm
import skimage.transform as sktf
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
from skimage.transform import warp
from skimage.measure import find_contours
from scipy.interpolate import Rbf
from skimage.transform import warp_polar, warp
from skimage.transform import resize
import pandas as pd
from intra_BSpline_align_velocity_mask import visualize_velocity_alignment, extract_time_profile_and_variance, suppress_high_variance_peaks, analyze_top_variance_peaks, extract_spatiotemporal_profile, analyze_variance_map, combined_velocity_diagnostics, save_visual_outputs_intra_bspline, visualize_deformable_alignment, compute_velocity_alignment_metrics

# ========== Utility: Create Circular Binary Template ==========
def extract_binary_masks(arr, threshold=1e-6):
    return (np.abs(arr) > threshold).astype(np.uint8)

def create_circular_template(radius=10, size=192):
    Y, X = np.ogrid[:size, :size]
    center = (size // 2, size // 2)
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    return (dist <= radius).astype(np.uint8)

def center_velocity_image(velocity, threshold=1e-6):
    binary_mask = (np.abs(velocity) > threshold).astype(np.uint8)

    if np.sum(binary_mask) == 0:
        print("[WARNING] Empty velocity mask — skipping centering.")
        center = (velocity.shape[0] // 2, velocity.shape[1] // 2)
        return velocity.copy(), center

    com_y, com_x = center_of_mass(binary_mask)
    print(f"[DEBUG] COM = ({com_y:.2f}, {com_x:.2f})")

    dy = velocity.shape[0] // 2 - com_y
    dx = velocity.shape[1] // 2 - com_x
    print(f"[DEBUG] shift dy={dy:.2f}, dx={dx:.2f}")

    shifted = shift(velocity, shift=(dy, dx), order=1, mode='constant', cval=0.0)
    print(f"[DEBUG] After shift → max: {np.max(shifted):.4f}, min: {np.min(shifted):.4f}")
    return shifted, (com_y, com_x)

def center_mask_image(mask):
    com_y, com_x = center_of_mass(mask)
    dy = mask.shape[0] // 2 - com_y
    dx = mask.shape[1] // 2 - com_x
    return shift(mask, shift=(dy, dx), order=0, mode='constant', cval=0)

def apply_transform_to_velocity(velocity, template_shape, transform):
    fixed = sitk.GetImageFromArray(np.zeros(template_shape, dtype=np.float32))
    moving = sitk.GetImageFromArray(velocity.astype(np.float32))

    resampled = sitk.Resample(
        moving, fixed, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return sitk.GetArrayFromImage(resampled)

# def align_velocity_with_mask_transform(velocity_array, mask_array, template_mask, transform_func):
#     T, H, W = velocity_array.shape
#     aligned = np.zeros_like(velocity_array)

#     for t in range(T):
#         mask = center_mask_image(mask_array[t])
#         velocity = center_velocity_image(velocity_array[t])
#         transform = transform_func(mask, template_mask)
#         aligned[t] = apply_transform_to_velocity(velocity, template_mask.shape, transform)
#     return aligned

def align_velocity_with_mask_transform(velocity_array, mask_array, template_mask, transform_func):
    T, H, W = velocity_array.shape
    aligned = np.zeros_like(velocity_array)

    for t in range(T):
        mask = mask_array[t]
        velocity = velocity_array[t]

        print(f"[Frame {t}] mask sum: {np.sum(mask)}")
        com_moving = center_of_mass(mask) if np.sum(mask) > 0 else (np.nan, np.nan)
        com_fixed = center_of_mass(template_mask) if np.sum(template_mask) > 0 else (np.nan, np.nan)
        print(f"[Frame {t}] COM moving: {com_moving}, COM fixed: {com_fixed}")

        # --- Centering ---
        mask_centered = center_mask_image(mask)
        velocity_centered = center_velocity_image(velocity)

        # --- Compute Transform ---
        transform = transform_func(mask_centered, template_mask)
        params = transform.GetParameters()
        print(f"[Frame {t}] Transform parameters: {params}")

        #  Check 1: Affine matrix determinant
        if isinstance(transform, sitk.AffineTransform):
            matrix = np.array(transform.GetMatrix()).reshape((2, 2))
            det = np.linalg.det(matrix)
            print(f"[Frame {t}] Affine det: {det:.4f}")
            if np.abs(det) < 1e-3:
                print(f"[WARNING] Frame {t}: Degenerate transform (det ≈ 0), skipping alignment.")
                aligned[t] = velocity_centered  # Or velocity
                continue

        # --- Apply transform ---
        aligned_frame = apply_transform_to_velocity(velocity_centered, template_mask.shape, transform)

        #  Check 2: Mask coverage after alignment
        aligned_mask = extract_binary_masks(aligned_frame, threshold=1e-5)
        nonzero_count = np.count_nonzero(aligned_mask)
        if nonzero_count > 0.9 * aligned_mask.size:
            print(f"[WARNING] Frame {t}: Output mask mostly filled → likely incorrect transform.")
            aligned[t] = velocity_centered  # Or velocity
            continue

        aligned[t] = aligned_frame

        # --- Post-transform debug ---
        print(f"[DEBUG] Aligned mask frame {t} — nonzero pixels: {nonzero_count}")
        print(f"[DEBUG] Aligned frame stats — min: {aligned_frame.min():.2f}, max: {aligned_frame.max():.2f}, std: {aligned_frame.std():.2f}")

    return aligned



# ========== Method 1: Affine (Rigid or Similarity) ==========
def get_affine_transform_to_circle(moving_mask, template_mask):
    # Center the moving mask before registration
    moving_mask = center_mask_image(moving_mask)
    fixed = sitk.GetImageFromArray(template_mask.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_mask.astype(np.float32))
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsRegularStepGradientDescent(
        1.0, 0.01, 200)
    registration.SetInterpolator(sitk.sitkNearestNeighbor)
    registration.SetInitialTransform(initial_transform)

    final_transform = registration.Execute(fixed, moving)
    return final_transform

# ========== Method 2: BSpline ==========
def get_bspline_transform_to_circle(moving_mask, template_mask):
    # Center the moving mask before registration
    moving_mask = center_mask_image(moving_mask)
    fixed = sitk.GetImageFromArray(template_mask.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_mask.astype(np.float32))

    mesh_size = [10, 10]
    initial_transform = sitk.BSplineTransformInitializer(fixed, mesh_size)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.3)
    registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                      numberOfIterations=100)
    registration.SetInterpolator(sitk.sitkNearestNeighbor)
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)
    return final_transform

# ========== Method 3: Polar Warp And Adaptive Polar ==========

# === 1a. Standard Polar Mapping Function ===
def get_polar_mapping_to_circle(mask, target_radius=9.1, output_shape=(192, 192)):
    H, W = output_shape
    cy, cx = H // 2, W // 2

    polar_mask = warp_polar(mask.astype(np.float32), radius=H//2, scaling='linear')
    resized_polar = resize(polar_mask, (int(target_radius * 2), polar_mask.shape[1]), order=1, preserve_range=True)

    polar_full = np.zeros_like(polar_mask)
    pad_top = (H // 2) - int(target_radius)
    polar_full[pad_top:pad_top + int(target_radius * 2), :] = resized_polar

    def polar_inverse_map(coords):
        theta = 2 * np.pi * coords[:, 1] / polar_full.shape[1]
        r = (H // 2) * coords[:, 0] / polar_full.shape[0]
        x = r * np.cos(theta) + cx
        y = r * np.sin(theta) + cy
        return np.stack([y, x], axis=1)

    return polar_inverse_map

# === 1b. Adaptive Polar Mapping Function ===
def get_polar_adaptive_mapping_to_circle(mask, target_radius=9.1, output_shape=(192, 192)):
    H, W = output_shape
    cy, cx = H // 2, W // 2

    contours = find_contours(mask, level=0.5)
    if len(contours) == 0:
        print("[WARNING] Empty mask: polar mapping skipped.")
        return None

    contour = contours[0]
    y, x = contour[:, 0], contour[:, 1]
    dy, dx = y - cy, x - cx
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dy**2 + dx**2)

    sorted_idx = np.argsort(angles)
    angles_sorted = angles[sorted_idx]
    radii_sorted = radii[sorted_idx]
    radius_func = interp1d(angles_sorted, radii_sorted, kind='linear', fill_value='extrapolate')

    def inverse_map(coords):
        coords = np.array(coords)
        theta = 2 * np.pi * coords[:, 1] / output_shape[1]
        r_src = radius_func(theta)
        r_frac = coords[:, 0] / output_shape[0]
        r_new = r_frac * r_src
        x = r_new * np.cos(theta) + cx
        y = r_new * np.sin(theta) + cy
        return np.stack([y, x], axis=1)

    return inverse_map

# === 2. Generalized Polar-Based Velocity Warp ===
def apply_polar_variant_to_velocity(velocity, mask, get_mapping_func, target_radius=10, output_shape=(192, 192)):
    if np.max(np.abs(velocity)) < 1e-3:
        print("[SKIP] Near-zero velocity frame — skipping warp.")
        return velocity.copy()

    mask = center_mask_image(mask)
    velocity, _ = center_velocity_image(velocity)
    inverse_map = get_mapping_func(mask, target_radius=target_radius, output_shape=output_shape)
    if inverse_map is None:
        return velocity.copy()

    warped = warp(
        velocity,
        inverse_map=inverse_map,
        output_shape=output_shape,
        order=1,
        mode='constant',
        cval=0.0,
        preserve_range=True
    )
    return warped

# === 3. Generalized Framewise Alignment ===
def align_velocity_with_polar_variant(velocity_array, get_mapping_func, target_radius=9.1):
    T, H, W = velocity_array.shape
    aligned = np.zeros_like(velocity_array)
    threshold = 1e-6

    for t in range(T):
        velocity = velocity_array[t]
        mask = (np.abs(velocity) > threshold).astype(np.uint8)
        aligned[t] = apply_polar_variant_to_velocity(velocity, mask, get_mapping_func, target_radius=target_radius, output_shape=(H, W))
        print(f"[DEBUG] Frame {t}: max={aligned[t].max():.2f}, nonzero={np.count_nonzero(aligned[t])}")

    return aligned





# ========== Method 4: Polar + Deformable ==========
def get_hybrid_polar_mapping(mask, target_radius=9.1, output_size=192):
    """
    Generate radial deformable mapping from polar mask to circle radius.
    Returns: inverse-warped binary mask
    """
    mask = center_mask_image(mask)
    polar = sktf.warp_polar(mask, radius=output_size // 2)

    # Compute scaling factor for each angle (based on argmax radius)
    radius_profile = np.argmax(polar, axis=0) + 1e-5
    scale_factors = target_radius / radius_profile
    scale_factors = np.clip(scale_factors, 0.5, 2.0)

    # Generate radial coordinate grid
    r_idx = np.arange(polar.shape[0])
    r_scaled = np.outer(scale_factors, r_idx)

    theta_idx = np.tile(np.arange(polar.shape[1]), (polar.shape[0], 1))
    r_coords = np.clip(r_scaled, 0, polar.shape[0] - 1)

    # Deform and inverse-warp to cartesian
    deformed_polar = map_coordinates(polar, [r_coords, theta_idx], order=1)
    cartesian = sktf.warp_polar(deformed_polar, radius=output_size // 2, inverse=True)
    return (cartesian > 0.5).astype(np.uint8)

def apply_hybrid_polar_to_velocity(velocity, target_radius=9.1, output_size=192):
    """
    Apply hybrid polar + radial deformable warp to velocity image.
    Handles empty input gracefully.
    """
    velocity = center_velocity_image(velocity)

    if np.max(np.abs(velocity)) < 1e-3:
        print("[SKIP] Velocity mostly zero — skipping hybrid warp.")
        return velocity.copy()

    polar = sktf.warp_polar(velocity, radius=output_size // 2)

    # Get polar shape
    R, Theta = polar.shape

    # 1. Compute scale factors for each angle (based on argmax radius)
    abs_polar = np.abs(polar)
    radius_profile = np.argmax(abs_polar, axis=0)
    max_vals = np.max(abs_polar, axis=0)

    # Handle angles with no signal
    radius_profile[max_vals < 1e-3] = R // 2  # fallback radius
    scale_factors = target_radius / (radius_profile + 1e-5)
    scale_factors = np.clip(scale_factors, 0.5, 2.0)

    # 2. Create coordinate grids
    r_idx = np.arange(R).reshape(-1, 1)  # Shape (R, 1)
    theta_idx = np.arange(Theta).reshape(1, -1)  # Shape (1, Theta)

    scale_matrix = scale_factors.reshape(1, -1)  # Shape (1, Theta)

    # 3. Apply radial rescaling
    r_coords = scale_matrix * r_idx  # Shape (R, Theta)
    theta_coords = np.tile(theta_idx, (R, 1))  # Shape (R, Theta)

    # 4. Interpolate in polar space
    deformed_polar = map_coordinates(polar, [r_coords, theta_coords], order=1, mode='constant', cval=0.0)

    # 5. Convert back to Cartesian
    def polar_to_cartesian(coords):
        theta = 2 * np.pi * coords[:, 1] / deformed_polar.shape[1]
        r = (output_size // 2) * coords[:, 0] / deformed_polar.shape[0]
        x = r * np.cos(theta) + output_size / 2
        y = r * np.sin(theta) + output_size / 2
        return np.stack([y, x], axis=1)

    inverse = warp(
        deformed_polar,
        inverse_map=polar_to_cartesian,
        output_shape=(output_size, output_size),
        order=1,
        mode='constant',
        cval=0.0,
        preserve_range=True
    )
    return inverse



def align_hybrid_polar_bspline(velocity_array, target_radius=9.1):
    T, H, W = velocity_array.shape
    aligned_velocity = np.zeros_like(velocity_array)

    for t in range(T):
        aligned_velocity[t] = apply_hybrid_polar_to_velocity(
            velocity_array[t], target_radius=target_radius, output_size=H
        )
        print(f"[DEBUG] Frame {t}: max={aligned_velocity[t].max():.2f}, nonzero={np.count_nonzero(aligned_velocity[t])}")


    return aligned_velocity

# ========== Method 5: TPS ==========
from scipy.spatial.distance import pdist, squareform

def get_tps_mapping_to_circle(mask, template_radius=9.1, output_shape=(192, 192)):
    H, W = output_shape
    mask = center_mask_image(mask)

    contours = find_contours(mask, level=0.5)
    if len(contours) == 0:
        return None, None

    src_points = contours[0]  # shape (N, 2)

    # 🧼 حذف نقاط تکراری با threshold فاصله
    dists = squareform(pdist(src_points))
    keep_mask = np.all(dists > 1e-3, axis=1)
    src_points = src_points[keep_mask]

    N = len(src_points)
    if N < 10:
        print("[WARNING] Too few unique contour points for TPS.")
        return None, None

    # تولید نقاط دایره هدف
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    target_y = H // 2 + template_radius * np.sin(angles)
    target_x = W // 2 + template_radius * np.cos(angles)
    dst_points = np.stack([target_y, target_x], axis=1)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    try:
        rbf_y = Rbf(src_points[:, 0], src_points[:, 1], dst_points[:, 0], function='thin_plate')
        rbf_x = Rbf(src_points[:, 0], src_points[:, 1], dst_points[:, 1], function='thin_plate')
    except np.linalg.LinAlgError as e:
        print(f"[TPS Error] Frame skipped due to singular matrix: {e}")
        return None, None

    map_y = rbf_y(yy, xx)
    map_x = rbf_x(yy, xx)
    return map_y, map_x

# Apply the warp
def apply_tps_to_velocity(velocity, map_y, map_x):
    return map_coordinates(velocity, [map_y, map_x], order=1, mode='constant', cval=0.0)

def align_velocity_with_tps(velocity_array, mask_array, target_radius=9.1):
    T, H, W = velocity_array.shape
    aligned_velocity = np.zeros_like(velocity_array)

    for t in range(T):
        velocity = center_velocity_image(velocity_array[t])
        mask = mask_array[t]

        map_y, map_x = get_tps_mapping_to_circle(mask, template_radius=target_radius, output_shape=(H, W))

        if map_y is None or map_x is None:
            print(f"[FALLBACK] Frame {t}: TPS mapping failed — using original velocity.")
            aligned_velocity[t] = velocity  # fallback
        else:
            aligned_velocity[t] = apply_tps_to_velocity(velocity, map_y, map_x)

    return aligned_velocity


def process_inter_subject_alignment(
    input_path,
    output_path,
    axis='horizontal',
    method="bspline",
    target_radius=9.1,
    threshold=1e-6,
    limit=1000
):
    os.makedirs(output_path, exist_ok=True)
    filenames = sorted([f for f in os.listdir(input_path) if f.endswith('.npy')])[:limit]
    summary_rows = []

    circular_template = create_circular_template(radius=target_radius, size=192)
    template_mask = circular_template
    ####Debug
    # template_save_path = os.path.join(output_path, f"circular_template_r{target_radius}.npy")
    # np.save(template_save_path, template_mask)
    # print(f"[DEBUG] Saved circular template at {template_save_path}")

    for i, fname in enumerate(tqdm(filenames)):
        if i % 50 == 0:
            print(f"\n[INFO] Processing sample {i+1}/{len(filenames)}: {fname}")
        path = os.path.join(input_path, fname)
        velocity_array = np.load(path)
        mask_array = extract_binary_masks(velocity_array, threshold=threshold)

        # --- Select & apply alignment method ---
        if method == "affine":
            aligned_velocity = align_velocity_with_mask_transform(
                velocity_array, mask_array, template_mask, get_affine_transform_to_circle
            )
        elif method == "bspline":
            aligned_velocity = align_velocity_with_mask_transform(
                velocity_array, mask_array, template_mask, get_bspline_transform_to_circle
            )
        elif method == "tps":
            aligned_velocity = align_velocity_with_tps(velocity_array, mask_array, target_radius=target_radius)
        elif method == "polar":
            aligned_velocity = align_velocity_with_polar_variant(velocity_array, get_polar_mapping_to_circle, target_radius=10)
        elif method == "polar_adaptive":
            aligned_velocity = align_velocity_with_polar_variant(velocity_array, get_polar_adaptive_mapping_to_circle, target_radius=10)
        elif method == "hybrid":
            aligned_velocity = align_hybrid_polar_bspline(velocity_array, target_radius=target_radius)
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # --- Evaluation ---
        binary_mask = extract_binary_masks(velocity_array, threshold=threshold)
        aligned_mask = extract_binary_masks(aligned_velocity, threshold=threshold)
        print("=" * 80)
        for t in range(velocity_array.shape[0]):
            print(f"[DEBUG] Aligned mask frame {t} — nonzero: {np.count_nonzero(aligned_mask[t])}")
        print("=" * 80)    
        ref_idx = velocity_array.shape[0] // 2  # 
        
        # visualize_velocity_alignment(
        #     velocity_array, aligned_velocity, ref_idx,
        #     output_path=output_path,
        #     base_name=os.path.splitext(fname)[0]
        # )
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(aligned_velocity, aligned_mask, ref_idx)
        methods_needing_cleaning = {"bspline", "tps", "hybrid"}
        if method in methods_needing_cleaning:
            cleaned_variance_map, cleaned_velocity_array = suppress_high_variance_peaks(
                variance_map=variance_map,
                template_mask=aligned_mask[ref_idx],
                z_thresh=3,
                velocity_array=aligned_velocity,
                method='smooth' # or 'zero' or 'none'
            )
            final_velocity = cleaned_velocity_array
            final_variance_map = cleaned_variance_map
        else:
            final_velocity = aligned_velocity
            final_variance_map = variance_map
        
        print("\n[Before Suppression]")
        analyze_top_variance_peaks(variance_map, aligned_mask[ref_idx], num_peaks=5)

        if method in methods_needing_cleaning:
            print("\n[After Suppression]")
            analyze_top_variance_peaks(final_variance_map, aligned_mask[ref_idx], num_peaks=5)

        spatiotemporal_profile = extract_spatiotemporal_profile(final_velocity, axis=axis)
                
        # ------------------ Variance Analysis ------------------
        print("\n[Variance Map Analysis]")
        analyze_variance_map(
            variance_map=final_variance_map,
            template_mask=aligned_mask[ref_idx],  # یا template_mask اگر داری
            threshold=100,
            name=fname
        )

        result = {
            'filename': fname,
            'ref_idx': ref_idx,
            'aligned_velocity': final_velocity,  #
            'time_profile': time_profile,
            'variance_map': final_variance_map,  # 
            'patch_mask': patch_mask,
            'ref_frame_img': final_velocity[ref_idx],  #
            'spatiotemporal_profile': spatiotemporal_profile,
            'binary_mask': binary_mask,
            'aligned_mask': aligned_mask,
            'original_mask': binary_mask  #
        }
        # --- Save result ---
        base = os.path.splitext(fname)[0]
        output_path_npy = os.path.join(output_path, f"{base}_aligned_{method}.npy")
        np.save(output_path_npy, final_velocity)

        additional_metrics = compute_velocity_alignment_metrics(
            aligned_velocity=final_velocity,
            original_velocity=velocity_array,
            template_mask=aligned_mask[ref_idx]
        )
        print("\n[Additional Velocity Metrics]")
        for k, v in additional_metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n[Before Alignment Diagnostics]")
        combined_velocity_diagnostics(velocity_array, binary_mask[ref_idx])
        print("\n[After Alignment Diagnostics]")
        combined_velocity_diagnostics(final_velocity, aligned_mask[ref_idx])

        
        # Save visual + numerical summaries
        ###save_visual_outputs_intra_bspline(result, output_path, template_mask=aligned_mask[ref_idx])
        ###visualize_deformable_alignment(binary_mask, aligned_mask, ref_idx, output_path, base_name=os.path.splitext(fname)[0])
        
        aligned_mask = extract_binary_masks(final_velocity, threshold=threshold)
        print("=" * 80)
        print(f"[{fname}] Frame-wise Radius Summary (From Mask and Velocity)")
        print(f"{'Frame':>5} | {'r_mask_before':>13} | {'r_mask_after':>13} | {'r_vel_before':>13} | {'r_vel_after':>13}")
        print("-" * 80)
        for i in range(velocity_array.shape[0]):
            vel_before_bin = (np.abs(velocity_array[i]) > threshold).astype(np.uint8)
            vel_after_bin = (np.abs(final_velocity[i]) > threshold).astype(np.uint8)

            r_mask_before = np.sqrt(np.sum(binary_mask[i]) / np.pi)
            r_mask_after = np.sqrt(np.sum(aligned_mask[i]) / np.pi)
            r_vel_before = np.sqrt(np.sum(vel_before_bin) / np.pi)
            r_vel_after = np.sqrt(np.sum(vel_after_bin) / np.pi)

            print(f"{i:5d} | {r_mask_before:13.2f} | {r_mask_after:13.2f} | {r_vel_before:13.2f} | {r_vel_after:13.2f}")
        print("=" * 80)

        summary_rows.append({
            'filename': fname,
            'method': method,
            'ref_idx': ref_idx,
            'max_variance': final_variance_map.max(),
            'mean_variance': final_variance_map.mean(),
            'snr': additional_metrics.get('in_out_energy_ratio', np.nan),
            'energy_preservation': additional_metrics.get('energy_preservation', np.nan),
            'mean_temporal_correlation': additional_metrics.get('mean_temporal_correlation', np.nan),
            'mean_ssim': additional_metrics.get('mean_ssim_with_mean_frame', np.nan),
        })

         # --- Save CSV summary ---
        df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(output_path, f"summary_metrics_{method}.csv")
        df.to_csv(summary_csv_path, index=False)
        print(f"\n[Saved Summary CSV] {summary_csv_path}")



if __name__ == "__main__":

    # Define input/output directories
    input_folder = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data"
    #\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\results2
    output_folder = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/output/inter_subject_polar"  # <- تغییر بر اساس روش

    # Define method: 'affine', 'bspline', 'tps', 'polar', 'hybrid'
    method = "polar"  # Change to: 'affine', 'bspline', 'polar', polar_adaptive , 'hybrid', 'tps'

    # Run inter-subject alignment pipeline
    process_inter_subject_alignment(
        input_path=input_folder,
        output_path=output_folder,
        method=method,
        target_radius=10,
        threshold=1e-6,
        limit=1  # Start with 1 for testing/visualization
    )
