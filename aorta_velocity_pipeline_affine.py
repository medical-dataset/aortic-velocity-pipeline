"""
affine_align_pipeline.py

This script performs spatial alignment of 30-frame aortic velocity data (VENC MRI) 
to a circular binary mask template using two-step registration:

1. Center-of-mass (COM) alignment: rigidly centers each binary mask frame.
2. Affine transformation: scales and translates each frame to match a canonical circular shape.

Input:
- Velocity data as 3D NumPy arrays (shape: [30, H, W]), one file per subject.
- A circular reference template (2D NumPy array) for geometric alignment.

Output:
- Aligned velocity arrays (same shape as input)
- Temporal mean velocity profiles (time_profile)
- Spatial variance maps (variance_map)
- Spatiotemporal velocity slices (frame vs. position)
- Diagnostic visualizations: patch overlays, variance crops, GIFs, etc.

Usage:
- Configure `input_folder`, `template_path`, and `output_folder`
- Run the script directly, or import `process_affine_folder()` into another pipeline
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import center_of_mass, shift
from skimage.transform import AffineTransform, warp
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
# --- Helper Functions ---
def extract_binary_masks(arr, threshold=1e-6):
    return (np.abs(arr) > threshold).astype(np.uint8)

def choose_reference_frame(mask_array, method='similarity'):
    """
    Chooses a reference frame from within a subject's mask array using a method.
    """
    num_frames = mask_array.shape[0]
    if method == 'area':
        areas = mask_array.sum(axis=(1, 2))
        mean_area = np.mean(areas)
        return np.argmin(np.abs(areas - mean_area))
    elif method == 'similarity':
        total_scores = [
            np.mean([ssim(mask_array[i], mask_array[j]) for j in range(num_frames) if i != j])
            for i in range(num_frames)
        ]
        return np.argmax(total_scores)
    else:
        raise ValueError("Invalid method: choose 'area' or 'similarity'")
    
def com_align(mask, target_com):
    cy, cx = center_of_mass(mask)
    dy, dx = target_com[0] - cy, target_com[1] - cx
    return shift(mask, (dy, dx), order=0, mode='constant'), (dy, dx)

# def compute_affine_to_template(src_mask, template_mask):
#     from skimage.measure import regionprops
#     props_src = regionprops(src_mask.astype(int))[0]
#     props_tgt = regionprops(template_mask.astype(int))[0]

#     src_c = props_src.centroid
#     tgt_c = props_tgt.centroid
#     src_m = props_src.moments_central
#     tgt_m = props_tgt.moments_central

#     scale_y = np.sqrt(tgt_m[2, 0]) / np.sqrt(src_m[2, 0])
#     scale_x = np.sqrt(tgt_m[0, 2]) / np.sqrt(src_m[0, 2])
#     #print(f"[Affine Transform] Scale: (x={scale_x:.2f}, y={scale_y:.2f}) | Translation: ({tgt_c[1] - scale_x * src_c[1]:.2f}, {tgt_c[0] - scale_y * src_c[0]:.2f})")

#     transform = AffineTransform(scale=(scale_x, scale_y),
#         translation=(tgt_c[1] - scale_x * src_c[1], tgt_c[0] - scale_y * src_c[0]))
#     return transform
def compute_affine_to_template(src_mask, template_mask, min_scale=0.85, max_scale=1.15, force_isotropic=True, frame_idx=None):
    """
    Compute an affine transform from src_mask to template_mask.
    Uses centroid and area to estimate uniform scaling and translation.
    
    Args:
        src_mask (np.ndarray): binary mask to align (H, W)
        template_mask (np.ndarray): reference binary mask (H, W)
        min_scale (float): lower bound for scale factor
        max_scale (float): upper bound for scale factor
        force_isotropic (bool): if True, use the same scale for x and y

    Returns:
        AffineTransform object
    """
    from skimage.measure import regionprops
    from skimage.transform import AffineTransform
    import numpy as np

    props_src = regionprops(src_mask.astype(int))
    props_tgt = regionprops(template_mask.astype(int))

    if not props_src or not props_tgt:
        print("[Warning] Empty regionprops. Returning identity.")
        return AffineTransform()

    props_src = props_src[0]
    props_tgt = props_tgt[0]

    # Compute centroids
    src_c = props_src.centroid
    tgt_c = props_tgt.centroid

    # Compute area-based scale
    area_src = props_src.area
    area_tgt = props_tgt.area
    scale = np.sqrt(area_tgt / (area_src + 1e-6))

    # Clamp scale
    scale = np.clip(scale, min_scale, max_scale)

    if force_isotropic:
        scale_x = scale_y = scale
    else:
        # fallback to moments for anisotropic scaling
        src_m = props_src.moments_central
        tgt_m = props_tgt.moments_central
        scale_y = np.clip(np.sqrt(tgt_m[2, 0]) / np.sqrt(src_m[2, 0] + 1e-6), min_scale, max_scale)
        scale_x = np.clip(np.sqrt(tgt_m[0, 2]) / np.sqrt(src_m[0, 2] + 1e-6), min_scale, max_scale)

    # Compute translation to align centroids
    t_x = tgt_c[1] - scale_x * src_c[1]
    t_y = tgt_c[0] - scale_y * src_c[0]
    if frame_idx is not None:
        print(f"[DEBUG] Frame {frame_idx}: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")

    return AffineTransform(scale=(scale_x, scale_y), translation=(t_x, t_y))


def apply_affine(mask, transform):
    return warp(mask, inverse_map=transform.inverse, output_shape=mask.shape,
                preserve_range=True, order=0).astype(np.uint8)



def apply_affine_to_velocity(velocity_array, transforms, template, threshold=1e-6):
    """
    Aligns each velocity frame using:
    1. COM shift based on binary mask
    2. Affine transform from precomputed transforms

    Parameters:
        velocity_array (np.ndarray): shape (T, H, W), raw velocity data
        transforms (list): list of AffineTransform objects for each frame
        template (np.ndarray): circular binary mask template
        threshold (float): threshold to binarize velocity

    Returns:
        aligned_velocity (np.ndarray): same shape, float32, NaNs replaced with 0
    """
    #  Ensure we use float for safe NaN handling
    velocity_array = velocity_array.astype(np.float32)
    aligned_velocity = np.zeros_like(velocity_array, dtype=np.float32)
    
    ref_com = center_of_mass(template)

    for i in range(velocity_array.shape[0]):
        #  Step 1: Binary mask from velocity
        binary_mask = (np.abs(velocity_array[i]) > threshold).astype(np.uint8)

        #  If mask is completely empty, skip frame safely
        if np.count_nonzero(binary_mask) == 0:
            print(f"[Warning] Frame {i} is empty after thresholding. Skipping.")
            continue

        #  Step 2: COM shift based on mask
        frame_com = center_of_mass(binary_mask)
        dy, dx = ref_com[0] - frame_com[0], ref_com[1] - frame_com[1]
        shifted = shift(velocity_array[i], shift=(dy, dx), order=1, mode='constant')

        # Step 3: Apply affine transform
        warped = warp(
            shifted,
            inverse_map=transforms[i].inverse,
            output_shape=velocity_array[i].shape,
            preserve_range=True,
            order=1,         # linear interpolation
            cval=np.nan      # NaNs for out-of-bounds
        )

        # Step 4: Clean NaNs → 0
        warped = np.nan_to_num(warped, nan=0.0)

        aligned_velocity[i] = warped

    return aligned_velocity


def affine_align_masks_and_collect_transforms(mask_array, template):
    aligned = np.zeros_like(mask_array)
    transforms = []
    ref_com = center_of_mass(template)
    for i in range(mask_array.shape[0]):
        shifted, _ = com_align(mask_array[i], ref_com)
        #aff = compute_affine_to_template(shifted, template)
        aff = compute_affine_to_template(shifted, template, min_scale=0.85, max_scale=1.15, force_isotropic=True, frame_idx=i)
        aligned[i] = apply_affine(shifted, aff)
        transforms.append(aff)
    return aligned, transforms


def compute_manual_affine_from_velocity(src_velocity, template_velocity):
    """
    Compute an affine transform between a shifted velocity frame and the template,
    using velocity intensities directly for statistical matching.
    
    We assume inputs are already centered (COM-aligned).
    
    Parameters:
        src_velocity (2D np.ndarray): Source velocity frame (float32)
        template_velocity (2D np.ndarray): Reference velocity frame (float32)

    Returns:
        AffineTransform: Estimated affine transform (scale + translation)
    """
    from skimage.measure import regionprops


    src_mask = (np.abs(src_velocity) > 1e-6).astype(np.uint8)
    tgt_mask = (np.abs(template_velocity) > 1e-6).astype(np.uint8)

    props_src = regionprops(src_mask, intensity_image=src_velocity)
    props_tgt = regionprops(tgt_mask, intensity_image=template_velocity)

    if len(props_src) == 0 or len(props_tgt) == 0:
        print(f"[Warning] Empty regionprops. Skipping affine.")
        return AffineTransform()

    props_src = props_src[0]
    props_tgt = props_tgt[0]

    src_c = props_src.weighted_centroid
    tgt_c = props_tgt.weighted_centroid
    src_m = props_src.weighted_moments_central
    tgt_m = props_tgt.weighted_moments_central

    # Validate moments before taking sqrt
    if src_m[2, 0] <= 0 or src_m[0, 2] <= 0 or tgt_m[2, 0] <= 0 or tgt_m[0, 2] <= 0:
        print(f"[Warning] Invalid central moments. Skipping affine.")
        return AffineTransform()

    scale_y = np.sqrt(tgt_m[2, 0]) / (np.sqrt(src_m[2, 0]) + 1e-8)
    scale_x = np.sqrt(tgt_m[0, 2]) / (np.sqrt(src_m[0, 2]) + 1e-8)

    t_y = tgt_c[0] - scale_y * src_c[0]
    t_x = tgt_c[1] - scale_x * src_c[1]

    return AffineTransform(scale=(scale_x, scale_y), translation=(t_x, t_y))

def sitk_affine_registration_velocity(fixed_velocity, moving_velocity, fixed_mask=None, moving_mask=None,
                                      min_signal_pixels=100, max_allowed_scale=3.0, fallback_metric='MeanSquares'):
    """
    Robust affine registration using velocity images with safety checks.

    Parameters:
        fixed_velocity (np.ndarray): Template velocity image (2D float32).
        moving_velocity (np.ndarray): Frame to align (2D float32).
        fixed_mask (np.ndarray or None): Optional binary mask.
        moving_mask (np.ndarray or None): Optional binary mask.
        min_signal_pixels (int): Minimum number of valid pixels to perform registration.
        max_allowed_scale (float): Maximum allowed scale in either axis.
        fallback_metric (str): Metric to use if NormalizedCorrelation is unavailable.

    Returns:
        sitk.Transform or None: Final transform or None if failed.
        np.ndarray or None: Warped image or None if skipped.
    """
    import SimpleITK as sitk
    import numpy as np

    # Check signal sufficiency
    if np.count_nonzero(np.abs(moving_velocity) > 1e-6) < min_signal_pixels:
        print("[Skip] Too few non-zero velocity pixels. Skipping registration.")
        return None, None

    fixed = sitk.GetImageFromArray(fixed_velocity.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_velocity.astype(np.float32))

    if fixed_mask is not None:
        fixed_mask = sitk.GetImageFromArray((fixed_mask > 0).astype(np.uint8))
    if moving_mask is not None:
        moving_mask = sitk.GetImageFromArray((moving_mask > 0).astype(np.uint8))

    # Initialize registration
    registration = sitk.ImageRegistrationMethod()

    # Metric setup with fallback
    try:
        registration.SetMetricAsNormalizedCorrelation()
    except Exception:
        print("[Fallback] Using SetMetricAs%s()" % fallback_metric)
        if fallback_metric == 'MeanSquares':
            registration.SetMetricAsMeanSquares()
        elif fallback_metric == 'Correlation':
            registration.SetMetricAsCorrelation()
        else:
            raise ValueError(f"Unsupported fallback metric: {fallback_metric}")

    # Apply masks if provided
    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration.SetMetricMovingMask(moving_mask)

    # Set transform type
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Optimizer
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6, relaxationFactor=0.5
    )

    try:
        final_transform = registration.Execute(fixed, moving)

          # Check scaling if transform is Affine
        if isinstance(final_transform, sitk.AffineTransform):
            matrix = np.array(final_transform.GetMatrix()).reshape(2, 2)
            u, s, vh = np.linalg.svd(matrix)
            scale_y, scale_x = s[0], s[1]
            if scale_x > max_allowed_scale or scale_y > max_allowed_scale:
                print(f"[Reject] Scale too large: x={scale_x:.2f}, y={scale_y:.2f}. Skipping frame.")
                return None, None

        # Apply transform
        resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        warped = sitk.GetArrayFromImage(resampled)
        return final_transform, warped

    except Exception as e:
        print(f"[Error] Registration failed: {str(e)}")
        return None, None


def affine_align_velocity_pipeline(velocity_array, template_velocity, method='manual'):
    """
    Align velocity frames using either 'manual' or 'sitk' intensity-based affine alignment.
    COM shift is always based on binary masks. Affine transform is based on velocity intensities.
    
    Parameters:
        velocity_array (np.ndarray): shape (T, H, W), input sequence
        template_velocity (np.ndarray): shape (H, W), template frame
        method (str): 'manual' or 'sitk'
        
    Returns:
        aligned_velocity (np.ndarray): aligned velocity frames
        aligned_masks (np.ndarray): binary masks after alignment
        transforms (list): list of transforms (or None) per frame
    """
    import numpy as np
    from scipy.ndimage import center_of_mass, shift
    from skimage.transform import warp

    T, H, W = velocity_array.shape
    aligned_velocity = np.zeros((T, H, W), dtype=np.float32)
    aligned_masks = np.zeros((T, H, W), dtype=np.uint8)
    transforms = []

    ref_mask = (np.abs(template_velocity) > 1e-6).astype(np.uint8)
    ref_com = center_of_mass(ref_mask)

    for i in range(T):
        frame = velocity_array[i]
        mask = (np.abs(frame) > 1e-6).astype(np.uint8)

        if np.count_nonzero(mask) < 20:
            print(f"[Skip] Frame {i} has insufficient signal. Skipping.")
            aligned_velocity[i] = 0.0
            aligned_masks[i] = 0
            transforms.append(None)
            continue

        # Step 1: COM alignment based on mask
        frame_com = center_of_mass(mask)
        dy, dx = ref_com[0] - frame_com[0], ref_com[1] - frame_com[1]
        shifted = shift(frame, shift=(dy, dx), order=1, mode='constant')

        # Step 2: Affine registration using intensity
        if method == 'manual':
            try:
                aff = compute_manual_affine_from_velocity(shifted, template_velocity)
                warped = warp(shifted, inverse_map=aff.inverse, output_shape=(H, W),
                              order=1, preserve_range=True, cval=0.0)
            except Exception as e:
                print(f"[Warning] Frame {i}: Manual affine failed: {e}")
                warped = None
                aff = None
        elif method == 'sitk':
            aff, warped = sitk_affine_registration_velocity(template_velocity, shifted)
        else:
            raise ValueError("Invalid method. Use 'manual' or 'sitk'.")

        if warped is None:
            print(f"[Skip] Frame {i} alignment failed. Skipping.")
            aligned_velocity[i] = 0.0
            aligned_masks[i] = 0
            transforms.append(None)
            continue

        aligned_velocity[i] = warped
        aligned_masks[i] = (np.abs(warped) > 1e-6).astype(np.uint8)
        transforms.append(aff)

    return aligned_velocity, aligned_masks, transforms



def extract_time_profile_and_variance(velocity_array, mask_array, ref_idx=15, patch_size=5):
    cy, cx = center_of_mass(mask_array[ref_idx])
    cy, cx = int(round(cy)), int(round(cx))
    r = patch_size // 2
    patch_mask = np.zeros_like(mask_array[0], dtype=bool)
    patch_mask[cy - r:cy + r + 1, cx - r:cx + r + 1] = True
    time_profile = velocity_array[:, patch_mask].mean(axis=1)
    variance_map = np.var(velocity_array, axis=0)
    return time_profile, variance_map, patch_mask


def extract_spatiotemporal_profile(velocity_array, axis='horizontal', line_index=None):
    """
    Extracts a 2D spatiotemporal slice from a 3D velocity array.

    Parameters:
        velocity_array (np.ndarray): 3D array of shape (T, H, W)
        axis (str): one of 'horizontal', 'vertical', 'full_x', 'full_y'
        line_index (int or None): specific row/column index (optional)

    Returns:
        np.ndarray: 2D spatiotemporal profile (frame × space)
    """
    _, H, W = velocity_array.shape

    if axis == 'horizontal':
        y = H // 2 if line_index is None else line_index
        return velocity_array[:, y, :]
    elif axis == 'vertical':
        x = W // 2 if line_index is None else line_index
        return velocity_array[:, :, x]
    elif axis == 'full_x':
        return np.mean(velocity_array, axis=1)  # Average over rows → frame × width
    elif axis == 'full_y':
        return np.mean(velocity_array, axis=2)  # Average over columns → frame × height
    else:
        raise ValueError("Invalid axis. Must be one of: 'horizontal', 'vertical', 'full_x', 'full_y'")


# Function to create a deformable alignment GIF
def create_alignment_gif(mask_array_before, mask_array_after, save_path='alignment.gif', template_mask=None):
    import matplotlib.pyplot as plt
    from scipy.ndimage import center_of_mass
    import numpy as np
    import imageio
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    frames = []
    num_frames = mask_array_before.shape[0]
    crop_size = 80  # Zoom on mask

    # Compute template radius once if provided
    if template_mask is not None:
        template_area = np.sum(template_mask > 0)
        template_radius = np.sqrt(template_area / np.pi)
    else:
        template_radius = None

    for i in range(num_frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        com = center_of_mass(mask_array_before[i])
        cy, cx = int(round(com[0])), int(round(com[1]))
        r = crop_size // 2
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy + r, cy - r)  # Inverse y-axis for correct orientation

        # Show mask before alignment
        ax.imshow(mask_array_before[i], cmap='Blues', alpha=0.8)

        # Show aligned mask contour
        ax.contour(mask_array_after[i], levels=[0.5], colors='orange', linewidths=2)

        # Compute COM and radius info
        com_before = center_of_mass(mask_array_before[i])
        com_after = center_of_mass(mask_array_after[i])
        dy, dx = com_after[0] - com_before[0], com_after[1] - com_before[1]

        area_before = np.sum(mask_array_before[i])
        area_after = np.sum(mask_array_after[i])
        delta_area = abs(int(area_after) - int(area_before))

        radius_before = np.sqrt(area_before / np.pi)
        radius_after = np.sqrt(area_after / np.pi)

        # Arrows and annotations
        ax.arrow(com_before[1], com_before[0], dx, dy, color='cyan', head_width=2, head_length=2, length_includes_head=True)

        # Title with radius info
        title = f"Frame {i} | dy={dy:+.1f}, dx={dx:+.1f}, ΔA={delta_area:+}\n"
        title += f"r_before={radius_before:.1f}, r_after={radius_after:.1f}"
        if template_radius is not None:
            title += f", r_template={template_radius:.1f}"
        ax.set_title(title, fontsize=10)

        ax.axis('off')

        # Legend
        blue_patch = mpatches.Patch(color='blue', label='Before Align')
        orange_line = mpatches.Patch(color='orange', label='After Align (Contour)')
        cyan_line = mpatches.Patch(color='cyan', label='COM Shift')
        ax.legend(handles=[blue_patch, orange_line, cyan_line], loc='lower right', fontsize=8, framealpha=0.9)

        canvas = FigureCanvas(fig)
        canvas.draw()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=0.3)

# -------------------- Visualization & Saving --------------------
def save_visual_outputs_affine(result, output_dir, template_mask=None):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(result['filename'])[0]

    def safe_imshow_with_colorbar(data, cmap, title, fname, label):
        plt.figure(figsize=(6, 6))
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        if vmin != vmax:
            plt.colorbar(im, label=label)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    # Time-Velocity Profile
    plt.figure(figsize=(8, 4))
    plt.plot(result['time_profile'])
    plt.title(f"Time-Velocity Profile: {base}")
    plt.xlabel("Frame")
    plt.ylabel("Mean Velocity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_time_profile.png"))
    plt.close()

    # Variance Map
    safe_imshow_with_colorbar(result['variance_map'], 'hot', f"Turbulence Map: {base}",
        os.path.join(output_dir, f"{base}_variance_map.png"), "Temporal Variance")

    # Patch Overlay on Reference Frame
    vmin, vmax = np.nanmin(result['ref_frame_img']), np.nanmax(result['ref_frame_img'])
    plt.figure(figsize=(6, 6))
    im = plt.imshow(result['ref_frame_img'], cmap='bwr', vmin=vmin, vmax=vmax)
    plt.contour(result['patch_mask'], colors='yellow')
    if vmin != vmax:
        plt.colorbar(im, label="Velocity")
    plt.title(f"Patch Overlay on Velocity (Frame {result['ref_idx']})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_velocity_patch.png"))
    plt.close()

 # --- Zoomed Crops from Variance Map ---
    crop_size = 40
    cy_var, cx_var = center_of_mass(result['variance_map'])
    # Handle NaN COM gracefully
    if np.isnan(cy_var) or np.isnan(cx_var):
        print(f"[Skip] Variance map has NaN center of mass. Skipping crop visualizations.")
        return
    cy_var, cx_var = int(round(cy_var)), int(round(cx_var))
    r = crop_size // 2
    cropped = result['variance_map'][cy_var - r:cy_var + r, cx_var - r:cx_var + r]

    # Nearest Interpolation (pixelated)
    vmin = np.nanmin(result['variance_map'])
    vmax = np.nanmax(result['variance_map'])
    plt.figure(figsize=(5, 5))
    if vmin != vmax:
        im = plt.imshow(result['variance_map'], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Temporal Variance")
    else:
        im = plt.imshow(result['variance_map'], cmap='jet')
    plt.imshow(cropped, cmap='jet', interpolation='nearest')
    plt.title("Zoomed Variance (Interpolation: Nearest)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_variance_zoom_nearest.png"))
    plt.close()

    # Bilinear Interpolation (smooth)
    
    vmin = np.nanmin(result['variance_map'])
    vmax = np.nanmax(result['variance_map'])
    plt.figure(figsize=(5, 5))
    if vmin != vmax:
        im = plt.imshow(result['variance_map'], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Temporal Variance")
    else:
        im = plt.imshow(result['variance_map'], cmap='jet')
    plt.imshow(cropped, cmap='jet', interpolation='bilinear')
    plt.title("Zoomed Variance (Interpolation: Bilinear)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_variance_zoom_bilinear.png"))
    plt.close()



    # Spatiotemporal Profile
    plt.figure(figsize=(10, 5))
    sns.heatmap(result['spatiotemporal_profile'], cmap='coolwarm', cbar_kws={'label': 'Velocity'})
    plt.title(f"Spatiotemporal Profile: {base}")
    plt.xlabel("Spatial Position")
    plt.ylabel("Frame")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_spatiotemporal.png"))
    plt.close()

    # Reference Frame Only
    safe_imshow_with_colorbar(result['ref_frame_img'], 'bwr', f"Reference Frame: {base}",
        os.path.join(output_dir, f"{base}_reference_frame.png"), "Velocity")

    # Create deformable alignment GIF (optional visualization)
    gif_path = os.path.join(output_dir, f"{base}_deformable_alignment.gif")
    create_alignment_gif(
        mask_array_before=result['original_mask'],
        mask_array_after=result['aligned_mask'],
        save_path=gif_path,
        template_mask=template_mask  
    )

    # Save aligned velocity array
    np.save(os.path.join(output_dir, f"{base}_aligned_velocity.npy"), result['aligned_velocity'])


def visualize_affine_alignment(mask_array_before, mask_array_after, ref_idx, output_path=None, base_name="alignment", num_show=14, template_mask=None):
    import matplotlib.pyplot as plt
    import os
    from scipy.ndimage import center_of_mass
    import numpy as np

    indices = list(range(max(0, ref_idx - num_show), min(mask_array_before.shape[0], ref_idx + num_show + 1)))
    ref_com = center_of_mass(mask_array_before[ref_idx])
    fig, axs = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    fig.suptitle(f"Deformable Alignment (Ref Frame: {ref_idx})", fontsize=14)

    # Compute template radius if provided
    if template_mask is not None:
        template_area = np.sum(template_mask)
        template_radius = np.sqrt(template_area / np.pi)
    else:
        template_radius = None

    for i, idx in enumerate(indices):
        axs[0, i].imshow(mask_array_before[idx], cmap='gray')
        axs[0, i].set_title(f"Before Frame {idx}")
        axs[0, i].axis('off')

        axs[1, i].imshow(mask_array_after[idx], cmap='gray')
        axs[1, i].set_title(f"After Frame {idx}")
        axs[1, i].axis('off')

        com_after = center_of_mass(mask_array_after[idx])
        dy, dx = com_after[0] - ref_com[0], com_after[1] - ref_com[1]
        delta_area = np.sum(mask_array_after[idx].astype(np.int32)) - np.sum(mask_array_before[idx].astype(np.int32))

        # Compute radii
        area_before = np.sum(mask_array_before[idx])
        area_after = np.sum(mask_array_after[idx])
        r_before = np.sqrt(area_before / np.pi)
        r_after = np.sqrt(area_after / np.pi)

        radius_text = f"r₀={r_before:.1f}, r₁={r_after:.1f}"
        if template_radius is not None:
            radius_text += f", r∘={template_radius:.1f}"

        axs[1, i].text(2, 10, f"dy={dy:+.1f}\ndx={dx:+.1f}\nΔA={delta_area:+.0f}\n{radius_text}",
                      color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, f"{base_name}_deformable_alignment.png"), dpi=150)
        plt.close()
    else:
        plt.show()


def extract_binary_masks(arr, threshold=1e-6):
    return (np.abs(arr) > threshold).astype(np.uint8)

# ... (unchanged code above) ...

def visualize_velocity_alignment(velocity_array_before, velocity_array_after, ref_idx, output_path=None, base_name="velocity", num_show=14, crop_center=(96, 96), crop_radius=20, threshold=1e-6,  template_mask=None):
    """
    Visualizes velocity frames before and after alignment side-by-side with COM alignment and cropping.
    Now also overlays the equivalent radius of each binary mask on the frames.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import center_of_mass
    from scipy.ndimage import shift
    import os
    # Compute color scale based on both arrays (symmetric)

    abs_max = max(
        np.max(np.abs(velocity_array_before)),
        np.max(np.abs(velocity_array_after))
    )
    vmin, vmax = -abs_max, abs_max

    # Debug: show raw min/max
    print(f"[Before Alignment]  min: {np.min(velocity_array_before):.2f}, max: {np.max(velocity_array_before):.2f}")
    print(f"[After Alignment]   min: {np.min(velocity_array_after):.2f}, max: {np.max(velocity_array_after):.2f}")
    print(f"[Symmetric Colormap] vmin: {vmin:.2f}, vmax: {vmax:.2f}")
    #  Compute template radius if available
    template_radius = None
    if template_mask is not None:
        template_area = np.sum(template_mask > 0)
        template_radius = np.sqrt(template_area / np.pi)

    indices = list(range(max(0, ref_idx - num_show), min(velocity_array_before.shape[0], ref_idx + num_show + 1)))
    n_cols = len(indices)
    fig, axs = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), gridspec_kw={'hspace': 0.2})

    title = f"Velocity Alignment (Ref Frame: {ref_idx})"
    if template_radius is not None:
        title += f" | Template r≈{template_radius:.1f}"
    fig.suptitle(title, fontsize=16)
    
    cmap = 'seismic'

    for i, idx in enumerate(indices):
        # --- BEFORE ---
        before = velocity_array_before[idx]
        before_binary = (np.abs(before) > threshold).astype(np.uint8)
        com = center_of_mass(before_binary)
        dy, dx = crop_center[0] - com[0], crop_center[1] - com[1]
        before_shifted = shift(before, shift=(dy, dx), order=1, mode='constant')
        area_before = np.sum(before_binary)
        radius_before = np.sqrt(area_before / np.pi)

        r = crop_radius
        y, x = crop_center
        cropped_before = before_shifted[y - r:y + r + 1, x - r:x + r + 1]
        im1 = axs[0, i].imshow(cropped_before, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f"Before Frame {idx}\nr≈{radius_before:.1f}", fontsize=10)
        axs[0, i].axis('off')

        # --- AFTER ---
        after = velocity_array_after[idx]
        after_binary = (np.abs(after) > threshold).astype(np.uint8)
        area_after = np.sum(after_binary)
        radius_after = np.sqrt(area_after / np.pi)
        cropped_after = after[y - r:y + r + 1, x - r:x + r + 1]
        im2 = axs[1, i].imshow(cropped_after, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f"After Frame {idx}\nr≈{radius_after:.1f}", fontsize=10)
        axs[1, i].axis('off')

        # Highlight reference frame
        if idx == ref_idx:
            for ax in [axs[0, i], axs[1, i]]:
                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_edgecolor('orange')

    # Colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.35])
    fig.colorbar(im1, cax=cbar_ax1, label='Velocity')
    cbar_ax2 = fig.add_axes([0.92, 0.12, 0.01, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label='Velocity')

    if output_path:
        fname = os.path.join(output_path, f"{base_name}_velocity_alignment.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


import numpy as np
from scipy.ndimage import center_of_mass

def combined_velocity_diagnostics(velocity_array, template_mask, threshold=1e-6, frames=None):
    """
    Combined diagnostic for velocity alignment:
    - min/max values
    - percent non-zero
    - center of mass distance to template
    - energy inside/outside template mask

    Parameters:
        velocity_array (np.ndarray): shape (T, H, W), velocity values
        template_mask (np.ndarray): shape (H, W), binary mask of circular template
        threshold (float): threshold for binarization of velocity
        frames (list or None): list of frame indices to analyze, or all if None
    """
    T, H, W = velocity_array.shape
    if frames is None:
        frames = list(range(T))
    
    template_com = center_of_mass(template_mask)
    template_mask_bool = template_mask.astype(bool)

    print("="*70)
    print(f"{'Frame':>5} | {'Min':>7} {'Max':>7} | {'%NonZero':>9} | {'COM Dist':>9} | {'In-Mask':>8} | {'Out-Mask':>9}")
    print("-"*70)

    for idx in frames:
        frame = velocity_array[idx]
        min_val, max_val = np.min(frame), np.max(frame)
        nonzero_ratio = np.count_nonzero(np.abs(frame) > threshold) / frame.size * 100

        # Binarize and compute COM
        binary_mask = (np.abs(frame) > threshold).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            com_dist = np.nan
        else:
            com = center_of_mass(binary_mask)
            com_dist = np.sqrt((com[0] - template_com[0])**2 + (com[1] - template_com[1])**2)

        # Energy inside/outside template mask
        in_mask_energy = np.sum(np.abs(frame)[template_mask_bool])
        out_mask_energy = np.sum(np.abs(frame)[~template_mask_bool])

        print(f"{idx:5d} | {min_val:7.2f} {max_val:7.2f} | {nonzero_ratio:9.2f}% | {com_dist:9.2f} | {in_mask_energy:8.1f} | {out_mask_energy:9.1f}")
    
    # print("="*70)
    # # --- Debug Frame 11 ----
    # print("="*40)
    # print("[Debug] Checking raw values of Frame 11 (before alignment):")
    # print("Unique values in velocity_array[11]:", np.unique(velocity_array[11]))
    # print("Non-zero count:", np.count_nonzero(velocity_array[11]))
    # print("Frame 11 raw values:\n", velocity_array[11])
    # print("="*40)

import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

def analyze_variance_map(variance_map, template_mask=None, threshold=100, name=''):
    """
    Analyze a temporal variance map in detail.

    Parameters:
        variance_map (np.ndarray): 2D array of temporal variance.
        template_mask (np.ndarray or None): 2D binary mask of vessel region.
        threshold (float): threshold for high variance pixels.
        name (str): optional name to include in print statements.

    Returns:
        A dictionary of computed statistics.
    """
    print("="*80)
    print(f"[{name}] Temporal Variance Map Analysis")
    print("="*80)

    stats = {}

    vmax = np.max(variance_map)
    vmean = np.mean(variance_map)
    vstd = np.std(variance_map)
    print(f"Max Variance:      {vmax:.2f}")
    print(f"Mean Variance:     {vmean:.2f}")
    print(f"Std Dev Variance:  {vstd:.2f}")
    stats.update({'max': vmax, 'mean': vmean, 'std': vstd})

    # Location of peak variance
    peak_y, peak_x = np.unravel_index(np.argmax(variance_map), variance_map.shape)
    print(f"Peak Location:     (y={peak_y}, x={peak_x})")
    stats['peak_location'] = (peak_y, peak_x)

    # Number of high variance pixels
    high_var_count = np.sum(variance_map > threshold)
    print(f"#Pixels > {threshold}: {high_var_count}")
    stats['num_high_pixels'] = high_var_count

    if template_mask is not None:
        in_mask = variance_map[template_mask > 0]
        out_mask = variance_map[template_mask == 0]

        in_mean = np.mean(in_mask)
        out_mean = np.mean(out_mask)
        snr = in_mean / (out_mean + 1e-6)
        energy_ratio = np.sum(in_mask) / (np.sum(out_mask) + 1e-6)

        print(f"Mean Variance (In Mask):   {in_mean:.2f}")
        print(f"Mean Variance (Out Mask):  {out_mean:.2f}")
        print(f"SNR (In/Out):              {snr:.2f}")
        print(f"Energy Ratio (In/Out):     {energy_ratio:.2f}")

        stats.update({
            'in_mask_mean': in_mean,
            'out_mask_mean': out_mean,
            'snr': snr,
            'energy_ratio': energy_ratio
        })

        # Distance of peak to COM
        com = center_of_mass(template_mask)
        dist_to_com = np.sqrt((peak_y - com[0])**2 + (peak_x - com[1])**2)
        print(f"Distance Peak to COM:      {dist_to_com:.2f}")
        stats['dist_to_com'] = dist_to_com
    else:
        print("No mask provided — skipping in/out comparisons.")

    return stats

def compute_velocity_alignment_metrics(aligned_velocity, original_velocity, template_mask, threshold=1e-6):
    """
    Compute velocity-based metrics for evaluating alignment quality.

    Returns:
        dict with:
        - energy_preservation: Ratio of aligned vs. original velocity magnitude
        - in_out_energy_ratio: Signal concentration within vs. outside the mask
        - mean_temporal_correlation: Correlation of adjacent frames
        - mean_ssim_with_mean_frame: Structural similarity to the mean frame
    """
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import pearsonr
    import numpy as np

    T, H, W = aligned_velocity.shape
    template_mask = template_mask.astype(bool)

    energy_orig = np.sum(np.abs(original_velocity))
    energy_aligned = np.sum(np.abs(aligned_velocity))
    energy_preservation = energy_aligned / (energy_orig + 1e-6)

    in_mask_energy = np.sum(np.abs(aligned_velocity[:, template_mask]))
    out_mask_energy = np.sum(np.abs(aligned_velocity[:, ~template_mask]))
    in_out_ratio = in_mask_energy / (out_mask_energy + 1e-6)

    # Frame-to-frame temporal correlation
    temporal_corrs = []
    for i in range(T - 1):
        flat1 = aligned_velocity[i].flatten()
        flat2 = aligned_velocity[i + 1].flatten()
        corr, _ = pearsonr(flat1, flat2)
        temporal_corrs.append(corr)
    mean_temporal_corr = np.mean(temporal_corrs)

    # SSIM to mean frame
    mean_frame = np.mean(aligned_velocity, axis=0)
    ssim_scores = [
        ssim(aligned_velocity[i], mean_frame, data_range=mean_frame.max() - mean_frame.min())
        for i in range(T)
    ]
    mean_ssim = np.mean(ssim_scores)

    return {
        'energy_preservation': energy_preservation,
        'in_out_energy_ratio': in_out_ratio,
        'mean_temporal_correlation': mean_temporal_corr,
        'mean_ssim_with_mean_frame': mean_ssim
    }

def process_affine_folder(folder_path, output_path, template_path=None, axis='horizontal', use_internal_template=False, use_velocity_for_affine=False, method='manual'):
    """
    Process velocity frames with COM + affine alignment.

    Args:
        folder_path (str): Directory with velocity .npy files.
        output_path (str): Where to save results.
        template_path (str): Path to external template (optional if using internal frame).
        axis (str): Axis for spatiotemporal profile.
        use_internal_template (bool): If True, select ref frame from sample instead of external template.
        use_velocity_for_affine (bool): If True, use velocity intensities to compute affine transform (but COM still from mask).
        method (str): Affine alignment method for intensity-based mode ('manual' or 'sitk').
    """
    os.makedirs(output_path, exist_ok=True)
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])[:1000]
    summary_rows = []
    for i, fname in enumerate(tqdm(filenames)):
        if i % 50 == 0:
            print(f"\n[INFO] Processing sample {i+1}/{len(filenames)}: {fname}")
        path = os.path.join(folder_path, fname)
        velocity_array = np.load(path).astype(np.float32)
        
        # Step 1: Create binary mask from velocity values
        binary_mask = (np.abs(velocity_array) > 1e-6).astype(np.uint8)
        
        # Step 2: Choose template
        if use_internal_template:
            # Use similarity to select best internal frame as template
            ref_idx = choose_reference_frame(binary_mask, method='similarity')
            template = velocity_array[ref_idx] if use_velocity_for_affine else binary_mask[ref_idx]
            print(f"[INFO] Using internal frame {ref_idx} as template.")
        else:
            # Load external template
            template = np.load(template_path)
            #print("#############For Debug####################")
            template_area = np.sum(template > 0)
            template_radius = np.sqrt(template_area / np.pi)
            #print(f"[DEBUG] Loaded template from {template_path} — radius ≈ {template_radius:.2f} pixels")

            ref_idx = None  # External template is used; no internal reference index
        
        # Step 3: Alignment
        if use_velocity_for_affine:
            aligned_velocity, aligned_masks, transforms = affine_align_velocity_pipeline(
                velocity_array, template, method=method)
        else:
            aligned_masks, transforms = affine_align_masks_and_collect_transforms(
                binary_mask, template)
            aligned_velocity = apply_affine_to_velocity(
                velocity_array, transforms, template)

        if aligned_velocity is None or np.all(aligned_velocity == 0):
            print(f"[Error] Aligned velocity is invalid for {fname}. Skipping.")
            continue

        # Determine reference frame index and template for analysis
        ref_frame_used = 15 if ref_idx is None else ref_idx
        template_for_analysis = template if ref_idx is None else aligned_masks[ref_idx]

        # Step 4:
        # Determine the reference template for diagnostics
        if use_internal_template:
            template_for_analysis = aligned_masks[ref_idx]
        else:
            template_for_analysis = template
        
        # visualize_velocity_alignment(
        #     velocity_array_before=velocity_array,
        #     velocity_array_after=aligned_velocity,
        #     ref_idx=ref_frame_used,
        #     output_path=output_path,
        #     base_name=os.path.splitext(fname)[0],
        #     template_mask=template_for_analysis
        # )

        # Step 5: Feature extraction
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(
            aligned_velocity, aligned_masks, ref_idx=ref_frame_used)
        spatiotemporal_profile = extract_spatiotemporal_profile(
            aligned_velocity, axis=axis)
        # --- Variance Map Diagnostics ---
        print("\n[Variance Map Analysis]")
        analyze_variance_map(
            variance_map=variance_map,
            template_mask=template_for_analysis,
            threshold=100,
            name=fname
        )
        # Step 6: Pack result
        result = {
            'filename': fname,
            'ref_idx': ref_frame_used,
            'aligned_velocity': aligned_velocity,
            'time_profile': time_profile,
            'variance_map': variance_map,
            'patch_mask': patch_mask,
            'ref_frame_img': template_for_analysis,
            'spatiotemporal_profile': spatiotemporal_profile,
            'binary_mask': binary_mask,
            'original_mask': binary_mask,
            'aligned_mask': aligned_masks
        }
        
        # Save aligned velocity array even if skipping visualizations
        base = os.path.splitext(fname)[0]
        np.save(os.path.join(output_path, f"{base}_circle10.npy"), result['aligned_velocity'])


        # Step 7: Compute additional velocity alignment metrics
        additional_metrics = compute_velocity_alignment_metrics(
            aligned_velocity=aligned_velocity,
            original_velocity=velocity_array,
            template_mask=template_for_analysis
        )

        # Log additional metrics to console
        # print("[Additional Metrics]")
        # for k, v in additional_metrics.items():
        #     print(f"{k}: {v:.4f}")


        # --- Save Visual Outputs ---
        ###save_visual_outputs_affine(result, output_path, template_mask=template_for_analysis)



        # --- Combined Velocity Diagnostics ---
        combined_velocity_diagnostics(result['aligned_velocity'], template_for_analysis, threshold=1e-6)

        # --- Visualize Mask Alignment ---
        # visualize_affine_alignment(
        #     mask_array_before=result['original_mask'],
        #     mask_array_after=result['aligned_mask'],
        #    ref_idx=ref_frame_used,
        #     output_path=output_path,
        #     base_name=os.path.splitext(fname)[0],
        #     template_mask=template_for_analysis
        # )

        # --- Radius Summary ---
        # print("=" * 80)
        # print(f"[{fname}] Frame-wise Radius Summary (From Mask and Velocity)")
        # print(f"{'Frame':>5} | {'r_mask_before':>13} | {'r_mask_after':>13} | {'r_vel_before':>13} | {'r_vel_after':>13}")
        # print("-" * 80)

        # for i in range(velocity_array.shape[0]):
        #     vel_before_bin = (np.abs(velocity_array[i]) > 1e-6).astype(np.uint8)
        #     vel_after_bin = (np.abs(aligned_velocity[i]) > 1e-6).astype(np.uint8)

        #     r_mask_before = np.sqrt(np.sum(binary_mask[i]) / np.pi)
        #     r_mask_after = np.sqrt(np.sum(aligned_masks[i]) / np.pi)
        #     r_vel_before = np.sqrt(np.sum(vel_before_bin) / np.pi)
        #     r_vel_after = np.sqrt(np.sum(vel_after_bin) / np.pi)

        #     print(f"{i:5d} | {r_mask_before:13.2f} | {r_mask_after:13.2f} | {r_vel_before:13.2f} | {r_vel_after:13.2f}")
        # print("=" * 80)

        import pandas as pd
 
        summary_rows.append({
            'filename': fname,
            'method': 'Affine_mask',  # Change this per script: 'Affine', 'BSpline_mask', etc.
            'ref_idx': ref_idx,
            'max_variance': result['variance_map'].max(),
            'mean_variance': result['variance_map'].mean(),
            'snr': additional_metrics['in_out_energy_ratio'],
            'energy_preservation': additional_metrics['energy_preservation'],
            'mean_temporal_correlation': additional_metrics['mean_temporal_correlation'],
            'mean_ssim': additional_metrics['mean_ssim_with_mean_frame'],
        })

        # After the loop (outside the for-loop):
        df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(output_path, 'summary_affine_mask.csv')
        df.to_csv(summary_csv_path, index=False)
        print(f"\n[Saved Summary CSV] {summary_csv_path}")








# -------------------- Run Entry Point --------------------
if __name__ == "__main__":
    # Define input/output directories
    input_folder = r"Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\output_bspline_intra_raw_velocity_1007337"
        #Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/data
    #input_folder = r"\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\results2"
    output_folder = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/output/inter_subject_affine_10"

    # Define circular template (shared for all)
    template_path = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/templates/circular_template_r10.npy"

    # Choose axis for spatiotemporal profile ('horizontal', 'vertical', 'full_x', 'full_y')
    axis = 'horizontal'

    # Run the affine alignment process
    process_affine_folder(
        folder_path=input_folder,
        output_path=output_folder,
        template_path=template_path,
        axis=axis,
        use_internal_template=False, #or False True
        use_velocity_for_affine=False, #or False True
        method='sitk'             #or manual sitk
    )
