"""
This script performs intra-subject alignment of aortic velocity MRI frames.

Initially, we attempted to directly register the raw velocity frames using deformable (BSpline) transforms.
However, this approach resulted in unstable and artifact-prone results due to noise and interpolation errors outside the vessel region.

To improve robustness, we switched to a mask-guided approach:
- Binary masks were extracted from each velocity frame.
- Deformable registration was computed using these masks.
- The resulting transforms were then applied to the original velocity frames.

This strategy ensures that the transformation is driven only by the relevant vessel region,
reducing the influence of noisy background and improving alignment consistency across frames.
"""
import os
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import center_of_mass, shift
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

def extract_binary_masks(velocity_array, threshold=1e-6):
    return (np.abs(velocity_array) > threshold).astype(np.uint8)


def choose_reference_frame(mask_array, method='similarity', external_template=None):
    num_frames = mask_array.shape[0]
    areas = mask_array.sum(axis=(1, 2))
    if method == 'area':
        mean_area = np.mean(areas)
        return np.argmin(np.abs(areas - mean_area))
    elif method == 'similarity':
        total_scores = [
            np.mean([ssim(mask_array[i], mask_array[j]) for j in range(num_frames) if i != j])
            for i in range(num_frames)
        ]
        return np.argmax(total_scores)
    elif method == 'template':
        if external_template is None:
            raise ValueError("Template image required for template-based reference selection")
        scores = [ssim(mask_array[i], external_template) for i in range(num_frames)]
        return np.argmax(scores)
    else:
        raise ValueError("Invalid reference selection method")

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import center_of_mass, shift

def get_bspline_transform(fixed_mask, moving_mask, mesh_size=(8, 8)):
    """
    Compute BSpline transform from moving_mask to fixed_mask.
    """
    fixed_itk = sitk.GetImageFromArray(fixed_mask.astype(np.float32))
    moving_itk = sitk.GetImageFromArray(moving_mask.astype(np.float32))

    transform = sitk.BSplineTransformInitializer(fixed_itk, mesh_size)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.5)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e+7
    )
    registration_method.SetInitialTransform(transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_itk, moving_itk)
    return final_transform

def apply_transform_to_velocity(velocity, reference_shape, transform):
    """
    Warp a single velocity frame using a given transform.
    Ensures reference_shape is a valid 2D tuple.
    """
    if not (isinstance(reference_shape, tuple) and len(reference_shape) == 2):
        raise ValueError(f"Invalid reference_shape: {reference_shape}. Must be a (H, W) tuple.")

    moving_itk = sitk.GetImageFromArray(velocity.astype(np.float32))
    reference_itk = sitk.GetImageFromArray(np.zeros(reference_shape, dtype=np.float32))
    warped_itk = sitk.Resample(
        moving_itk, reference_itk, transform,
        sitk.sitkLinear, 0.0, moving_itk.GetPixelID()
    )
    return sitk.GetArrayFromImage(warped_itk)


def hybrid_bspline_mask_guided_alignment(velocity_array, threshold=1e-6, ref_idx=None):
    """
    Hybrid alignment: use binary mask to compute BSpline transform, then apply to velocity.
    """
    T, H, W = velocity_array.shape
    aligned_velocity = np.zeros_like(velocity_array, dtype=np.float32)

    # Step 1: Extract masks
    binary_masks = (np.abs(velocity_array) > threshold).astype(np.uint8)

    # Step 2: Choose reference frame
    if ref_idx is None:
        ref_idx = 15
    ref_mask = binary_masks[ref_idx]
    ref_com = center_of_mass(ref_mask)

    for i in range(T):
        if i == ref_idx:
            aligned_velocity[i] = velocity_array[i]
            continue

        mask = binary_masks[i]
        if np.count_nonzero(mask) == 0:
            print(f"[Warning] Frame {i} has empty mask. Skipping.")
            continue

        # Optional COM pre-alignment
        com = center_of_mass(mask)
        dy, dx = ref_com[0] - com[0], ref_com[1] - com[1]
        shifted_velocity = shift(velocity_array[i], shift=(dy, dx), order=1, mode='constant')
        shifted_mask = shift(mask, shift=(dy, dx), order=0, mode='constant')

        # BSpline transform from mask to reference
        transform = get_bspline_transform(ref_mask, shifted_mask)

        # Apply transform to velocity
        warped = apply_transform_to_velocity(shifted_velocity, ref_mask.shape, transform)
        aligned_velocity[i] = warped

    return aligned_velocity, ref_idx


# -------------------- Feature Extraction --------------------
def extract_time_profile_and_variance(velocity_array, mask_array, ref_frame_used, patch_size=5):
    cy, cx = center_of_mass(mask_array[ref_frame_used])
    cy, cx = int(round(cy)), int(round(cx))
    r = patch_size // 2
    patch_mask = np.zeros_like(velocity_array[0], dtype=bool)
    patch_mask[cy - r:cy + r + 1, cx - r:cx + r + 1] = True
    time_profile = velocity_array[:, patch_mask].mean(axis=1)
    variance_map = np.var(velocity_array, axis=0)
    return time_profile, variance_map, patch_mask

def extract_spatiotemporal_profile(velocity_array, axis='horizontal', line_index=None):
    _, H, W = velocity_array.shape
    if axis == 'horizontal':
        y = H // 2 if line_index is None else line_index
        return velocity_array[:, y, :]
    elif axis == 'vertical':
        x = W // 2 if line_index is None else line_index
        return velocity_array[:, :, x]
    elif axis == 'full_x':
        return np.mean(velocity_array, axis=1)
    elif axis == 'full_y':
        return np.mean(velocity_array, axis=2)
    else:
        raise ValueError("Invalid axis")


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

def visualize_deformable_alignment(mask_array_before, mask_array_after, ref_frame_used, output_path=None, base_name="alignment", num_show=14, template_mask=None):
    import matplotlib.pyplot as plt
    import os
    from scipy.ndimage import center_of_mass
    import numpy as np
    import matplotlib.patches as mpatches

    indices = list(range(max(0, ref_frame_used - num_show), min(mask_array_before.shape[0], ref_frame_used + num_show + 1)))
    ref_com = center_of_mass(mask_array_before[ref_frame_used])
    fig, axs = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    fig.suptitle(f"Deformable Alignment (Ref Frame: {ref_frame_used})", fontsize=14)

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
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"{base_name}_deformable_alignment.png"), dpi=150)
        plt.close()
    else:
        plt.show()


# ... (unchanged code above) ...

def visualize_velocity_alignment(velocity_array_before, velocity_array_after, ref_frame_used, output_path=None, base_name="velocity", num_show=14, crop_center=(96, 96), crop_radius=20, threshold=1e-6,  template_mask=None):
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

    indices = list(range(max(0, ref_frame_used - num_show), min(velocity_array_before.shape[0], ref_frame_used + num_show + 1)))
    n_cols = len(indices)
    fig, axs = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), gridspec_kw={'hspace': 0.2})

    title = f"Velocity Alignment (Ref Frame: {ref_frame_used})"
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
        if idx == ref_frame_used:
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
    
# -------------------- Visualization & Saving --------------------
def save_visual_outputs_intra_bspline(result, output_dir, template_mask=None):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(result['filename'])[0]

    # Clean variance map before visualization (optional peak suppression)
    cleaned_variance_map = result['variance_map']

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
    safe_imshow_with_colorbar( cleaned_variance_map, 'hot', f"Turbulence Map: {base}",
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
    cy_var, cx_var = center_of_mass(cleaned_variance_map)
    cy_var, cx_var = int(round(cy_var)), int(round(cx_var))
    r = crop_size // 2
    cropped = cleaned_variance_map[cy_var - r:cy_var + r, cx_var - r:cx_var + r]

    # Nearest Interpolation (pixelated)
    vmin = np.nanmin(cleaned_variance_map)
    vmax = np.nanmax(cleaned_variance_map)
    plt.figure(figsize=(5, 5))
    if vmin != vmax:
        im = plt.imshow(cleaned_variance_map, cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Temporal Variance")
    else:
        im = plt.imshow(cleaned_variance_map, cmap='jet')
    plt.imshow(cropped, cmap='jet', interpolation='nearest')
    plt.title("Zoomed Variance (Interpolation: Nearest)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_variance_zoom_nearest.png"))
    plt.close()

    # Bilinear Interpolation (smooth)
    
    vmin = np.nanmin(cleaned_variance_map)
    vmax = np.nanmax(cleaned_variance_map)
    plt.figure(figsize=(5, 5))
    if vmin != vmax:
        im = plt.imshow(cleaned_variance_map, cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Temporal Variance")
    else:
        im = plt.imshow(cleaned_variance_map, cmap='jet')
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
    np.save(os.path.join(output_dir, f"{base}_circle10.npy"), result['aligned_velocity'])


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

def analyze_top_variance_peaks(variance_map, template_mask, num_peaks=5):
    """
    Analyze top N variance peaks and determine whether they are likely noise.
    
    Parameters:
        variance_map (np.ndarray): 2D array of temporal variance.
        template_mask (np.ndarray): 2D binary mask of the region of interest.
        num_peaks (int): Number of top peaks to analyze.
    """
    from scipy.ndimage import center_of_mass
    import numpy as np

    # Flatten and sort variance values with indices
    flat_indices = np.argsort(variance_map.ravel())[::-1]  # descending
    coords = np.column_stack(np.unravel_index(flat_indices, variance_map.shape))

    template_com = center_of_mass(template_mask)
    mask_bool = template_mask.astype(bool)

    mean_in = variance_map[mask_bool].mean()
    std_in = variance_map[mask_bool].std()

    print("="*80)
    print(f"{'Rank':>4} | {'Variance':>9} | {'Y':>3} {'X':>3} | {'InMask':>7} | {'Dist2COM':>9} | {'Z-Score':>8}")
    print("-"*80)

    counted = 0
    visited = set()
    for y, x in coords:
        # Skip repeated or neighboring points
        if (y, x) in visited:
            continue
        visited.add((y, x))

        value = variance_map[y, x]
        in_mask = mask_bool[y, x]
        dist_to_com = np.sqrt((y - template_com[0])**2 + (x - template_com[1])**2)
        z_score = (value - mean_in) / std_in if std_in > 1e-3 else 0.0

        print(f"{counted+1:>4} | {value:9.2f} | {y:3d} {x:3d} | {str(in_mask):>7} | {dist_to_com:9.2f} | {z_score:8.2f}")

        counted += 1
        if counted >= num_peaks:
            break


def suppress_high_variance_peaks(variance_map, template_mask, z_thresh=3, velocity_array=None, method='none'):
    """
    Suppress isolated high-variance peaks in variance map,
    and optionally apply the same suppression to velocity array.
    
    Parameters:
        variance_map (np.ndarray): 2D temporal variance map.
        template_mask (np.ndarray): 2D binary mask of vessel region.
        z_thresh (float): Z-score threshold for defining outliers.
        velocity_array (np.ndarray or None): (T, H, W) velocity data.
        method (str): 'none', 'zero', or 'smooth' (applied to velocity).
    
    Returns:
        cleaned_map (np.ndarray): modified variance map.
        cleaned_velocity (np.ndarray or None): modified velocity array (if method != 'none').
    """
    mean_in = variance_map[template_mask > 0].mean()
    std_in = variance_map[template_mask > 0].std()
    z_map = (variance_map - mean_in) / (std_in + 1e-6)
    unstable_mask = z_map > z_thresh

    cleaned_map = variance_map.copy()
    cleaned_map[unstable_mask] = mean_in  # or set to 0 if you prefer

    if velocity_array is None or method == 'none':
        return cleaned_map, None

    cleaned_velocity = velocity_array.copy()

    if method == 'zero':
        for t in range(cleaned_velocity.shape[0]):
            cleaned_velocity[t][unstable_mask] = 0.0

    elif method == 'smooth':
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(cleaned_velocity, size=(1, 3, 3))
        for t in range(cleaned_velocity.shape[0]):
            cleaned_velocity[t][unstable_mask] = smoothed[t][unstable_mask]

    else:
        raise ValueError("Invalid method. Choose 'none', 'zero', or 'smooth'.")

    return cleaned_map, cleaned_velocity

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

def process_intra_subject_alignment(folder_path, output_path, axis='horizontal', threshold=1e-6,
                                     template_path=None, use_internal_template=True):
    os.makedirs(output_path, exist_ok=True)
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])[:1000]
    summary_rows = []
    # Load external template if applicable
    if not use_internal_template:
        if template_path is None:
            raise ValueError("External template path must be provided when use_internal_template=False")
        external_template = np.load(template_path).astype(np.uint8)
        print(f"[INFO] Loaded external template from {template_path}")
    else:
        external_template = None

    for i, fname in enumerate(tqdm(filenames)):
        if i % 50 == 0:
            print(f"\n[INFO] Processing sample {i+1}/{len(filenames)}: {fname}")
        path = os.path.join(folder_path, fname)
        velocity_array = np.load(path)

        # ------------------ Alignment ------------------
        aligned_velocity, ref_idx = perform_intra_subject_alignment(
            velocity_array,
            ref_strategy='similarity',
            threshold=threshold,
            external_template_mask=external_template,
            force_external_template=not use_internal_template
        )
        binary_mask = extract_binary_masks(velocity_array)
        aligned_mask = extract_binary_masks(aligned_velocity)
       
        # When external template is used (ref_idx is None), use a default frame index for visualization
        # and rely on the external mask for all diagnostics and alignment comparisons
        ref_frame_used = 15 if ref_idx is None else ref_idx
        template_for_analysis = external_template if ref_idx is None else aligned_mask[ref_idx]

        # ------------------ Diagnostics ------------------
        visualize_velocity_alignment(velocity_array, aligned_velocity, ref_frame_used, output_path, base_name=os.path.splitext(fname)[0])
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(aligned_velocity, aligned_mask, ref_frame_used)
        cleaned_variance_map, cleaned_velocity_array = suppress_high_variance_peaks(
            variance_map=variance_map,
            template_mask=template_for_analysis,
            z_thresh=3,
            velocity_array=aligned_velocity,
            method='smooth'  # or 'zero' or 'none'
        )
        
        analyze_top_variance_peaks(variance_map, template_for_analysis, num_peaks=5)
        print("\n[After Suppression]")
        analyze_top_variance_peaks(cleaned_variance_map, template_for_analysis, num_peaks=5)

        spatiotemporal_profile = extract_spatiotemporal_profile(cleaned_velocity_array, axis=axis)
        
        # ------------------ Variance Analysis ------------------
        print("\n[Variance Map Analysis]")
        analyze_variance_map(
            variance_map=cleaned_variance_map,
            template_mask=template_for_analysis,
            threshold=100,
            name=fname
        )

        result = {
            'filename': fname,
            'ref_idx': ref_frame_used,
            'aligned_velocity': cleaned_velocity_array,
            'time_profile': time_profile,
            'variance_map': cleaned_variance_map,
            'patch_mask': patch_mask,
            'ref_frame_img': template_for_analysis,
            'spatiotemporal_profile': spatiotemporal_profile,
            'binary_mask': binary_mask,
            'aligned_mask': aligned_mask,
            'original_mask': binary_mask
        }
        # Save aligned velocity array even if skipping visualizations
        base = os.path.splitext(fname)[0]
        np.save(os.path.join(output_path, f"{base}_aligned_velocity.npy"), result['aligned_velocity'])

        # ------------------ Additional Metrics ------------------
        additional_metrics = compute_velocity_alignment_metrics(
            aligned_velocity=cleaned_velocity_array,
            original_velocity=velocity_array,
            template_mask=template_for_analysis
        )
        # print("\n[Additional Velocity Metrics]")
        # for k, v in additional_metrics.items():
        #     print(f"{k}: {v:.4f}")

        # Optional: اگر یک template داری
        # combined_velocity_diagnostics(result['aligned_velocity'], template_mask)
        print("\n[Before Alignment Diagnostics]")
        combined_velocity_diagnostics(velocity_array, template_for_analysis)

        print("\n[After Alignment Diagnostics]")
        combined_velocity_diagnostics(cleaned_velocity_array, template_for_analysis)

        # Save visual + numerical summaries
        save_visual_outputs_intra_bspline(result, output_path, template_mask=template_for_analysis)
        visualize_deformable_alignment(binary_mask, aligned_mask, ref_frame_used, output_path, base_name=os.path.splitext(fname)[0])
        
        aligned_mask = extract_binary_masks(cleaned_velocity_array, threshold=threshold)
        # print("=" * 80)
        # print(f"[{fname}] Frame-wise Radius Summary (From Mask and Velocity)")
        # print(f"{'Frame':>5} | {'r_mask_before':>13} | {'r_mask_after':>13} | {'r_vel_before':>13} | {'r_vel_after':>13}")
        # print("-" * 80)
        # for i in range(velocity_array.shape[0]):
        #     vel_before_bin = (np.abs(velocity_array[i]) > threshold).astype(np.uint8)
        #     vel_after_bin = (np.abs(cleaned_velocity_array[i]) > threshold).astype(np.uint8)

        #     r_mask_before = np.sqrt(np.sum(binary_mask[i]) / np.pi)
        #     r_mask_after = np.sqrt(np.sum(aligned_mask[i]) / np.pi)
        #     r_vel_before = np.sqrt(np.sum(vel_before_bin) / np.pi)
        #     r_vel_after = np.sqrt(np.sum(vel_after_bin) / np.pi)

        #     print(f"{i:5d} | {r_mask_before:13.2f} | {r_mask_after:13.2f} | {r_vel_before:13.2f} | {r_vel_after:13.2f}")
        # print("=" * 80)

        import pandas as pd
 
        summary_rows.append({
            'filename': fname,
            'method': 'BSpline_mask',  # Change this per script: 'Affine', 'BSpline_mask', etc.
            'ref_idx': ref_frame_used,
            'max_variance': result['variance_map'].max(),
            'mean_variance': result['variance_map'].mean(),
            'snr': additional_metrics['in_out_energy_ratio'],
            'energy_preservation': additional_metrics['energy_preservation'],
            'mean_temporal_correlation': additional_metrics['mean_temporal_correlation'],
            'mean_ssim': additional_metrics['mean_ssim_with_mean_frame'],
        })

        # After the loop (outside the for-loop):
        df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(output_path, 'summary_metricsBSpline_mask.csv')
        df.to_csv(summary_csv_path, index=False)
        print(f"\n[Saved Summary CSV] {summary_csv_path}")



def perform_intra_subject_alignment(velocity_array, ref_strategy='similarity', threshold=1e-6,
                                     external_template_mask=None, force_external_template=False):
    """
    Performs intra-subject alignment using binary masks for BSpline registration,
    and applies the transforms to the corresponding velocity frames.
    
    Parameters:
        velocity_array (np.ndarray): shape (T, H, W), original velocity values.
        ref_strategy (str): 'similarity' | 'area' | 'template'
        threshold (float): for binary mask generation from velocity.
    
    Returns:
        aligned_velocity (np.ndarray): aligned velocity array.
        ref_idx (int): chosen reference frame index.
    """
    # Step 1: Extract binary masks
    binary_mask = extract_binary_masks(velocity_array, threshold=threshold)

    if force_external_template:
        ref_mask = external_template_mask.astype(np.uint8)
        ref_idx = -1  # external template, no internal frame index-not applicable: external reference used
        print("[INFO] Using external binary mask template.")
    else:
        ref_idx = choose_reference_frame(binary_mask, method=ref_strategy)
        ref_mask = binary_mask[ref_idx]
        print(f"[INFO] Chosen internal reference frame index: {ref_idx}")

    ref_com = center_of_mass(ref_mask)
    aligned_velocity = np.zeros_like(velocity_array)

    for i in range(velocity_array.shape[0]):
        if not force_external_template and i == ref_idx:
            aligned_velocity[i] = velocity_array[ref_idx]
            continue

        moving_mask = binary_mask[i]
        moving_velocity = velocity_array[i]

        # Step 3: COM alignment on mask (for stability)
        com_moving = center_of_mass(moving_mask)
        dy, dx = ref_com[0] - com_moving[0], ref_com[1] - com_moving[1]
        shifted_mask = shift(moving_mask, shift=(dy, dx), order=0, mode='constant')
        shifted_velocity = shift(moving_velocity, shift=(dy, dx), order=1, mode='constant')

        # Step 4: Compute BSpline transform between masks
        transform = get_bspline_transform(ref_mask, shifted_mask)
        # Print transform parameters
        params = transform.GetParameters()
        # print(f"[Frame {i}] Transform parameters (len={len(params)}):")
        # print(np.array(params).reshape(-1))  # Optional: reshape for readability

        # Step 5: Apply transform to velocity frame
        warped_velocity = apply_transform_to_velocity(shifted_velocity, ref_mask.shape, transform)
        aligned_velocity[i] = warped_velocity
        # if np.isnan(warped_velocity).any() or np.max(np.abs(warped_velocity)) < 1e-3:
        #     print(f"[WARNING] Frame {i} warped result might be invalid (low values or NaNs)")


    return aligned_velocity, None if force_external_template else ref_idx


if __name__ == "__main__":
    from tqdm import tqdm
    import os

    # Define input/output directories
    #input_folder = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/data"
    #input_folder = r"\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\results2"
    input_folder = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data\visout"
    #Y:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\output_bspline_intra_raw_velocity_1007337
    output_folder = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data\visout_interCircle10"
    #Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/output/inter_subject_BSpline_10
    # Define circular template (shared for all)
    template_path = r"P:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/templates/circular_template_r10.npy"
    # Choose axis for spatiotemporal profile ('horizontal', 'vertical', 'full_x', 'full_y')
    axis = 'horizontal'

    # Run the new intra-subject alignment pipeline
    process_intra_subject_alignment(
        folder_path=input_folder,
        output_path=output_folder,
        axis=axis,
        threshold=1e-6,
        template_path=template_path,
        use_internal_template=False
    )
