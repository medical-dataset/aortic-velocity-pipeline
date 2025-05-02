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
from scipy.ndimage import center_of_mass, shift
from skimage.transform import AffineTransform, warp
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
from tqdm import tqdm
# --- Helper Functions ---
def extract_binary_masks(arr, threshold=1e-6):
    return (np.abs(arr) > threshold).astype(np.uint8)

def com_align(mask, target_com):
    cy, cx = center_of_mass(mask)
    dy, dx = target_com[0] - cy, target_com[1] - cx
    return shift(mask, (dy, dx), order=0, mode='constant'), (dy, dx)

def compute_affine_to_template(src_mask, template_mask):
    from skimage.measure import regionprops
    props_src = regionprops(src_mask.astype(int))[0]
    props_tgt = regionprops(template_mask.astype(int))[0]

    src_c = props_src.centroid
    tgt_c = props_tgt.centroid
    src_m = props_src.moments_central
    tgt_m = props_tgt.moments_central

    scale_y = np.sqrt(tgt_m[2, 0]) / np.sqrt(src_m[2, 0])
    scale_x = np.sqrt(tgt_m[0, 2]) / np.sqrt(src_m[0, 2])
    print(f"[Affine Transform] Scale: (x={scale_x:.2f}, y={scale_y:.2f}) | Translation: ({tgt_c[1] - scale_x * src_c[1]:.2f}, {tgt_c[0] - scale_y * src_c[0]:.2f})")

    transform = AffineTransform(scale=(scale_x, scale_y),
        translation=(tgt_c[1] - scale_x * src_c[1], tgt_c[0] - scale_y * src_c[0]))
    return transform

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
        aff = compute_affine_to_template(shifted, template)
        aligned[i] = apply_affine(shifted, aff)
        transforms.append(aff)
    return aligned, transforms

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


def visualize_affine_alignment(mask_array_before, mask_array_after, ref_idx, output_path=None, base_name="alignment", num_show=5, template_mask=None):
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

def visualize_velocity_alignment(velocity_array_before, velocity_array_after, ref_idx, output_path=None, base_name="velocity", num_show=5, crop_center=(96, 96), crop_radius=20, threshold=1e-6,  template_mask=None):
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





def process_affine_folder(folder_path, output_path, template_path, axis='horizontal'):
    """
    Processes all velocity .npy files in a folder using COM + affine alignment to a given template.
    Saves aligned velocity, variance maps, time profiles, spatiotemporal profiles, and visualizations.

    Parameters:
        folder_path (str): Directory with input .npy files (velocity arrays).
        output_path (str): Directory to save outputs.
        template_path (str): Path to the circular template .npy file.
        axis (str): Axis for spatiotemporal profile ('horizontal', 'vertical', 'full_x', 'full_y').
    """

    os.makedirs(output_path, exist_ok=True)
    template = np.load(template_path)
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    for fname in tqdm(filenames):
        path = os.path.join(folder_path, fname)
        velocity_array = np.load(path)
        binary_mask = extract_binary_masks(velocity_array)

        aligned_masks, transforms = affine_align_masks_and_collect_transforms(binary_mask, template)

        ref_idx = 15
        aligned_velocity = apply_affine_to_velocity(velocity_array, transforms, template)
        if aligned_velocity is None or np.all(aligned_velocity == 0):
            print(f"[Error] Aligned velocity is invalid for {fname}. Skipping.")
            continue

        visualize_velocity_alignment(
            velocity_array_before=velocity_array,
            velocity_array_after=aligned_velocity,
            ref_idx=ref_idx,
            output_path=output_path,
            base_name=os.path.splitext(fname)[0],
            template_mask=template
        )
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(
            aligned_velocity, aligned_masks, ref_idx=ref_idx)
        spatiotemporal_profile = extract_spatiotemporal_profile(
            aligned_velocity, axis=axis)

        result = {
            'filename': fname,
            'ref_idx': ref_idx,
            'aligned_velocity': aligned_velocity,
            'time_profile': time_profile,
            'variance_map': variance_map,
            'patch_mask': patch_mask,
            'ref_frame_img': aligned_velocity[ref_idx],
            'spatiotemporal_profile': spatiotemporal_profile,
            'binary_mask': binary_mask,
            'original_mask': binary_mask,
            'aligned_mask': aligned_masks
        }
        
        combined_velocity_diagnostics(result['aligned_velocity'], template, threshold=1e-6)

        save_visual_outputs_affine(result, output_path, template_mask=None)
        visualize_affine_alignment(
            mask_array_before=result['original_mask'],
            mask_array_after=result['aligned_mask'],
            ref_idx=result['ref_idx'],
            output_path=output_path,
            base_name=os.path.splitext(fname)[0],
            template_mask=template
        )
        # Print radius summary for terminal
        print("=" * 80)
        print(f"[{fname}] Frame-wise Radius Summary (From Mask and Velocity)")
        print(f"{'Frame':>5} | {'r_mask_before':>13} | {'r_mask_after':>13} | {'r_vel_before':>13} | {'r_vel_after':>13}")
        print("-" * 80)

        for i in range(velocity_array.shape[0]):
            vel_before_bin = (np.abs(velocity_array[i]) > 1e-6).astype(np.uint8)
            vel_after_bin = (np.abs(aligned_velocity[i]) > 1e-6).astype(np.uint8)
            
            r_mask_before = np.sqrt(np.sum(binary_mask[i]) / np.pi)
            r_mask_after = np.sqrt(np.sum(aligned_masks[i]) / np.pi)
            r_vel_before = np.sqrt(np.sum(vel_before_bin) / np.pi)
            r_vel_after = np.sqrt(np.sum(vel_after_bin) / np.pi)

            print(f"{i:5d} | {r_mask_before:13.2f} | {r_mask_after:13.2f} | {r_vel_before:13.2f} | {r_vel_after:13.2f}")
        print("=" * 80)



# -------------------- Run Entry Point --------------------
if __name__ == "__main__":
    # Define input/output directories
    input_folder = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/data"
    output_folder = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/output/output_affine_circular_correct"

    # Define circular template (shared for all)
    template_path = r"Y:/Projects/DeepFlow/deepFlowDocker/scripts/Registration/templates/circle_template.npy"

    # Choose axis for spatiotemporal profile ('horizontal', 'vertical', 'full_x', 'full_y')
    axis = 'horizontal'

    # Run the affine alignment process
    process_affine_folder(
        folder_path=input_folder,
        output_path=output_folder,
        template_path=template_path,
        axis=axis
    )







'''
def save_outputs(result, output_dir):
    """
    Save plots and aligned velocity array safely.
    Skips problematic plots (NaN, constant values, too small patches) and prints a final summary.
    """

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(result['filename'])[0]

    # Counters for summary
    count_saved = 0
    count_skipped = 0

    def safe_imshow_with_colorbar(data, cmap, title, fname, label):
        """Helper function to safely plot a 2D map with colorbar."""
        nonlocal count_saved, count_skipped
        plt.figure(figsize=(6, 6))
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        if np.isnan(vmin) or np.isnan(vmax):
            print(f"[Warning] Skipping plot '{title}': Data contains only NaNs.")
            plt.close()
            count_skipped += 1
            return

        if vmin == vmax:
            plt.imshow(data, cmap=cmap)
            plt.title(f"{title} (constant value: {vmin:.2f})")
            print(f"[Warning] Plotting '{title}' without colorbar (constant value).")
        else:
            im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, label=label)
            plt.title(title)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        count_saved += 1
        print(f"[Info] Saved: {fname}")

    # --- Plot Time-Velocity Profile ---
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(result['time_profile'])
        plt.title(f"Time-Velocity Profile: {base}")
        plt.xlabel("Frame")
        plt.ylabel("Mean Velocity")
        plt.grid(True)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{base}_time_profile.png")
        plt.savefig(fname)
        plt.close()
        count_saved += 1
        print(f"[Info] Saved: {fname}")
    except Exception as e:
        print(f"[Warning] Failed to plot Time-Velocity Profile: {e}")
        count_skipped += 1

    # --- Variance Map ---
    safe_imshow_with_colorbar(
        data=result['variance_map'],
        cmap='hot',
        title="Variance Map (Turbulence)",
        fname=os.path.join(output_dir, f"{base}_variance_map.png"),
        label="Variance"
    )

    # --- Patch Overlay on Reference Frame ---
    # --- Patch Overlay on Reference Frame ---
    try:
        plt.figure(figsize=(6, 6))
        ref_img = result['ref_frame_img']
        patch_mask = result['patch_mask']

        vmin = np.nanmin(ref_img)
        vmax = np.nanmax(ref_img)

        if np.isnan(vmin) or np.isnan(vmax):
            print(f"[Warning] Skipping Patch Overlay: Reference frame contains NaN.")
            plt.close()
            count_skipped += 1
            return
        else:
            patch_pixels = patch_mask.sum()
            if patch_pixels < 9:  # Check before drawing contour
                print(f"[Warning] Skipping Patch Overlay: Patch mask too small (pixels={patch_pixels}).")
                plt.close()
                count_skipped += 1
                return  # 🔥 حتماً این را اضافه کن
            else:
                plt.imshow(ref_img, cmap='bwr', vmin=vmin, vmax=vmax)
                plt.contour(patch_mask, colors='yellow')
                plt.title(f"Patch Overlay (Frame {result['ref_idx']})")
                if vmin != vmax:
                    plt.colorbar(label="Velocity")
                else:
                    print(f"[Warning] Constant reference frame detected: Skipping colorbar.")
                plt.axis('off')
                plt.tight_layout()
                fname = os.path.join(output_dir, f"{base}_patch_overlay.png")
                plt.savefig(fname)
                plt.close()
                count_saved += 1
                print(f"[Info] Saved: {fname}")
    except Exception as e:
        print(f"[Warning] Failed to plot Patch Overlay: {e}")
        plt.close()
        count_skipped += 1



    # --- Spatiotemporal Profile ---
    try:
        spatiotemporal_profile = result['spatiotemporal_profile']
        if np.isnan(spatiotemporal_profile).all():
            print(f"[Warning] Skipping Spatiotemporal Profile: Data is all NaN.")
            count_skipped += 1
        else:
            plt.figure(figsize=(10, 5))
            sns.heatmap(spatiotemporal_profile, cmap='coolwarm', cbar_kws={'label': 'Velocity'})
            plt.title(f"Spatiotemporal Profile: {base}")
            plt.xlabel("Spatial Axis")
            plt.ylabel("Frame")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            fname = os.path.join(output_dir, f"{base}_spatiotemporal_profile.png")
            plt.savefig(fname)
            plt.close()
            count_saved += 1
            print(f"[Info] Saved: {fname}")
    except Exception as e:
        print(f"[Warning] Failed to plot Spatiotemporal Profile: {e}")
        plt.close()
        count_skipped += 1

    # --- Save aligned velocity array (.npy) ---
    try:
        aligned_velocity_path = os.path.join(output_dir, f"{base}_aligned_velocity.npy")
        np.save(aligned_velocity_path, result['aligned_velocity'])
        count_saved += 1
        print(f"[Info] Saved aligned velocity array: {aligned_velocity_path}")
    except Exception as e:
        print(f"[Warning] Failed to save aligned velocity array: {e}")
        count_skipped += 1

    # --- Final Summary ---
    print("="*60)
    print(f"Summary for {base}:")
    print(f"  {count_saved} files saved successfully.")
    print(f"  {count_skipped} files skipped due to invalid data.")
    print("="*60)
'''










'''
import os
import numpy as np
from scipy.ndimage import shift, center_of_mass, affine_transform
from skimage.transform import AffineTransform, warp
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
from aorta_velocity_pipeline import (
    extract_binary_masks,
    extract_time_profile_and_variance,
    extract_spatiotemporal_profile,
    save_analysis_outputs,
    visualize_alignment_effect,
    create_alignment_gif
)

def align_masks_to_circle_template(binary_masks, template_mask):
    aligned_masks = np.zeros_like(binary_masks)
    shifts = []
    transforms = []
    
    # Compute COM of template once
    template_com = center_of_mass(template_mask)

    for i in range(binary_masks.shape[0]):
        mask = binary_masks[i]
        com = center_of_mass(mask)
        dy, dx = template_com[0] - com[0], template_com[1] - com[1]
        shifted = shift(mask, (dy, dx), order=0, mode='constant', cval=0)

        # Compute affine transform from shifted mask to template
        tform = AffineTransform()
        tform.estimate(template_mask, shifted)  # binary -> binary alignment
        warped = warp(shifted, inverse_map=tform.inverse, order=0, preserve_range=True)

        aligned_masks[i] = (warped > 0.5).astype(np.uint8)
        shifts.append((dy, dx))
        transforms.append(tform.params)

    return aligned_masks, shifts, transforms

def align_velocity_to_mask_shifts(velocity_data, shifts):
    aligned_velocity = np.zeros_like(velocity_data)
    for i, (dy, dx) in enumerate(shifts):
        aligned_velocity[i] = shift(velocity_data[i], (dy, dx), order=1, mode='constant', cval=0)
    return aligned_velocity

def run_affine_alignment_pipeline(template_path, test_npy_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load template and sample ---
    template_mask = np.load(template_path)  # shape: (H, W)
    velocity_data = np.load(test_npy_path)  # shape: (30, H, W)

    binary_mask = extract_binary_masks(velocity_data)
    aligned_masks, shifts, _ = align_masks_to_circle_template(binary_mask, template_mask)
    aligned_velocity = align_velocity_to_mask_shifts(velocity_data, shifts)

    # --- Select reference frame and extract features ---
    ref_idx = 15  # or use a dynamic method
    time_profile, variance_map, patch_mask = extract_time_profile_and_variance(
        aligned_velocity, mask_array=aligned_masks, ref_idx=ref_idx)
    spatiotemporal_profile = extract_spatiotemporal_profile(aligned_velocity, axis='horizontal')

    result = {
        'filename': os.path.basename(test_npy_path),
        'ref_idx': ref_idx,
        'aligned_velocity': aligned_velocity,
        'time_profile': time_profile,
        'variance_map': variance_map,
        'patch_mask': patch_mask,
        'spatiotemporal_profile': spatiotemporal_profile,
        'binary_mask': aligned_masks,
        'ref_frame_img': aligned_velocity[ref_idx],
        'original_mask': binary_mask,
        'aligned_mask': aligned_masks
    }

    save_analysis_outputs(result, output_dir)
    print(f" Done. Results saved to: {output_dir}")

# ========== Example usage ==========
if __name__ == '__main__':
    template_path = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\templates\circle_template.npy"
    test_npy_path = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data\20213_2_0_masked.npy"
    output_dir = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\outputcircularaffine"

    run_affine_alignment_pipeline(template_path, test_npy_path, output_dir)








# aorta_velocity_pipeline_affine.py
# This script processes 30-frame aortic VENC velocity sequences and performs COM-based alignment
# followed by affine registration to a standard circular template. Outputs include spatiotemporal features,
# alignment visualization, variance maps, and GIF animations.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import center_of_mass, shift
from skimage.transform import AffineTransform, warp
from skimage.metrics import structural_similarity as ssim
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage import map_coordinates

# === Load Template ===
def load_template(path):
    return np.load(path)

# === Extract Binary Masks from Velocity Array ===
def extract_binary_masks(masked_velocity_array, threshold=1e-6):
    return (np.abs(masked_velocity_array) > threshold).astype(np.uint8)

# === Align to Template via Affine ===
def affine_align_frame_to_template(frame, template):
    from skimage.registration import phase_cross_correlation
    from skimage.transform import AffineTransform, warp

    # Step 1: Center of Mass shift
    com_template = center_of_mass(template)
    com_frame = center_of_mass(frame)
    shift_yx = np.array(com_template) - np.array(com_frame)
    shifted = shift(frame, shift=shift_yx, order=0, mode='constant')

    # Step 2: Affine registration (estimate transform using phase correlation)
    shift_estimate, _, _ = phase_cross_correlation(template, shifted, upsample_factor=10)
    affine = AffineTransform(translation=-shift_estimate[::-1])
    aligned = warp(shifted, inverse_map=affine.inverse, preserve_range=True, order=0)

    return aligned.astype(np.uint8)

# === Align All Frames ===
def align_all_frames_to_template(mask_array, template):
    aligned = np.zeros_like(mask_array)
    for i in range(mask_array.shape[0]):
        aligned[i] = affine_align_frame_to_template(mask_array[i], template)
    return aligned

# === Extract Time Profile + Variance ===
def extract_time_profile_and_variance(velocity_array, mask_array, ref_idx, patch_size=5):
    cy, cx = center_of_mass(mask_array[ref_idx])
    cy, cx = int(round(cy)), int(round(cx))
    r = patch_size // 2
    patch_mask = np.zeros_like(mask_array[0], dtype=bool)
    patch_mask[cy - r:cy + r + 1, cx - r:cx + r + 1] = True
    time_profile = velocity_array[:, patch_mask].mean(axis=1)
    variance_map = np.var(velocity_array, axis=0)
    return time_profile, variance_map, patch_mask

# === Spatiotemporal ===
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

# === Main Runner ===
def run_affine_alignment(input_folder, template_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    template = load_template(template_path)

    fname = [f for f in os.listdir(input_folder) if f.endswith('.npy')][0]
    path = os.path.join(input_folder, fname)
    velocity_array = np.load(path)
    binary_mask = extract_binary_masks(velocity_array)

    aligned_mask = align_all_frames_to_template(binary_mask, template)
    ref_idx = 15
    aligned_velocity = np.zeros_like(velocity_array)
    for i in range(30):
        aligned_velocity[i] = velocity_array[i] * aligned_mask[i]

    time_profile, variance_map, patch_mask = extract_time_profile_and_variance(
        aligned_velocity, aligned_mask, ref_idx)
    spatiotemporal_profile = extract_spatiotemporal_profile(aligned_velocity)

    result = {
        'filename': fname,
        'ref_idx': ref_idx,
        'aligned_velocity': aligned_velocity,
        'time_profile': time_profile,
        'variance_map': variance_map,
        'patch_mask': patch_mask,
        'binary_mask': binary_mask,
        'ref_frame_img': aligned_velocity[ref_idx],
        'original_mask': binary_mask,
        'aligned_mask': aligned_mask
    }

    # Visualization utilities must be available in scope
    from deformable_outputs import save_visual_outputs_deformable
    save_visual_outputs_deformable(result, output_folder)

# Example usage:
# run_affine_alignment("/path/to/input", "/path/to/template.npy", "/path/to/output")




















import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import shift, center_of_mass
from skimage.metrics import structural_similarity as ssim
from skimage.transform import AffineTransform, warp
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def extract_binary_masks(masked_velocity_array, threshold=1e-6):
    return (np.abs(masked_velocity_array) > threshold).astype(np.uint8)


def com_align(mask, target_com):
    cy, cx = center_of_mass(mask)
    dy, dx = target_com[0] - cy, target_com[1] - cx
    return shift(mask, (dy, dx), order=0, mode='constant', cval=0), (dy, dx)

def compute_affine_to_template(src_mask, template_mask):
    from skimage.measure import regionprops
    props_src = regionprops(src_mask.astype(int))[0]
    props_tgt = regionprops(template_mask.astype(int))[0]

    src_c = props_src.centroid
    tgt_c = props_tgt.centroid
    src_m = props_src.moments_central
    tgt_m = props_tgt.moments_central

    scale_y = np.sqrt(tgt_m[2, 0]) / np.sqrt(src_m[2, 0])
    scale_x = np.sqrt(tgt_m[0, 2]) / np.sqrt(src_m[0, 2])

    transform = AffineTransform(scale=(scale_x, scale_y), translation=(tgt_c[1] - scale_x * src_c[1], tgt_c[0] - scale_y * src_c[0]))
    return transform

def apply_affine(mask, transform):
    return warp(mask, inverse_map=transform.inverse, output_shape=mask.shape, preserve_range=True).astype(np.uint8)

def affine_align_masks_to_template(mask_array, template_mask):
    aligned = np.zeros_like(mask_array)
    shifts = []
    ref_com = center_of_mass(template_mask)
    for i in range(mask_array.shape[0]):
        shifted, _ = com_align(mask_array[i], ref_com)
        aff = compute_affine_to_template(shifted, template_mask)
        aligned[i] = apply_affine(shifted, aff)
    return aligned

def align_velocity_data_to_reference(velocity_array, shifts):
    aligned_velocity = np.zeros_like(velocity_array)
    for i, (dy, dx) in enumerate(shifts):
        aligned_velocity[i] = shift(velocity_array[i], (dy, dx), order=1, mode='constant', cval=0)
    return aligned_velocity

# ... reuse other utility functions: extract_time_profile_and_variance, etc.

# Final call like process_velocity_folder but using affine_align_masks_to_template instead of rigid alignment.

# Example:
# template_path = 'path_to_circular_template.npy'
# template_mask = np.load(template_path)
# results, templates = process_velocity_folder(..., template=template_mask)












# affine_aorta_alignment.py
# --------------------------------------------
# This script aligns all frames of velocity-encoded aortic MRI data to a fixed circular template using affine registration.
# It includes full visualization and output saving similar to the COM-aligned pipeline.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import AffineTransform, warp
from scipy.ndimage import center_of_mass
from skimage.metrics import structural_similarity as ssim
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
import matplotlib.patches as mpatches

# ==========================
# --- Helper Functions ---
# ==========================
def extract_binary_masks(masked_velocity_array, threshold=1e-6):
    return (np.abs(masked_velocity_array) > threshold).astype(np.uint8)

def load_template(template_path):
    return np.load(template_path)

def affine_align_to_template(src_mask, template):
    src_com = np.array(center_of_mass(src_mask))
    tgt_com = np.array(center_of_mass(template))
    dy, dx = tgt_com - src_com
    tform = AffineTransform(translation=(dx, dy))
    aligned = warp(src_mask, tform.inverse, order=0, preserve_range=True).astype(np.uint8)
    return aligned, (dy, dx)

def affine_align_velocity(velocity_frame, dy, dx):
    tform = AffineTransform(translation=(dx, dy))
    return warp(velocity_frame, tform.inverse, order=1, preserve_range=True)

def extract_time_profile_and_variance(velocity_array, mask_array=None, ref_mask=None, patch_size=5):
    if mask_array is not None and ref_mask is not None:
        cy, cx = center_of_mass(ref_mask)
        cy, cx = int(round(cy)), int(round(cx))
    else:
        cy, cx = velocity_array.shape[1] // 2, velocity_array.shape[2] // 2
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
        raise ValueError("Axis must be 'horizontal', 'vertical', 'full_x', or 'full_y'")

def visualize_alignment_effect(before, after, template, save_path=None):
    n = before.shape[0]
    fig, axs = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        axs[0, i].imshow(before[i], cmap='gray')
        axs[0, i].set_title(f"Before Frame {i}")
        axs[0, i].axis('off')

        axs[1, i].imshow(after[i], cmap='gray')
        axs[1, i].contour(template, levels=[0.5], colors='orange')
        axs[1, i].set_title(f"After Align {i}")
        axs[1, i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_alignment_gif(before, after, save_path='affine_alignment.gif'):
    frames = []
    for i in range(before.shape[0]):
        fig, ax = plt.subplots(figsize=(5, 5))
        com_before = center_of_mass(before[i])
        com_after = center_of_mass(after[i])
        cy, cx = int(round(com_before[0])), int(round(com_before[1]))
        ax.set_xlim(cx - 40, cx + 40)
        ax.set_ylim(cy + 40, cy - 40)
        ax.imshow(before[i], cmap='Blues', alpha=0.7)
        ax.contour(after[i], levels=[0.5], colors='orange', linewidths=2)
        ax.arrow(com_before[1], com_before[0], com_after[1] - com_before[1], com_after[0] - com_before[0],
                 color='cyan', head_width=2, head_length=2, length_includes_head=True)
        ax.set_title(f"Frame {i}")
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(save_path, frames, fps=0.4)

def plot_profiles_and_maps(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(result['filename'])[0]
    plt.figure(figsize=(8, 4))
    plt.plot(result['time_profile'])
    plt.title("Time-Velocity Profile")
    plt.savefig(os.path.join(output_dir, f"{base}_time_profile.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(result['variance_map'], cmap='hot')
    plt.title("Variance Map")
    plt.savefig(os.path.join(output_dir, f"{base}_variance_map.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(result['ref_frame_img'], cmap='bwr')
    plt.contour(result['patch_mask'], colors='yellow')
    plt.title("Reference Frame + Patch")
    plt.savefig(os.path.join(output_dir, f"{base}_patch_overlay.png"))
    plt.close()

    np.save(os.path.join(output_dir, f"{base}_aligned_velocity.npy"), result['aligned_velocity'])
'''




