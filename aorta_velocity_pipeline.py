# aorta_velocity_pipeline.py ###This be used by run_analysis.py
##Rigid one Based on COM(center_of_mass)
# ----------------------------
# This script provides the full pipeline to process masked aortic velocity .npy files,
# align them, analyze time and spatial features, and save output plots and data.

#import matplotlib
#matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import shift, center_of_mass
from skimage.metrics import structural_similarity as ssim
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# -------------------- Core Functions --------------------
def extract_binary_masks(masked_velocity_array, threshold=1e-6):
    """Converts a (30, 192, 192) masked velocity array to binary masks (1 inside aorta, 0 outside)."""
    return (np.abs(masked_velocity_array) > threshold).astype(np.uint8)

def choose_reference_frame(mask_array, method='similarity', external_template=None):
    """ Selects a reference frame from a (30, 192, 192) binary mask array using one of three methods:
    - 'similarity': frame most similar to others (L2 or SSIM)
    - 'area': frame with mask area closest to the mean
    - 'template': frame most similar to an external template
    Returns: index of reference frame"""

    num_frames = mask_array.shape[0]
    areas = mask_array.sum(axis=(1, 2))

    if method == 'area':
        mean_area = np.mean(areas)
        return np.argmin(np.abs(areas - mean_area))

    elif method == 'similarity':
        total_scores = []
        for i in range(num_frames):
            scores = [ssim(mask_array[i], mask_array[j]) for j in range(num_frames) if i != j]
            total_scores.append(np.mean(scores))
        return np.argmax(total_scores)

    elif method == 'template':
        if external_template is None:
            raise ValueError("External template must be provided for 'template' method.")
        scores = [ssim(mask_array[i], external_template) for i in range(num_frames)]
        return np.argmax(scores)
    else:
        raise ValueError("Invalid method")

def align_masks_to_reference(mask_array, ref_idx):
    aligned_array = np.zeros_like(mask_array)
    ref_com = center_of_mass(mask_array[ref_idx])
    shifts = []
    for i in range(mask_array.shape[0]):
        com = center_of_mass(mask_array[i])
        dy, dx = ref_com[0] - com[0], ref_com[1] - com[1]
        aligned_array[i] = shift(mask_array[i], (dy, dx), order=0, mode='constant', cval=0)
        shifts.append((dy, dx))
    return aligned_array, shifts

def align_velocity_data_to_reference(velocity_array, shifts):
    aligned_velocity = np.zeros_like(velocity_array)
    for i, (dy, dx) in enumerate(shifts):
        aligned_velocity[i] = shift(velocity_array[i], (dy, dx), order=1, mode='constant', cval=0)
    return aligned_velocity


def extract_time_profile_and_variance(velocity_array, mask_array=None, ref_idx=None, patch_size=5):
    if mask_array is not None and ref_idx is not None:
        cy, cx = center_of_mass(mask_array[ref_idx])
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
        return np.mean(velocity_array, axis=1)  # (30, W)
    elif axis == 'full_y':
        return np.mean(velocity_array, axis=2)  # (30, H)
    else:
        raise ValueError("Axis must be 'horizontal', 'vertical', 'full_x', or 'full_y'")

def build_soft_velocity_template(aligned_velocity_array_list):
    stacked = np.stack(aligned_velocity_array_list, axis=0)
    return np.mean(stacked, axis=0)

def compare_patient_to_template(patient_velocity_array, template_velocity_array, method='ssim'):
    similarity_scores = []
    for i in range(patient_velocity_array.shape[0]):
        if method == 'ssim':
            score, _ = ssim(patient_velocity_array[i], template_velocity_array[i], full=True)
        elif method == 'l2':
            score = -np.linalg.norm(patient_velocity_array[i] - template_velocity_array[i])
        else:
            raise ValueError("Invalid method")
        similarity_scores.append(score)
    return similarity_scores

def plot_spatiotemporal_profile(profile, axis='horizontal'):
    plt.figure(figsize=(12, 6))
    sns.heatmap(profile, cmap='coolwarm', cbar_kws={'label': 'Velocity'}, xticklabels=10, yticklabels=2)
    plt.title(f"Spatiotemporal Velocity Profile ({axis} line)")
    plt.xlabel("Spatial Position")
    plt.ylabel("Time Frame")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def visualize_alignment_effect(mask_array_before, mask_array_after, ref_idx, num_show=5, save_path=None):
    """
    Visualizes before and after alignment masks around the reference frame.
    Shows the amount of shift (dy, dx) applied to each frame.

    Parameters:
        mask_array_before: original binary masks (30, H, W)
        mask_array_after: aligned binary masks (30, H, W)
        ref_idx: index of the reference frame used for alignment
        num_show: number of frames before/after to show
    """

    indices = list(range(max(0, ref_idx - num_show), min(30, ref_idx + num_show + 1)))
    n = len(indices)
    ref_com = center_of_mass(mask_array_before[ref_idx])

    fig, axs = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(f"Alignment Visualization (Reference Frame: {ref_idx})", fontsize=14)

    for i, idx in enumerate(indices):
        axs[0, i].imshow(mask_array_before[idx], cmap='gray')
        axs[0, i].set_title(f"Before Align\nFrame {idx}")
        axs[0, i].axis('off')

        axs[1, i].imshow(mask_array_after[idx], cmap='gray')
        axs[1, i].set_title(f"After Align\nFrame {idx}")
        axs[1, i].axis('off')

        current_com = center_of_mass(mask_array_before[idx])
        dy = current_com[0] - ref_com[0]
        dx = current_com[1] - ref_com[1]

        axs[1, i].text(2, 10, f"Shift: dy={dy:+.1f}\ndx={dx:+.1f}", color='yellow', fontsize=9,
                       bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_alignment_gif(mask_array_before, mask_array_after, shifts=None, save_path='alignment.gif'):
    import matplotlib.pyplot as plt
    import numpy as np
    import imageio
    from scipy.ndimage import center_of_mass
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.patches as mpatches

    frames = []
    num_frames = mask_array_before.shape[0]
    crop_size = 80  # برای زوم روی ماسک

    for i in range(num_frames):
        fig, ax = plt.subplots(figsize=(5, 5))

        # مرکز تصویر برای زوم
        com = center_of_mass(mask_array_before[i])
        cy, cx = int(round(com[0])), int(round(com[1]))
        r = crop_size // 2
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy + r, cy - r)  # چون محور y در تصاویر بالا به پایینه

        # نمایش ماسک قبل از الاین - آبی
        ax.imshow(mask_array_before[i], cmap='Blues', alpha=0.8)

        # فقط کانتور ماسک بعد از الاین - نارنجی
        ax.contour(mask_array_after[i], levels=[0.5], colors='orange', linewidths=2)

        # رسم فلش مرکز جرم
        com_before = center_of_mass(mask_array_before[i])
        com_after = center_of_mass(mask_array_after[i])
        dx = com_after[1] - com_before[1]
        dy = com_after[0] - com_before[0]
        ax.arrow(com_before[1], com_before[0], dx, dy, color='cyan',
                 head_width=2, head_length=2, length_includes_head=True)

        ax.set_title(f"Frame {i} | Shift: dy={dy:+.1f}, dx={dx:+.1f}", fontsize=10)
        ax.axis('off')

        # legend دستی
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






def show_template_frame(template, frame_idx=10):
    plt.imshow(template[frame_idx], cmap='bwr')
    plt.title(f"Template Velocity (Frame {frame_idx})")
    plt.colorbar(label="Mean Velocity")
    plt.tight_layout()
    plt.show()

def plot_similarity_over_time(similarity_scores):
    plt.plot(similarity_scores)
    plt.title("Similarity to Template Over Time")
    plt.xlabel("Frame")
    plt.ylabel("SSIM Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_patch_on_velocity_frame(velocity_array, patch_mask, frame_idx=15):
    plt.figure(figsize=(6, 6))
    plt.imshow(velocity_array[frame_idx], cmap='bwr')
    plt.contour(patch_mask, colors='yellow')
    plt.title(f"Velocity Frame {frame_idx} with Patch Overlay")
    plt.colorbar(label="Velocity")
    plt.tight_layout()
    plt.show()

def plot_double_panel_variance_and_mask(variance_map, binary_mask, ref_idx):
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
    axs[0].imshow(variance_map, cmap='hot')
    axs[0].set_title("Temporal Variance Map")
    axs[1].imshow(variance_map * binary_mask[ref_idx], cmap='hot')
    axs[1].set_title("Variance × Binary Mask")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Fully Safe and Robust save_analysis_outputs 

def save_analysis_outputs(result, output_dir):
    import warnings
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(result['filename'])[0]

    # --- Time-Velocity Profile ---
    plt.figure(figsize=(8, 4))
    plt.plot(result['time_profile'])
    plt.title(f"Time-Velocity Profile: {base_name}")
    plt.xlabel("Frame")
    plt.ylabel("Mean Velocity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_time_profile.png"))
    plt.close()

    # --- Variance Map ---
    vmin = np.nanmin(result['variance_map'])
    vmax = np.nanmax(result['variance_map'])
    plt.figure(figsize=(5, 5))
    if vmin != vmax:
        im = plt.imshow(result['variance_map'], cmap='jet', vmin=vmin, vmax=vmax)  #hot
        plt.colorbar(im, label="Temporal Variance")
    else:
        im = plt.imshow(result['variance_map'], cmap='jet')
    plt.title(f"Turbulence Map: {base_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_variance_map.png"))
    plt.close()

    # --- Zoomed Crops from Variance Map ---
    from scipy.ndimage import center_of_mass
    crop_size = 40
    cy, cx = center_of_mass(result['binary_mask'][result['ref_idx']])
    cy, cx = int(round(cy)), int(round(cx))
    r = crop_size // 2
    cropped = result['variance_map'][cy - r:cy + r, cx - r:cx + r]

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
    plt.savefig(os.path.join(output_dir, f"{base_name}_variance_zoom_nearest.png"))
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
    plt.savefig(os.path.join(output_dir, f"{base_name}_variance_zoom_bilinear.png"))
    plt.close()

    # --- Spatiotemporal Profile ---
    plt.figure(figsize=(10, 5))
    sns.heatmap(result['spatiotemporal_profile'], cmap='coolwarm', cbar_kws={'label': 'Velocity'})
    plt.title(f"Spatiotemporal Profile: {base_name}")
    plt.xlabel("Spatial Position")
    plt.ylabel("Frame")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_spatiotemporal.png"))
    plt.close()

    # --- Velocity Frame + Patch Overlay ---
    vmin = np.nanmin(result['ref_frame_img'])
    vmax = np.nanmax(result['ref_frame_img'])
    plt.figure(figsize=(6, 6))
    if vmin != vmax:
        im = plt.imshow(result['ref_frame_img'], cmap='bwr', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Velocity")
    else:
        im = plt.imshow(result['ref_frame_img'], cmap='bwr')
    plt.contour(result['patch_mask'], colors='yellow')
    plt.title(f"Patch Overlay on Velocity (Frame {result['ref_idx']})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_velocity_patch.png"))
    plt.close()

    # --- Variance × Binary Mask (Double Panel) ---
    masked_variance = result['variance_map'] * result['binary_mask'][result['ref_idx']]
    vmin = np.nanmin(masked_variance)
    vmax = np.nanmax(masked_variance)
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
    axs[0].imshow(result['variance_map'], cmap='hot')
    axs[0].set_title("Temporal Variance Map")
    if vmin != vmax:
        axs[1].imshow(masked_variance, cmap='hot', vmin=vmin, vmax=vmax)
    else:
        axs[1].imshow(masked_variance, cmap='hot')
    axs[1].set_title("Variance × Binary Mask")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_variance_combo.png"))
    plt.close()

    # --- Reference Frame Only ---
    vmin = np.nanmin(result['ref_frame_img'])
    vmax = np.nanmax(result['ref_frame_img'])
    plt.figure(figsize=(6, 6))
    if vmin != vmax:
        im = plt.imshow(result['ref_frame_img'], cmap='bwr', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Velocity")
    else:
        im = plt.imshow(result['ref_frame_img'], cmap='bwr')
    plt.title(f"Reference Frame: {base_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_reference_frame.png"))
    plt.close()

    alignment_fig_path = os.path.join(output_dir, f"{base_name}_alignment.png")
    visualize_alignment_effect(
        result['original_mask'],
        result['aligned_mask'],
        result['ref_idx'],
        num_show=5,
        save_path=alignment_fig_path
    )

    # --- Visualization of Alignment (Animated GIF) ---
    gif_path = os.path.join(output_dir, f"{base_name}_alignment.gif")
    create_alignment_gif(
        mask_array_before=result['original_mask'],
        mask_array_after=result['aligned_mask'],
        shifts=None,  # Shifts are computed internally via COM difference
        save_path=gif_path
    )
    # --- Save aligned velocity numpy ---
    np.save(os.path.join(output_dir, f"{base_name}_aligned_velocity.npy"), result['aligned_velocity'])

def process_velocity_folder(folder_path, ref_method='similarity', patch_center=(96, 96), patch_size=5,
                            profile_axis='horizontal', line_index=None, template=None):
    from tqdm import tqdm
    all_results = []
    velocity_templates = []
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    for fname in tqdm(filenames, desc="Processing files"):
        path = os.path.join(folder_path, fname)
        velocity_data = np.load(path)
        binary_mask = extract_binary_masks(velocity_data)
        ref_idx = choose_reference_frame(binary_mask, method=ref_method, external_template=template)
        aligned_mask, shifts = align_masks_to_reference(binary_mask, ref_idx)
        aligned_velocity = align_velocity_data_to_reference(velocity_data, shifts)
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(
            aligned_velocity, mask_array=binary_mask, ref_idx=ref_idx, patch_size=5)
        spatiotemporal_profile = extract_spatiotemporal_profile(
            aligned_velocity, axis=profile_axis, line_index=line_index)
        result = {
            'filename': fname,
            'ref_idx': ref_idx,
            'aligned_velocity': aligned_velocity,
            'time_profile': time_profile,
            'variance_map': variance_map,
            'patch_mask': patch_mask,
            'spatiotemporal_profile': spatiotemporal_profile,
            'binary_mask': binary_mask,
            'ref_frame_img': aligned_velocity[ref_idx],
            'original_mask': binary_mask,
            'aligned_mask': aligned_mask
        }
        all_results.append(result)
        velocity_templates.append(aligned_velocity)
    return all_results, velocity_templates