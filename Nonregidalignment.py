# This code introduces two main approaches for processing velocity files and performing deformable (non-rigid) alignment
# Visualizing the results is one of the main goals:
# 1. Single Template Generation by one sample:
#   - If neither external_template_path nor external_template_dir is provided (i.e., both are None), 
#       the code will use a single sample from the input to generate a template automatically. 
#    - This approach generates a template from a single reference sample and registers all other samples to it.
#    - The alignment is done on a single reference mask (frame) in the velocity data, and the results are saved and visualized.
#    - This approach is mainly used when only a single reference sample/template is needed for deformable alignment.
# 2. Single External Template::
#    - If a file path to a single external template (external_template_path) is provided,
#        it will be used as the reference template for aligning all samples.
#    - `external_template_path` is used when you are using a **single external template** for alignment. 
#   This argument is a **file path** to a single template image that will be used as the reference for the alignment of all other frames.
#   Example:
#     external_template_path = "/path/to/single_template.npy"  
# 3. Multi-Template Generation:
#    - If a path to a directory containing multiple template frames (`external_template_dir`) is provided, 
#      the code will align the samples based on a set of templates.
#    - The code processes each velocity sample with the corresponding templates in the directory. 
#      The template files are named according to a specified prefix (`template_prefix`).
#    - The `external_template_dir` is used to load and apply multiple templates to align the samples, 
#      and the alignment is done using all frames in the velocity data.
#    - This approach is useful when the velocity data requires aligning to different phase-specific templates.
#    - If `external_template_path` is set to `None`, the code will either:
#     a) Generate a reference frame based on the input data sample itself (when `external_template_dir` is also `None`).
#     b) Or, it uses the provided multi-template directory (external_template_dir) to align each frame to its corresponding phase-specific template.

# # - This code is designed to work after the output of `aortic_velocity_template_builder.py`, as it processes the velocity 
#   data (and associated masks) into aligned and registered results.

# 4. **Test Folder for Alignment**:
#    - This script uses a "test" folder for alignment, where the input velocity data is processed.
#    - It generates registered (aligned) images based on either a single sample or one external template or a set of external templates.

# 5. **New Functionality**:
#    - The functions `deformable_register_all` and `deformable_align_masks_to_template` were modified to handle the multi-template scenario,
#        where each frame can be aligned to a corresponding template from the directory.


# ----------------------- Template Selection Logic -----------------------
# Based on the combination of arguments and the value of `ref_method`,
# the alignment method is selected as follows:

# Scenario 1: Auto-generate template from the input data (no external template)
#   external_template_path = None
#   external_template_dir = None
#   ref_method = 'similarity' or 'area'
#   → A single reference frame is selected from the input sample itself.

# Scenario 2: Use a single shared external template for all frames
#   external_template_path = "path/to/template.npy"
#   external_template_dir = None
#   ref_method = 'template'
#   → All frames are aligned to the same reference template.

# Scenario 3: Use multiple phase-specific external templates (one per frame)
#   external_template_path = None
#   external_template_dir = "path/to/template_folder"
#   ref_method = 'template'
#   → Each frame is aligned using its corresponding template from the folder
#     (template filenames should follow the pattern defined by `template_prefix`)
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from skimage.metrics import structural_similarity as ssim

# -------------------- Utility Functions --------------------
def extract_binary_masks(masked_velocity_array, threshold=1e-6):
    return (np.abs(masked_velocity_array) > threshold).astype(np.uint8)

def choose_reference_frame(mask_array, method='similarity', external_template=None):
    num_frames = mask_array.shape[0]
    areas = mask_array.sum(axis=(1, 2))
    if method == 'area':
        mean_area = np.mean(areas)
        return np.argmin(np.abs(areas - mean_area))
    elif method == 'similarity':
        total_scores = [np.mean([ssim(mask_array[i], mask_array[j]) for j in range(num_frames) if i != j]) for i in range(num_frames)]
        return np.argmax(total_scores)
    elif method == 'template':
        if external_template is None:
            raise ValueError("Template image required for template-based reference selection")
        scores = [ssim(mask_array[i], external_template) for i in range(num_frames)]
        return np.argmax(scores)
    else:
        raise ValueError("Invalid reference selection method")

# -------------------- Deformable Registration --------------------
def deformable_align_masks_to_template(mask_array, external_template):
    T, H, W = mask_array.shape
    ref_img = sitk.GetImageFromArray(external_template.astype(np.float32))
    aligned_masks, transforms = [], []
    for t in range(T):
        moving = sitk.GetImageFromArray(mask_array[t].astype(np.float32))
        transform = sitk.BSplineTransformInitializer(ref_img, [10, 10])
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsLBFGSB()
        registration.SetInitialTransform(transform, inPlace=False)
        registration.SetInterpolator(sitk.sitkLinear)            #sitkNearestNeighbor        or      sitkLinear 
        final_transform = registration.Execute(ref_img, moving)
        aligned = sitk.Resample(moving, ref_img, final_transform, sitk.sitkBSpline, 0.0)    #sitkBSpline  sitkLinear
        aligned = sitk.GetArrayFromImage(aligned) > 0.5                    # 0.5
        aligned_masks.append(aligned.astype(np.uint8))
        transforms.append(final_transform)
    return np.stack(aligned_masks), transforms

def deformable_align_velocity(velocity_array, transforms, ref_shape):
    aligned_velocity = []
    for v, tf in zip(velocity_array, transforms):
        v_img = sitk.GetImageFromArray(v.astype(np.float32))
        ref_img = sitk.GetImageFromArray(np.zeros(ref_shape, dtype=np.float32))
        aligned = sitk.Resample(v_img, ref_img, tf, sitk.sitkLinear, 0.0)
        aligned_velocity.append(sitk.GetArrayFromImage(aligned))
    return np.stack(aligned_velocity)

# -------------------- Feature Extraction --------------------
def extract_time_profile_and_variance(velocity_array, mask_array, ref_idx, patch_size=5):
    cy, cx = center_of_mass(mask_array[ref_idx])
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

        # Compute area difference (number of pixels in mask)
        area_before = np.sum(mask_array_before[i])
        area_after = np.sum(mask_array_after[i])
        delta_area = abs(int(area_after) - int(area_before))

        # Debug: Add this line here:
        print(f"Frame {i}: area_before = {area_before}, area_after = {area_after}, ΔA = {delta_area}")


        ax.arrow(com_before[1], com_before[0], dx, dy, color='cyan',
                 head_width=2, head_length=2, length_includes_head=True)

        # نمایش مقادیر
        ax.set_title(f"Frame {i} | dy={dy:+.1f}, dx={dx:+.1f}, ΔA={delta_area:+}", fontsize=10)
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



# -------------------- Visualization & Saving --------------------
def save_visual_outputs_deformable(result, output_dir):
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
        save_path=gif_path
    )

    # Save aligned velocity array
    np.save(os.path.join(output_dir, f"{base}_aligned_velocity.npy"), result['aligned_velocity'])

def visualize_deformable_alignment(mask_array_before, mask_array_after, ref_idx, output_path=None, base_name="alignment", num_show=5):
    indices = list(range(max(0, ref_idx - num_show), min(mask_array_before.shape[0], ref_idx + num_show + 1)))
    ref_com = center_of_mass(mask_array_before[ref_idx])
    fig, axs = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    fig.suptitle(f"Deformable Alignment (Ref Frame: {ref_idx})")
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
        axs[1, i].text(2, 10, f"dy={dy:+.1f}\ndx={dx:+.1f}\nΔA={delta_area:+.0f}",
                      color='yellow', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, f"{base_name}_deformable_alignment.png"), dpi=150)
        plt.close()
    else:
        plt.show()

# -------------------- Main Processing --------------------
def process_deformable_folder(folder_path, output_path, ref_method='similarity', axis='horizontal',
                              external_template_path=None, external_template_dir=None, template_prefix="template_phase"):
    from tqdm import tqdm
    '''
    # Load external template if needed
    external_template = None
    if ref_method == 'template':
        if external_template_path is None or not os.path.isfile(external_template_path):
            raise ValueError("You selected 'template' method but external_template_path is missing or invalid.")
        # Load template and convert to binary if needed
        external_template = np.load(external_template_path)
        if external_template.max() <= 1.0:
            # Probably a soft mask (probability map), apply threshold
            external_template = (external_template > 0.5).astype(np.uint8)
        else:
            # Already binary (e.g., values are 0 or 1)
            external_template = external_template.astype(np.uint8)

    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    results = []
    
    for fname in tqdm(filenames):
        path = os.path.join(folder_path, fname)
        vel = np.load(path)
        bin_mask = extract_binary_masks(vel)

        if ref_method == 'template' and external_template is not None:
            ref_idx = 0  # Dummy, for consistency only
            aligned_mask, transforms = deformable_align_masks_to_template(bin_mask, external_template)
        else:
            ref_idx = choose_reference_frame(bin_mask, method=ref_method, external_template=external_template)
            aligned_mask, transforms = deformable_align_masks_to_template(bin_mask, bin_mask[ref_idx])

        aligned_vel = deformable_align_velocity(vel, transforms, ref_shape=aligned_mask[0].shape)
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(aligned_vel, aligned_mask, ref_idx)
        spatiotemporal_profile = extract_spatiotemporal_profile(aligned_vel, axis=axis)
    '''
    external_template = None
    external_templates_dict = {}

    external_template = None
    external_templates_dict = {}

    if ref_method == 'template':
        if external_template_dir:
            for i in range(30):
                tpl_path = os.path.join(external_template_dir, f"{template_prefix}_{i:02d}_similarity_10.npy")
                if os.path.exists(tpl_path):
                    tpl = np.load(tpl_path)
                    tpl = (tpl > 0.5).astype(np.uint8) if tpl.max() <= 1.0 else tpl.astype(np.uint8)
                    external_templates_dict[i] = tpl
            if len(external_templates_dict) < 30:
                raise ValueError("Missing some template frames in folder.")
        elif external_template_path:
            external_template = np.load(external_template_path)
            external_template = (external_template > 0.5).astype(np.uint8) if external_template.max() <= 1.0 else external_template.astype(np.uint8)
        else:
            raise ValueError("You selected 'template' method but provided no valid template path or directory.")

    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    results = []
    for fname in tqdm(filenames):
        path = os.path.join(folder_path, fname)
        vel = np.load(path)
        bin_mask = extract_binary_masks(vel)

        if ref_method == 'template' and external_templates_dict:
            aligned_mask_list, transforms_list = [], []
            for i in range(bin_mask.shape[0]):
                tpl = external_templates_dict.get(i)
                if tpl is None:
                    raise ValueError(f"Missing template for frame {i:02d}")
                aligned_i, tf_i = deformable_align_masks_to_template(np.expand_dims(bin_mask[i], axis=0), tpl)
                aligned_mask_list.append(aligned_i[0])
                transforms_list.append(tf_i[0])
            aligned_mask = np.stack(aligned_mask_list)
            transforms = transforms_list
            ref_idx = 0  # dummy for naming
        elif ref_method == 'template':
            ref_idx = 0
            aligned_mask, transforms = deformable_align_masks_to_template(bin_mask, external_template)
        else:
            ref_idx = choose_reference_frame(bin_mask, method=ref_method, external_template=external_template)
            aligned_mask, transforms = deformable_align_masks_to_template(bin_mask, bin_mask[ref_idx])

        aligned_vel = deformable_align_velocity(vel, transforms, ref_shape=aligned_mask[0].shape)
        time_profile, variance_map, patch_mask = extract_time_profile_and_variance(aligned_vel, aligned_mask, ref_idx)
        spatiotemporal_profile = extract_spatiotemporal_profile(aligned_vel, axis=axis)
        result = {
            'filename': fname,
            'aligned_velocity': aligned_vel,
            'ref_idx': ref_idx,
            'ref_frame_img': aligned_vel[ref_idx],
            'time_profile': time_profile,
            'variance_map': variance_map,
            'patch_mask': patch_mask,
            'spatiotemporal_profile': spatiotemporal_profile,
            'binary_mask': bin_mask,
            'original_mask': bin_mask,
            'aligned_mask': aligned_mask
        }

        results.append(result)
        save_visual_outputs_deformable(result, output_path)
        visualize_deformable_alignment(
            mask_array_before=result['original_mask'],
            mask_array_after=result['aligned_mask'],
            ref_idx=result['ref_idx'],
            output_path=output_path,
            base_name=os.path.splitext(fname)[0]
        )
    
    return results, None


# -------------------- Run Entry Point --------------------
if __name__ == "__main__":
    # Define input/output directories
    input_folder = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data"
    output_folder = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\1simNonrigid"
    # ------------------------ Template Options ------------------------
    # You must set ONLY ONE of the following:
    # 1) external_template_path : use a single template for all frames
    # 2) external_template_dir  : use one template per frame (multi-template)
    # 3) leave both as None     : auto-generate template from input sample

    # --- OPTION 1: Use a single shared template for all frames ---
    external_template_path = None  # # None or set a path to a single .npy file
    # Example: r"...\final_template_similarity_1000.npy"

    # --- OPTION 2: Use a separate template for each frame (multi-template alignment) ---
    external_template_dir = None#r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\output\Alignment\Template10allsimNonrigid"
    # None or set a path to a folder with multiple templates (e.g., template_phase_00.npy, etc.)
    template_prefix = "template_phase"  #Used only if external_template_dir is provided, Expected filenames: template_phase_00_similarity_10.npy, etc.
    
    # Choose alignment method: 'similarity', 'area', or 'template'
    ref_method = 'similarity'  # Use 'template' if external_template_path or external_template_dir is provided

    # ------------------------ Alignment Settings ------------------------
    # ref_method: 
    # 'template' → use external_template_path or external_template_dir
    # 'similarity' or 'area' → auto-select reference frame from the input
    ref_method = 'similarity'

    # Choose axis for spatiotemporal profile ('horizontal', 'vertical', 'full_x', 'full_y')
    axis = 'horizontal'
    # Run the alignment process
    process_deformable_folder(
        folder_path=input_folder,
        output_path=output_folder,
        ref_method=ref_method,                 
        axis=axis,  
        external_template_path=external_template_path,           
        external_template_dir=external_template_dir,  
        template_prefix=template_prefix
    )
