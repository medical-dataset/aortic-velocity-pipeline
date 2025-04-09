"""
# ----------------------------------------------
# Template Builder for Aortic Velocity Masked Data
# ----------------------------------------------
# This script processes masked velocity data, extracts binary masks, 
# and generates a static template using a set of registered frames.
# It includes various methods for frame selection, deformable image registration,
# and template construction (mean, PCA, or probabilistic methods).
# The script supports different strategies for selecting frames and phases
# from the dataset and allows batch processing for multiple templates.
## Users can select the number of samples from the input folder to generate a template.
# Input:  Folder containing .npy files with velocity masks
# Output: A set of template images and corresponding transformation files
The script includes the following steps:
1. **Mask Extraction**: It extracts binary masks from masked velocity arrays based on a threshold.
2. **Frame Selection**: It selects a frame from the stack of frames using methods like 'area' or 'similarity' based on the mask data.
3. **Template Frame Selection**: It identifies the most representative frame from the stack to use as a template.
4. **Deformable Registration**: It performs deformable registration of frames using SimpleITK to align frames based on a reference.
5. **Template Construction**: Constructs a template using methods like 'mean', 'probabilistic', or 'PCA'.
6. **Multi-phase Selection**: It allows for the selection of specific phases from the frames.

Functions included:
- `extract_binary_masks`: Converts velocity data to binary masks.
- `select_frame`: Selects a frame based on the chosen method.
- `choose_template_frame`: Chooses a reference frame to be used as a template.
- `deformable_register_all`: Performs deformable image registration on the frames.
- `build_template`: Constructs a template based on the chosen method.
- `get_phase_frames`: Extracts specific phases from the stack of frames.
- `save_template`: Saves the generated template image and its numpy array.

# - Optional binarization: The generated template can be optionally thresholded to a binary mask using `binarize_output=True`.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA

# ---------------------- Mask Extraction ----------------------
def extract_binary_masks(masked_velocity_array, threshold=1e-6):
    return (np.abs(masked_velocity_array) > threshold).astype(np.uint8)

# ---------------------- Frame Selection ----------------------
def select_frame(mask_array, method='similarity'):
    num_frames = mask_array.shape[0]
    if method == 'area':
        areas = mask_array.sum(axis=(1, 2))
        return np.argmin(np.abs(areas - areas.mean()))
    elif method == 'similarity':
        scores = [
            np.mean([ssim(mask_array[i], mask_array[j]) for j in range(num_frames) if i != j])
            for i in range(num_frames)
        ]
        return np.argmax(scores)
    else:
        raise ValueError("Unknown frame selection method")

# ---------------------- Template Frame Selection ----------------------
def choose_template_frame(frames_stack, method='similarity'):
    if len(frames_stack) == 1:
        return 0
    if method == 'similarity':
        scores = [
            np.mean([ssim(frames_stack[i], frames_stack[j]) for j in range(len(frames_stack)) if i != j])
            for i in range(len(frames_stack))
        ]
        return np.argmax(scores)
    elif method == 'area':
        areas = np.sum(frames_stack, axis=(1, 2))
        return np.argmin(np.abs(areas - np.mean(areas)))
    else:
        raise ValueError("Invalid method for choosing template")

# ---------------------- Deformable Registration ----------------------
def deformable_register_all(frames, reference):
    ref_img = sitk.GetImageFromArray(reference.astype(np.float32))
    registered = []
    transforms = []
    for frame in frames:
        mov_img = sitk.GetImageFromArray(frame.astype(np.float32))
        tf_init = sitk.BSplineTransformInitializer(ref_img, [10, 10])
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsLBFGSB()
        registration.SetInitialTransform(tf_init, inPlace=False)
        registration.SetInterpolator(sitk.sitkLinear)
        tf_final = registration.Execute(ref_img, mov_img)
        warped = sitk.Resample(mov_img, ref_img, tf_final, sitk.sitkLinear, 0.0)
        registered.append(sitk.GetArrayFromImage(warped))
        transforms.append(tf_final)
    return np.stack(registered), transforms

# ---------------------- Template Construction ----------------------
def build_template(registered_stack, method='mean', binarize=False):
    """
    Constructs a template from a stack of aligned frames.

    Parameters:
        registered_stack (np.ndarray): Stack of aligned binary masks.
        method (str): Template construction method - 'mean', 'probabilistic', or 'pca'.
        binarize (bool): If True, apply thresholding (0.5) to produce binary mask output.

    Returns:
        np.ndarray: The constructed template (binary or soft).
    """
    if method == 'mean':
        template = np.mean(registered_stack, axis=0)
    elif method == 'probabilistic':
        template = np.mean(registered_stack > 0.5, axis=0)
    elif method == 'pca':
        flat = registered_stack.reshape(registered_stack.shape[0], -1)
        pca = PCA(n_components=1)
        comp1 = pca.fit_transform(flat)
        template = pca.components_[0].reshape(registered_stack.shape[1:])
    else:
        raise ValueError("Unknown template building method")
    # Optional binarization to convert soft mask to binary mask
    if binarize:
        template = (template > 0.5).astype(np.uint8)

    return template

# ---------------------- Multi-phase Selector ----------------------
def get_phase_frames(mask_array, strategy='static', phases=[15]):
    if strategy == 'static':
        return [mask_array[phases[0]]]
    elif strategy == 'multi':
        return [mask_array[p] for p in phases]
    elif strategy == 'all':
        return [mask_array[i] for i in range(mask_array.shape[0])]
    else:
        raise ValueError("Invalid phase selection strategy")
#=====================
def save_template(template, output_path, name, binarize_for_visual=False):
    np.save(os.path.join(output_path, f"{name}.npy"), template)
    # Optional visualization as binary
    if binarize_for_visual:
        vis_template = (template > 0.5).astype(np.uint8)
    else:
        vis_template = template
    plt.imshow(vis_template, cmap='gray')
    plt.title(f"Template - {name}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()

# ---------------------- Main Pipeline ----------------------
'''
def build_static_template_from_folder(folder_path,
                                      num_samples=10,
                                      frame_selection_method='similarity',
                                      template_build_method='mean',
                                      phase_strategy='static',
                                      phase_frames=[15],    
                                      output_path="template_output"):
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    selected_frames = []
    all_ids = []
    os.makedirs(output_path, exist_ok=True)
    filenames = filenames[:num_samples]
    selected_masks = []
    for fname in filenames:
        full_path = os.path.join(folder_path, fname)
        data = np.load(full_path)
        masks = extract_binary_masks(data)

        if phase_strategy == 'static':
            idx = select_frame(masks, method=frame_selection_method)
            selected_frames.append(masks[idx])
        elif phase_strategy == 'multi':
            selected_frames.extend(get_phase_frames(masks, 'multi', phase_frames))
        elif phase_strategy == 'all':
            selected_frames.extend(get_phase_frames(masks, 'all'))

        all_ids.append(fname)

    selected_frames = np.array(selected_frames)
    ref_idx = choose_template_frame(selected_frames, method='similarity')
    ref_frame = selected_frames[ref_idx]

    registered, transforms = deformable_register_all(selected_frames, ref_frame)
    template = build_template(registered, method=template_build_method)

    save_template(template, output_path, name=f"final_template_{selection_method}_{num_samples}")
    return template, registered, transforms, filenames
'''
def build_static_template_from_folder(folder_path,
                                      num_samples=10,
                                      frame_selection_method='similarity',
                                      template_build_method='mean',
                                      phase_strategy='static',
                                      phase_frames=[15],
                                      output_path="template_output",
                                      per_phase_template=False,
                                      binarize_output=False,
                                      ref_strategy='similarity'):
    os.makedirs(output_path, exist_ok=True)
    if per_phase_template and phase_strategy == 'static':
        print("Warning: per_phase_template=True has no effect with phase_strategy='static'. Ignoring per_phase_template.")
        per_phase_template = False

    # Step 1: Load selected .npy files and extract binary masks
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')][:num_samples]
    if len(filenames) < num_samples:
        print(f"Warning: only {len(filenames)} file(s) found in '{folder_path}', but num_samples was set to {num_samples}. Proceeding with available files.")
    
    all_masks = []
    for fname in filenames:
        full_path = os.path.join(folder_path, fname)
        data = np.load(full_path)
        masks = extract_binary_masks(data) # shape: (30, H, W)
        all_masks.append(masks)
    all_masks = np.array(all_masks)  # shape: (N, 30, H, W)

    # Step 2: Per-phase template generation (only for 'multi' or 'all' strategies)
    if per_phase_template and phase_strategy in ['multi', 'all']:
        for phase_idx in range(all_masks.shape[1]):  # loop over 30 phases
            phase_stack = all_masks[:, phase_idx, :, :]  # (N, H, W)
            
            # Skip empty phases
            if np.sum(phase_stack) == 0:
                print(f"Skipping phase {phase_idx}: all-zero masks.")
                continue
            
            # Choose reference frame and register
            ref_idx = choose_template_frame(phase_stack, method=ref_strategy)
            reference = phase_stack[ref_idx]
            registered, transforms = deformable_register_all(phase_stack, reference)
            
            # Build and save template
            template = build_template(registered, method=template_build_method, binarize=binarize_output)
            save_template(template, output_path, name=f"template_phase_{phase_idx:02d}_{frame_selection_method}_{num_samples}", binarize_for_visual=True)

        return None, None, None, filenames  # skipping batch return for 30 outputs

    # Step 3: Legacy mode – build one single template
    selected_frames = []
    for masks in all_masks:
        if phase_strategy == 'static':
            idx = select_frame(masks, method=frame_selection_method)
            selected_frames.append(masks[idx])
        elif phase_strategy == 'multi':
            selected_frames.extend(get_phase_frames(masks, 'multi', phase_frames))
        elif phase_strategy == 'all':
            selected_frames.extend(get_phase_frames(masks, 'all'))

    selected_frames = np.array(selected_frames)
    
    # Choose global reference frame
    ref_idx = choose_template_frame(selected_frames, method=ref_strategy)
    ref_frame = selected_frames[ref_idx]
    
    # Register all frames to reference
    registered, transforms = deformable_register_all(selected_frames, ref_frame)
    
    # Build final template
    template = build_template(registered, method=template_build_method, binarize=binarize_output)
    
    # Optional: force binary output
    template = (template > 0.5).astype(np.uint8)  # Add this if you want binary output

    save_template(template, output_path, name=f"final_template_{frame_selection_method}_{num_samples}")
    return template, registered, transforms, filenames


# Template Builder for Aortic Velocity Masked Data
# --------------------------------------------------
# This script loads masked velocity data and constructs one or more static templates
# using deformable registration and various selection/aggregation strategies.

if __name__ == '__main__':
    # ------------------------------ Template Construction Modes ------------------------------
    # This table summarizes how the combination of `phase_strategy` and `per_phase_template`
    # affects the template building process.

    # ┌────────────────────────────┬──────────────────────────────────────────────┬──────────────────────────────┐
    # │        Configuration       │         Frames Used for Template             │       Templates Created       │
    # ├────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────┤
    # │ phase_strategy = 'static'  │ 1 representative frame per sample            │ 1 template                    │
    # │ per_phase_template = False │                                              │ (default for this mode)       │
    # ├────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────┤
    # │ phase_strategy = 'static'  │ 1 representative frame per sample            │ 1 template                    │
    # │ per_phase_template = True  │                                              │ ( has no effect)             │
    # ├────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────┤
    # │ phase_strategy = 'all'     │ All 30 frames from each sample               │ 1 template                    │
    # │ per_phase_template = False │                                              │                              │
    # ├────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────┤
    # │ phase_strategy = 'all'     │ All 30 frames from each sample               │ 30 templates (1 per frame)    │
    # │ per_phase_template = True  │                                              │                              │
    # ├────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────┤
    # │ phase_strategy = 'multi'   │ Selected frame indices (e.g., 5, 10, 15)     │ 1 template                    │
    # │ per_phase_template = False │                                              │                              │
    # ├────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────┤
    # │ phase_strategy = 'multi'   │ Selected frame indices (e.g., 5, 10, 15)     │ N templates (1 per selected frame) │
    # │ per_phase_template = True  │                                              │                              │
    # └────────────────────────────┴──────────────────────────────────────────────┴──────────────────────────────┘

    # Notes:
    # - All modes perform deformable registration before template construction.
    # - per_phase_template=True means a separate template is built for each selected frame index (phase).
    # - per_phase_template has no effect when using phase_strategy='static'.

    # Input folder containing .npy velocity files (each with 30 frames)
    input_folder = r"\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\output\results"
    #\\isd_netapp\cardiac$\Majid\deepflow\deepFlowDocker\output\results
    #P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data
    # Folder to save the generated templates and their PNG previews
    output_path = r"P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\templates\Template10allsimNonrigid"
    #P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\templates\Template1simNonrigid
    # P:\Projects\DeepFlow\deepFlowDocker\scripts\Registration\output\Template10allsimNonrigid
    # Number of samples (velocity sequences) to use for building the template
    num_samples = 10
    # Strategy for selecting frames from each sample:
    # - 'static': Select exactly one representative frame from each sample (automatically chosen based on similarity or area).
    #             Used when building a single template (not per-frame templates).
    # - 'multi':  Select a custom list of specific frame indices (e.g., [5, 10, 15]) from each sample.
    # - 'all':    Use all frames (typically 30) from each sample.
    phase_strategy = 'all'

    # Method for selecting the most representative frame from each sample:
    # - 'similarity': highest average SSIM with all other frames
    # - 'area': closest to mean mask area
    selection_method = 'similarity'

    # How to pick the global reference frame (from among selected frames)
    # - 'similarity': most representative across all selected frames
    # - 'area': closest to average area
    ref_strategy = 'similarity'

    # After all selected 2D frames are deformably aligned, this specifies how to merge them into a single static template.
    # Method used to build the final template from multiple registered frames:
    # This setting only matters when building a template from more than one frame,
    # such as when combining data from multiple samples or multiple phases.
    # It is ignored when using a single input sample and a single frame.
    # - 'mean': average of the registered frames
    # - 'pca': principal component analysis (first component)
    # - 'probabilistic': proportion of foreground (active) pixels across frames
    template_mode = 'mean'

    # Only used when phase_strategy is 'multi': list of frame indices to include
    selected_phases = [5, 10, 15, 20, 25]
    
    # Whether to generate 30 separate templates (one per phase)
    per_phase_template = True   # Options: True, False    True = one template per frame index   False = one overall template
    
    # Binarize the final template? Applies threshold of 0.5 to soft mask
    binarize_output = True  # Set to False to keep soft probability mask

    build_static_template_from_folder(
        folder_path=input_folder,
        output_path=output_path,
        num_samples=num_samples,
        frame_selection_method=selection_method,
        template_build_method=template_mode,
        phase_strategy=phase_strategy,             
        phase_frames=selected_phases,    
        per_phase_template=per_phase_template,
        binarize_output=binarize_output,
        ref_strategy=ref_strategy         
    )