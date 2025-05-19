

I'll merge these two scripts into a unified version when I have more time.

intra_BSpline_align_velocity_mask.py
intra_BSpline_align_velocity.py




def process_affine_folder(folder_path, output_path, use_velocity_for_BSpline=True, axis='horizontal', threshold=1e-6):
    os.makedirs(output_path, exist_ok=True)
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    for fname in tqdm(filenames, desc="Processing"):
        path = os.path.join(folder_path, fname)
        velocity_array = np.load(path)

        aligned_velocity, ref_idx = perform_intra_subject_alignment(
            velocity_array, use_velocity_for_BSpline=use_velocity_for_BSpline,
            ref_strategy='similarity', threshold=threshold
        )

        # Optionally, continue with diagnostics, visualization, etc.
        # You can plug in your existing visualizers and statistical summaries here.

        np.save(os.path.join(output_path, f"{os.path.splitext(fname)[0]}_aligned.npy"), aligned_velocity)

