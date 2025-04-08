#  Aortic Velocity Registration & Template Pipeline

This repository provides a modular pipeline for processing, registering, and analyzing masked aortic velocity `.npy` data using both **rigid** and **non-rigid (deformable)** registration techniques.

> Developed for cardiac flow studies using velocity-encoded MRI data.

---

##  Repository Structure

| File | Description |
|------|-------------|
| `aorta_velocity_pipeline.py` | Full pipeline for **rigid registration** using Center of Mass (COM). Includes feature extraction and visualization. |
| `aortic_velocity_template_builder.py` | Builds **static templates** from masked velocity data using **deformable registration** (SimpleITK). |
| `Nonregidalignment.py` | Performs **non-rigid alignment** using either a single template or per-frame templates. Saves profiles, maps, and overlays. |
| `run_analysis.py` | A runnable script for batch rigid analysis using the above pipeline. |

---

## ⚙️ Features

-  Rigid (COM-based) alignment
-  Deformable (BSpline) registration with SimpleITK
-  Static template generation from velocity masks
-  Multi-template and single-template support
-  Visualization: time profiles, spatiotemporal maps, variance maps
-  Export of aligned velocities and animated GIFs

---

##  Installation

Requires Python 3.8+

```bash
git clone https://github.com/yourusername/aortic-velocity-registration.git
cd aortic-velocity-registration
pip install -r requirements.txt
```

**Dependencies** (if not using `requirements.txt`):

```bash
pip install numpy matplotlib seaborn scikit-image SimpleITK scikit-learn imageio tqdm
```

---

##  Usage

### 1. Run Rigid Alignment

```bash
python run_analysis.py
```

- Uses: `aorta_velocity_pipeline.py`
- Input: Folder with `.npy` velocity files
- Output: Aligned velocity + visualizations

---

### 2. Build a Template (Deformable)

```bash
python aortic_velocity_template_builder.py
```

- Output: One or more `.npy` template masks
- Mode: Per-phase template (30), or global
- Options: `mean`, `pca`, `probabilistic`

---

### 3. Run Non-Rigid Deformable Alignment

```bash
python Nonregidalignment.py
```

**Supported modes:**

| Template Type        | Required Input                      |
|----------------------|--------------------------------------|
| Auto (from data)     | No external input                   |
| Single template      | `external_template_path`            |
| Multi-template (30)  | `external_template_dir` + prefix    |

---

##  Input Format

- Each `.npy` file should contain a 3D NumPy array of shape:
  ```
  (30, height, width)
  ```
- Values represent **velocity magnitudes**, already masked (non-zero inside aorta).

---

##  Output

Each processed sample generates:

```
output_folder/
├── *_time_profile.png
├── *_variance_map.png
├── *_velocity_patch.png
├── *_reference_frame.png
├── *_alignment.gif
├── *_spatiotemporal.png
├── *_variance_zoom_*.png
└── *_aligned_velocity.npy
```

---

##  Customization

-  Switch reference frame method: `'area'`, `'similarity'`, `'template'`
-  Choose spatiotemporal profile axis: `'horizontal'`, `'vertical'`, `'full_x'`, `'full_y'`
-  Change patch size or frame index in core functions
-  Easily extendable to magnitude/phase images or different velocity encodings

---

##  Citation

> If you use this work in your research, please cite:
> _"Aortic Velocity Template-Based Registration Framework for 4D Flow MRI", Vafaeezadeh et al., 2025._

---

##  Future Plans

- [ ] Add CLI interface with `argparse`
- [ ] Export numeric metrics as `.csv`
- [ ] Jupyter notebook with evaluation dashboard
- [ ] Docker integration for reproducibility

---

##  Acknowledgments

Developed by [Majid Vafaeezadeh](https://github.com/yourusername) for the DeepFlow project at Herat Cardiac Research Lab 🫀

---

## 📄 License

MIT License
