'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import center_of_mass
import SimpleITK as sitk

# ----------------------- تنظیمات -----------------------
sample_path = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data\20213_2_0_masked.npy"
template_dir = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\templates\Template10allsimNonrigid"
output_dir = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\notebooks\analysis_output\15frame"
os.makedirs(output_dir, exist_ok=True)

frame_index = 15
template_prefix = "template_phase"
template_suffix = "_similarity_10.npy"
num_templates = 30

# ----------------------- توابع -----------------------
def compute_stats(mask):
    area = int(np.sum(mask))
    com_y, com_x = center_of_mass(mask) if area > 0 else (np.nan, np.nan)
    return area, com_y, com_x

def register_mask_to_template(mask, template):
    ref_img = sitk.GetImageFromArray(template.astype(np.float32))
    moving_img = sitk.GetImageFromArray(mask.astype(np.float32))

    transform = sitk.BSplineTransformInitializer(ref_img, [10, 10])
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsLBFGSB()
    registration.SetInitialTransform(transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)

    final_transform = registration.Execute(ref_img, moving_img)
    aligned = sitk.Resample(moving_img, ref_img, final_transform, sitk.sitkBSpline, 0.0)
    aligned = sitk.GetArrayFromImage(aligned) > 0.5
    return aligned.astype(np.uint8)

# ----------------------- بارگذاری فریم -----------------------
data = np.load(sample_path)
frame = (np.abs(data[frame_index]) > 1e-6).astype(np.uint8)

# ----------------------- ثبت و ذخیره همه تمپلیت‌ها -----------------------
rows = []
fig, axs = plt.subplots(num_templates, 3, figsize=(12, 1.2 * num_templates))

for i in range(num_templates):
    tpl_path = os.path.join(template_dir, f"{template_prefix}_{i:02d}{template_suffix}")
    if not os.path.exists(tpl_path):
        print(f"⛔ Skipping template {i:02d}, not found.")
        continue

    template = np.load(tpl_path)
    template = (template > 0.5).astype(np.uint8)
    aligned = register_mask_to_template(frame, template)

    # آمارها
    area_tpl, cy_tpl, cx_tpl = compute_stats(template)
    area_inp, cy_inp, cx_inp = compute_stats(frame)
    area_reg, cy_reg, cx_reg = compute_stats(aligned)

    rows.append({
        'Template_Frame': i,
        'Frame_Index': frame_index,
        'Area_template': area_tpl,
        'COM_template_y': cy_tpl,
        'COM_template_x': cx_tpl,
        'Area_input': area_inp,
        'COM_input_y': cy_inp,
        'COM_input_x': cx_inp,
        'Delta_A_input_template': area_inp - area_tpl,
        'Delta_COM_y_input_template': cy_inp - cy_tpl,
        'Delta_COM_x_input_template': cx_inp - cx_tpl,
        'Area_aligned': area_reg,
        'COM_aligned_y': cy_reg,
        'COM_aligned_x': cx_reg,
        'Delta_A_after': area_reg - area_tpl,
        'Delta_COM_y_after': cy_reg - cy_tpl,
        'Delta_COM_x_after': cx_reg - cx_tpl,
    })

    # رسم هر سطر: [Input | Template | Aligned]
    titles = [
        f"Input\nA={area_inp} ΔA={area_inp - area_tpl}\nΔCOM=({cx_inp - cx_tpl:.1f}, {cy_inp - cy_tpl:.1f})",
        f"Template\nA={area_tpl}",
        f"Aligned\nA={area_reg} ΔA={area_reg - area_tpl}\nΔCOM=({cx_reg - cx_tpl:.1f}, {cy_reg - cy_tpl:.1f})",
    ]
    images = [frame, template, aligned]
    for j in range(3):
        ax = axs[i, j] if num_templates > 1 else axs[j]
        ax.imshow(images[j], cmap='gray')
        ax.set_title(titles[j], fontsize=8)
        ax.axis('off')

# ذخیره تصویر نهایی
plt.tight_layout()
big_image_path = os.path.join(output_dir, f"frame{frame_index}_all_templates_comparison.png")
plt.savefig(big_image_path, dpi=200)
print(f"📸 Full comparison image saved to: {big_image_path}")
plt.show()

# ذخیره فایل CSV
df = pd.DataFrame(rows)
csv_path = os.path.join(output_dir, f"frame{frame_index}_vs_all_templates.csv")
df.to_csv(csv_path, index=False)
print(f"✅ CSV saved to: {csv_path}")

'''
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import center_of_mass
import SimpleITK as sitk

# ----------------------- تنظیمات -----------------------
sample_path = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data\20213_2_0_masked.npy"
template_dir = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\templates\Template10allsimNonrigid"
output_dir = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\notebooks\analysis_output\15frame"
os.makedirs(output_dir, exist_ok=True)

frame_index = 15
template_number = 4  # عدد تمپلیت که هر بار تغییرش میدی

template_prefix = "template_phase"
template_suffix = "_similarity_10.npy"

# ----------------------- توابع -----------------------
def compute_stats(mask):
    area = int(np.sum(mask))
    com_y, com_x = center_of_mass(mask) if area > 0 else (np.nan, np.nan)
    return area, com_y, com_x

def register_mask_to_template(mask, template):
    ref_img = sitk.GetImageFromArray(template.astype(np.float32))
    moving_img = sitk.GetImageFromArray(mask.astype(np.float32))

    transform = sitk.BSplineTransformInitializer(ref_img, [10, 10])
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsLBFGSB()
    registration.SetInitialTransform(transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)

    final_transform = registration.Execute(ref_img, moving_img)
    aligned = sitk.Resample(moving_img, ref_img, final_transform, sitk.sitkBSpline, 0.0)
    aligned = sitk.GetArrayFromImage(aligned) > 0.5
    return aligned.astype(np.uint8)

def plot_comparison(input_mask, template, aligned_mask, output_path, title,
                    stats_input, stats_template, stats_aligned):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # ورودی
    axs[0].imshow(input_mask, cmap='gray')
    axs[0].set_title(f"Input\nArea: {stats_input['area']}\n"
                     f"COM: ({stats_input['com_x']:.1f}, {stats_input['com_y']:.1f})\n"
                     f"ΔArea: {stats_input['area'] - stats_template['area']}\n"
                     f"ΔCOM: ({stats_input['com_x'] - stats_template['com_x']:.1f}, "
                     f"{stats_input['com_y'] - stats_template['com_y']:.1f})")

    # تمپلیت
    axs[1].imshow(template, cmap='gray')
    axs[1].set_title(f"Template\nArea: {stats_template['area']}\n"
                     f"COM: ({stats_template['com_x']:.1f}, {stats_template['com_y']:.1f})")

    # رجیستر شده
    axs[2].imshow(aligned_mask, cmap='gray')
    axs[2].set_title(f"Aligned\nArea: {stats_aligned['area']}\n"
                     f"COM: ({stats_aligned['com_x']:.1f}, {stats_aligned['com_y']:.1f})\n"
                     f"ΔArea: {stats_aligned['area'] - stats_template['area']}\n"
                     f"ΔCOM: ({stats_aligned['com_x'] - stats_template['com_x']:.1f}, "
                     f"{stats_aligned['com_y'] - stats_template['com_y']:.1f})")

    for ax in axs:
        ax.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📸 Image saved to: {output_path}")
    plt.show()

# ----------------------- اجرای کد برای یک تمپلیت -----------------------
data = np.load(sample_path)
frame = (np.abs(data[frame_index]) > 1e-6).astype(np.uint8)

tpl_path = os.path.join(template_dir, f"{template_prefix}_{template_number:02d}{template_suffix}")
if not os.path.exists(tpl_path):
    raise FileNotFoundError(f"⛔ Template not found: {tpl_path}")

template = np.load(tpl_path)
template = (template > 0.5).astype(np.uint8)

# محاسبه آمارها
area_tpl, cy_tpl, cx_tpl = compute_stats(template)
area_inp, cy_inp, cx_inp = compute_stats(frame)

aligned = register_mask_to_template(frame, template)
area_reg, cy_reg, cx_reg = compute_stats(aligned)

# ذخیره اطلاعات عددی در DataFrame
row = {
    'Template_Frame': template_number,
    'Frame_Index': frame_index,
    'Area_template': area_tpl,
    'COM_template_y': cy_tpl,
    'COM_template_x': cx_tpl,
    'Area_input': area_inp,
    'COM_input_y': cy_inp,
    'COM_input_x': cx_inp,
    'Delta_A_input_template': area_inp - area_tpl,
    'Delta_COM_y_input_template': cy_inp - cy_tpl,
    'Delta_COM_x_input_template': cx_inp - cx_tpl,
    'Area_aligned': area_reg,
    'COM_aligned_y': cy_reg,
    'COM_aligned_x': cx_reg,
    'Delta_A_after': area_reg - area_tpl,
    'Delta_COM_y_after': cy_reg - cy_tpl,
    'Delta_COM_x_after': cx_reg - cx_tpl,
}

df = pd.DataFrame([row])

# ساخت اسم فایل بر اساس ترکیب شماره فریم و تمپلیت
csv_filename = f"frame{frame_index}_template{template_number:02d}.csv"
img_filename = f"frame{frame_index}_template{template_number:02d}.png"
csv_path = os.path.join(output_dir, csv_filename)
img_path = os.path.join(output_dir, img_filename)

df.to_csv(csv_path, index=False)
print(f"✅ CSV saved to: {csv_path}")
# آماده‌سازی داده‌ها برای نمایش در تصویر
stats_input = {
    'area': area_inp,
    'com_x': cx_inp,
    'com_y': cy_inp,
}
stats_template = {
    'area': area_tpl,
    'com_x': cx_tpl,
    'com_y': cy_tpl,
}
stats_aligned = {
    'area': area_reg,
    'com_x': cx_reg,
    'com_y': cy_reg,
}

# فراخوانی تابع با آمارها
plot_comparison(frame, template, aligned, img_path,
                title=f"Frame {frame_index} vs Template {template_number:02d}",
                stats_input=stats_input,
                stats_template=stats_template,
                stats_aligned=stats_aligned)
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import center_of_mass
import SimpleITK as sitk

# ----------------------- تنظیمات -----------------------
sample_path = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\data\20213_2_0_masked.npy"
template_dir = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\templates\Template10allsimNonrigid"
output_dir = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\scripts\Registration\notebooks\analysis_output\15frame"

os.makedirs(output_dir, exist_ok=True)

frame_index =1
template_prefix = "template_phase"
template_suffix = "_similarity_10.npy"
num_templates = 30

def compute_stats(mask):
    area = int(np.sum(mask))
    com_y, com_x = center_of_mass(mask) if area > 0 else (np.nan, np.nan)
    return area, com_y, com_x

def register_mask_to_template(mask, template):
    ref_img = sitk.GetImageFromArray(template.astype(np.float32))
    moving_img = sitk.GetImageFromArray(mask.astype(np.float32))

    transform = sitk.BSplineTransformInitializer(ref_img, [10, 10])
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsLBFGSB()
    registration.SetInitialTransform(transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)

    final_transform = registration.Execute(ref_img, moving_img)
    aligned = sitk.Resample(moving_img, ref_img, final_transform, sitk.sitkBSpline, 0.0)
    aligned = sitk.GetArrayFromImage(aligned) > 0.5
    return aligned.astype(np.uint8)

# Load target frame
data = np.load(sample_path)
frame = (np.abs(data[frame_index]) > 1e-6).astype(np.uint8)

rows = []
fig, axs = plt.subplots(num_templates, 3, figsize=(12, 1.2 * num_templates))

for i in range(num_templates):
    tpl_path = os.path.join(template_dir, f"{template_prefix}_{i:02d}{template_suffix}")
    if not os.path.exists(tpl_path):
        print(f"Template {i:02d} not found. Skipped.")
        continue

    template = np.load(tpl_path)
    template = (template > 0.5).astype(np.uint8)
    aligned = register_mask_to_template(frame, template)

    area_t, com_ty, com_tx = compute_stats(template)
    area_i, com_iy, com_ix = compute_stats(frame)
    area_a, com_ay, com_ax = compute_stats(aligned)

    row = {
        'Template_Frame': i,
        'Frame_Index': frame_index,
        'Area_template': area_t,
        'COM_template_y': com_ty, 'COM_template_x': com_tx,
        'Area_input': area_i,
        'COM_input_y': com_iy, 'COM_input_x': com_ix,
        'Delta_A': int(area_i - area_t),
        'Delta_COM_y': float(com_iy - com_ty),
        'Delta_COM_x': float(com_ix - com_tx),
        'Area_aligned': area_a,
        'COM_aligned_y': com_ay, 'COM_aligned_x': com_ax,
        'Delta_A_after': int(area_a - area_t),
        'Delta_COM_y_after': float(com_ay - com_ty),
        'Delta_COM_x_after': float(com_ax - com_tx),
        'Delta_A_input_aligned': int(area_a - area_i),
        'Delta_COM_y_input_aligned': float(com_ay - com_iy),
        'Delta_COM_x_input_aligned': float(com_ax - com_ix),
    }
    rows.append(row)

    titles = [
        f"Input\nA={area_i} ΔA={area_i - area_t}\nΔCOM=({com_ix - com_tx:.1f}, {com_iy - com_ty:.1f})",
        f"Template\nA={area_t}",
        f"Aligned\nA={area_a} ΔA={area_a - area_t}\nΔCOM=({com_ax - com_tx:.1f}, {com_ay - com_ty:.1f})\nΔA_inp={area_a - area_i}, ΔCOM_inp=({com_ax - com_ix:.1f}, {com_ay - com_iy:.1f})"
    ]
    images = [frame, template, aligned]
    for j in range(3):
        ax = axs[i, j] if num_templates > 1 else axs[j]
        ax.imshow(images[j], cmap='gray')
        ax.set_title(titles[j], fontsize=7)
        ax.axis('off')

plt.tight_layout()
image_path = os.path.join(output_dir, f"frame{frame_index:02d}_all_templates_comparison.png")
plt.savefig(image_path, dpi=200)
print(f"\nSaved image to: {image_path}")
plt.show()

df = pd.DataFrame(rows)
csv_path = os.path.join(output_dir, f"frame{frame_index:02d}_vs_all_templates.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("\nFull DataFrame:\n")
    print(df)