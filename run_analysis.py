#Rigid
from aorta_velocity_pipeline import process_velocity_folder, save_analysis_outputs
from aorta_velocity_pipeline import visualize_alignment_effect
import os

#  path of .npy
input_folder = r"\\isd_netapp\mvafaeez$\Projects\DeepFlow\deepFlowDocker\output\testalign"
output_folder = r"P:\Projects\DeepFlow\deepFlowDocker\output\Alignment\1simRigid"

#comloete process
# استفاده از روش Area برای انتخاب فریم مرجع
results, templates = process_velocity_folder(
    folder_path=input_folder,
    ref_method='similarity'  #  similarity     area
)
#output saving
for r in results:
    save_analysis_outputs(r, output_folder)
    