import os
import numpy as np
from eval_nonzero_stats import get_template_stats
from eval_visualize import show_template

template_dir = "templates/"
for file in os.listdir(template_dir):
    if file.endswith(".npy"):
        arr = np.load(os.path.join(template_dir, file))
        stats = get_template_stats(arr)
        print(f"{file}: {stats}")
        show_template(arr, title=file, threshold=0.5)
