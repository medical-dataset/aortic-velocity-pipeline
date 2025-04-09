"""
Script: eval_nonzero_stats.py
Purpose: Compute and report non-zero statistics from template arrays.
Author: Majid Vafaeezadeh
Date: 2025-04-08
"""
import numpy as np
def get_template_stats(template: np.ndarray):
    nonzero = np.count_nonzero(template)
    total = template.size
    min_val = np.min(template)
    max_val = np.max(template)
    return {
        "non_zero_count": nonzero,
        "non_zero_ratio": nonzero / total,
        "min": min_val,
        "max": max_val,
    }