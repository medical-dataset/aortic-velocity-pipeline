import matplotlib.pyplot as plt
import numpy as np
def show_template(template: np.ndarray, title="Template", threshold=None):
    if threshold is not None:
        template = (template > threshold).astype(np.uint8)
    plt.imshow(template, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
