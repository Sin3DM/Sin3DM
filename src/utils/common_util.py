import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_all(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def draw_scalar_field2D(arr, vmin=None, vmax=None, cmap=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
