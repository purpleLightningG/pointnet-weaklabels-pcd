
import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def center_scale_unit_sphere(xyz: np.ndarray):
    c = xyz.mean(0, keepdims=True)
    centered = xyz - c
    scale = max(1e-6, np.linalg.norm(centered, axis=1).max())
    return centered / scale
