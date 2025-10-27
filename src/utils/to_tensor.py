import numpy as np
import torch

def to_tensor(x, device=None):
    """Convert numpy array to torch tensor with optional device placement"""
    x = np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(x)
    if device:
        t = t.to(device)
    return t