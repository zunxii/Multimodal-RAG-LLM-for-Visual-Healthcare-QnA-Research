import numpy as np
import torch

def to_tensor(x, device=None):
    x = np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(x)
    if device:
        t = t.to(device)
    return t