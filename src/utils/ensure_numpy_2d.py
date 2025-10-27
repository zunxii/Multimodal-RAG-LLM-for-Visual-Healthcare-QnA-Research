import numpy as np

def ensure_numpy_2d(x):
    """
    Return numpy array with shape (N, D). 
    If input is (D,), convert to (1, D).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
