import numpy as np

def ensure_numpy_2d(x):
    """Return numpy array shape (N, D). If input is (D,), convert to (1,D)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    return x