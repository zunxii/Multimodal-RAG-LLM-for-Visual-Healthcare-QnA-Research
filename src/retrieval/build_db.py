import faiss
import numpy as np
from sklearn.preprocessing import normalize
import torch

from utils import to_tensor
from utils.ensure_numpy_2d import ensure_numpy_2d


def build_temp_db_from_caption_cloud(image_vector: np.ndarray,
                                     caption_cloud: list,
                                     text_encoder,     # encoder object with .encode(list_of_texts) -> np.ndarray (N, dt)
                                     fusion: torch.nn.Module,
                                     device=None):
    """
    Returns: index, emb_matrix (N,d), metadata list aligned with index
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 1) ensure shapes
    zI = ensure_numpy_2d(image_vector)           # (1, dv)
    texts = [c["text"] for c in caption_cloud]
    tau = ensure_numpy_2d(text_encoder.encode(texts))  # (N, dt)

    N = tau.shape[0]
    # 2) repeat image vector to (N, dv)
    zI_repeat = np.repeat(zI, N, axis=0)         # (N, dv)

    # 3) convert to torch and fuse
    zI_t = to_tensor(zI_repeat, device=device)   # torch (N, dv)
    tau_t = to_tensor(tau, device=device)        # torch (N, dt)
    fusion = fusion.to(device)
    fusion.eval()
    with torch.no_grad():
        phi_t = fusion(zI_t, tau_t)              # torch (N, d_out)

    phi = phi_t.cpu().numpy().astype(np.float32) # (N, d_out)
    # 4) normalize each row for cosine similarity
    phi = normalize(phi, axis=1)                 # rows length = 1

    # 5) build FAISS index (inner-product on normalized = cosine)
    d = phi.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(phi)   # now index contains N vectors

    # 6) metadata
    metadata = caption_cloud  # keep as-is; positions align with phi rows

    return index, phi, metadata