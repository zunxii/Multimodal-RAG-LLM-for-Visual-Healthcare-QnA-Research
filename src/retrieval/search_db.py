import numpy as np
from sklearn.preprocessing import normalize
import torch
from utils import to_tensor
from utils.ensure_numpy_2d import ensure_numpy_2d


def query_temp_db(index, phi_matrix, metadata, image_vector, q_clinical, text_encoder, fusion, top_k=5, device=None):
    """
    Returns list of neighbors: [{'score':..., 'index':..., 'caption':..., 'meta':...}, ...]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 1) encode query text
    q_vec = ensure_numpy_2d(text_encoder.encode([q_clinical]))  # (1, dt)
    # 2) image z(I) -> if image_vector already computed: ensure shape
    zI = ensure_numpy_2d(image_vector)    # (1, dv)

    # 3) convert to torch and fuse
    zI_t = to_tensor(zI, device=device).float()
    q_t = to_tensor(q_vec, device=device).float()
    fusion = fusion.to(device)
    with torch.no_grad():
        phi_q_t = fusion(zI_t, q_t)       # (1, d_out)
    phi_q = phi_q_t.cpu().numpy().astype(np.float32)
    # 4) normalize
    phi_q = normalize(phi_q, axis=1)

    # 5) search
    D, I = index.search(phi_q, top_k)     # D shape (1, K), I shape (1, K)
    neighbors = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        neighbors.append({
            "score": float(score),
            "index": int(idx),
            "caption": metadata[idx]["text"],
            "meta": metadata[idx]
        })
    return neighbors