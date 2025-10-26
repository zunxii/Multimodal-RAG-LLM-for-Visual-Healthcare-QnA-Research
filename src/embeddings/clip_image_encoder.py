from sentence_transformers import SentenceTransformer
import numpy as np

class CLIPImageEncoder:
    """
    Image encoder using CLIP model via sentence-transformers.
    """

    def __init__(self, model_name="clip-ViT-B-32", device=None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, image_path: str) -> np.ndarray:
        emb = self.model.encode([image_path], convert_to_numpy=True, show_progress_bar=False)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float32)
