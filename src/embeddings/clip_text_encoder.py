from sentence_transformers import SentenceTransformer
import numpy as np

class CLIPTextEncoder:
    """
    Text encoder using CLIP model (text tower).
    """

    def __init__(self, model_name="clip-ViT-B-32", device=None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float32)
