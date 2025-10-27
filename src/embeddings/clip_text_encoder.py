from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List


class CLIPTextEncoder:
    """
    Text encoder E_t: T → R^(dt) using CLIP model.
    Implements the text encoder from equation (2).
    """

    def __init__(self, model_name: str = "clip-ViT-B-32", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embedding vector(s).
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Normalized embedding vector(s) of shape (N, dt)
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
            
        # CLIP text encoding
        emb = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Ensure 2D shape
        if emb.ndim == 1:
            emb = emb[None, :]
            
        # L2 normalization for cosine similarity
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        
        return emb.astype(np.float32)
