from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import Union


class CLIPImageEncoder:
    """
    Image encoder E_v: I → R^(dv) using CLIP model.
    Implements the visual encoder from equation (1).
    """

    def __init__(self, model_name: str = "clip-ViT-B-32", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode(self, image_path: Union[str, Image.Image]) -> np.ndarray:
        """
        Encode image to embedding vector.
        
        Args:
            image_path: Path to image or PIL Image object
            
        Returns:
            Normalized embedding vector of shape (1, dv)
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
            
        # CLIP image encoding
        emb = self.model.encode(
            image, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=1
        )
        
        # Ensure 2D shape (1, dv)
        if emb.ndim == 1:
            emb = emb[None, :]
            
        # L2 normalization for cosine similarity
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        
        return emb.astype(np.float32)
