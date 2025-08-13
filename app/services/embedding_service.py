from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding Service initialized with model {settings.EMBEDDING_MODEL_NAME}. Dimension: {self.embedding_dim}")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Creates embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True)

# Singleton instance
embedding_service = EmbeddingService()