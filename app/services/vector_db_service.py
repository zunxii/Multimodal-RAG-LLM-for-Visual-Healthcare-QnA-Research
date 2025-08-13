import faiss
import numpy as np

class VectorDBService:
    def __init__(self, dimension: int):
        # Using a flat L2 index, which is exact and good for a small number of vectors (<1M)
        self.index = faiss.IndexFlatL2(dimension)
        self.descriptions = []
        print(f"Vector DB Service initialized with dimension {dimension}.")

    def build_index(self, vectors: np.ndarray, descriptions: list[str]):
        """Builds the FAISS index with the given vectors."""
        if vectors.shape[0] != len(descriptions):
            raise ValueError("Number of vectors and descriptions must be the same.")
        self.index.add(vectors)
        self.descriptions = descriptions
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_vector: np.ndarray, k: int = 1) -> tuple[int, str]:
        """
        Searches the index for the most similar vector.
        Returns the index and the corresponding description text.
        """
        if self.index.ntotal == 0:
            return -1, "No data in the index."
            
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Get the top result (index 0)
        retrieved_index = indices[0][0]
        retrieved_description = self.descriptions[retrieved_index]
        
        return retrieved_index, retrieved_description