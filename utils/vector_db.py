import faiss
import numpy as np
import pandas as pd
from config import VECTOR_EMBEDDINGS_PATH, CSV_DATA_PATH

df = pd.read_csv(CSV_DATA_PATH)

try:
    db_vectors = np.load(VECTOR_EMBEDDINGS_PATH)
except:
    # Mock embeddings for demo purposes
    db_vectors = np.random.rand(len(df), 512)
    np.save(VECTOR_EMBEDDINGS_PATH, db_vectors)

index = faiss.IndexFlatL2(db_vectors.shape[1])
index.add(db_vectors) # type: ignore

def search_similar_vectors(query_vector, k=3):
    D, I = index.search(np.array([query_vector]), k) # type: ignore
    return df.iloc[I[0]]['Summary'].tolist()
