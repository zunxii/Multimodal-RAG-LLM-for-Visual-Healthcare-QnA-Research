import json
import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any
from sklearn.preprocessing import normalize

from ..embeddings import CLIPImageEncoder, CLIPTextEncoder
from ..fusion import FusionMLP
from ..utils import ensure_numpy_2d, to_tensor, get_logger

logger = get_logger(__name__)


class MultimodalRetriever:
    """
    Dynamic multimodal retrieval system implementing the temporary database
    approach from the paper.
    
    Implements:
    - Building DB_temp^(I) from caption cloud (Algorithm 1, lines 12-19)
    - Query processing and retrieval (Algorithm 1, lines 20-28)
    """

    def __init__(
        self,
        image_encoder: CLIPImageEncoder = None,
        text_encoder: CLIPTextEncoder = None,
        fusion_model: FusionMLP = None,
        device: str = None
    ):
        """
        Initialize retriever with encoders and fusion model.
        
        Args:
            image_encoder: E_v for encoding images
            text_encoder: E_t for encoding text
            fusion_model: F_m for multimodal fusion
            device: Torch device ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize encoders
        self.image_encoder = image_encoder or CLIPImageEncoder(device=self.device)
        self.text_encoder = text_encoder or CLIPTextEncoder(device=self.device)
        
        # Initialize or infer fusion model
        if fusion_model is None:
            # Infer dimensions by encoding dummy data
            dv = 512  # CLIP ViT-B-32 default
            dt = 512
            self.fusion_model = FusionMLP(dv=dv, dt=dt, d_out=512)
        else:
            self.fusion_model = fusion_model
            
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()
        
        # Temporary database state
        self.index: faiss.Index = None
        self.phi_matrix: np.ndarray = None
        self.metadata: List[Dict[str, Any]] = None
        self.image_embedding: np.ndarray = None
        
        logger.info(f"Initialized MultimodalRetriever on device: {self.device}")

    def build_index(
        self, 
        image_path: str, 
        caption_cloud_path: str
    ) -> None:
        """
        Build temporary vector database DB_temp^(I) for the input image.
        
        Implements Algorithm 1 lines 12-19:
        - Compute z^(I) = E_v(I)
        - For each caption c_i: compute τ_i = E_t(c_i)
        - Fuse: φ_i = F_m(W_v·z^(I), W_t·τ_i)
        - Store in FAISS index
        
        Args:
            image_path: Path to input image I
            caption_cloud_path: Path to caption cloud JSON C^(I)_dyn
        """
        logger.info(f"Building index for image: {image_path}")
        
        # Load caption cloud
        with open(caption_cloud_path, 'r', encoding='utf-8') as f:
            caption_cloud = json.load(f)
        
        if len(caption_cloud) == 0:
            raise ValueError("Caption cloud is empty")
        
        # Step 1: Encode image once - z^(I) = E_v(I)
        logger.info("Encoding image...")
        self.image_embedding = self.image_encoder.encode(image_path)  # (1, dv)
        self.image_embedding = ensure_numpy_2d(self.image_embedding)
        
        # Step 2: Encode all captions - τ_i = E_t(c_i) for all i
        logger.info(f"Encoding {len(caption_cloud)} captions...")
        caption_texts = [c["text"] for c in caption_cloud]
        text_embeddings = self.text_encoder.encode(caption_texts)  # (N, dt)
        text_embeddings = ensure_numpy_2d(text_embeddings)
        
        N = len(caption_cloud)
        logger.info(f"Computing {N} multimodal fusions...")
        
        # Step 3: Repeat image embedding to match caption count
        image_repeated = np.repeat(self.image_embedding, N, axis=0)  # (N, dv)
        
        # Step 4: Convert to torch tensors
        z_v_tensor = to_tensor(image_repeated, device=self.device)
        tau_t_tensor = to_tensor(text_embeddings, device=self.device)
        
        # Step 5: Fuse - φ_i = F_m(W_v·z^(I), W_t·τ_i)
        with torch.no_grad():
            phi_tensor = self.fusion_model(z_v_tensor, tau_t_tensor)  # (N, d_out)
        
        phi = phi_tensor.cpu().numpy().astype(np.float32)
        
        # Step 6: Normalize for cosine similarity (inner product on normalized = cosine)
        phi = normalize(phi, axis=1)
        self.phi_matrix = phi
        
        # Step 7: Build FAISS index
        d = phi.shape[1]
        self.index = faiss.IndexFlatIP(d)  # Inner Product for normalized vectors
        self.index.add(phi)
        
        # Step 8: Store metadata
        self.metadata = caption_cloud
        
        logger.info(f"Index built: {N} vectors of dimension {d}")

    def retrieve(
        self, 
        q_clinical: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K neighbors from the temporary index.
        
        Implements Algorithm 1 lines 20-28:
        - Encode query: τ_q = E_t(q_clin)
        - Fuse with same image: φ_q = F_m(W_v·z^(I), W_t·τ_q)
        - Retrieve: R = TopK(φ_q, DB_temp^(I))
        
        Args:
            q_clinical: Clinicalized query string
            top_k: Number of neighbors to retrieve (K)
            
        Returns:
            List of neighbor dictionaries with scores and metadata
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        logger.info(f"Retrieving Top-{top_k} for query: '{q_clinical}'")
        
        # Step 1: Encode query text - τ_q = E_t(q_clin)
        query_text_emb = self.text_encoder.encode(q_clinical)  # (1, dt)
        query_text_emb = ensure_numpy_2d(query_text_emb)
        
        # Step 2: Reuse the same image embedding - z^(I)
        image_emb = ensure_numpy_2d(self.image_embedding)
        
        # Step 3: Convert to torch tensors
        z_v_tensor = to_tensor(image_emb, device=self.device)
        tau_q_tensor = to_tensor(query_text_emb, device=self.device)
        
        # Step 4: Fuse query - φ_q = F_m(W_v·z^(I), W_t·τ_q)
        with torch.no_grad():
            phi_q_tensor = self.fusion_model(z_v_tensor, tau_q_tensor)  # (1, d_out)
        
        phi_q = phi_q_tensor.cpu().numpy().astype(np.float32)
        
        # Step 5: Normalize query vector
        phi_q = normalize(phi_q, axis=1)
        
        # Step 6: Search FAISS index
        distances, indices = self.index.search(phi_q, top_k)
        
        # Step 7: Build neighbor list with evidence bundles
        neighbors = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            neighbors.append({
                "score": float(score),
                "index": int(idx),
                "caption": self.metadata[idx]["text"],
                "meta": self.metadata[idx]
            })
        
        logger.info(f"Retrieved {len(neighbors)} neighbors")
        return neighbors
