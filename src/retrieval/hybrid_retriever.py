"""
Hybrid retrieval system combining:
1. Knowledge base retrieval (from ClipSyntel)
2. Dynamic caption cloud retrieval (from query image)
"""

import numpy as np
import torch
from typing import List, Dict, Any
from sklearn.preprocessing import normalize

from .knowledge_base import MedicalKnowledgeBase
from .multimodal_retriever import MultimodalRetriever
from ..utils import ensure_numpy_2d, to_tensor, get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines static KB retrieval with dynamic caption cloud retrieval.
    
    Retrieval Strategy:
    1. Query both KB and dynamic caption cloud
    2. Merge and re-rank results
    3. Return diverse, high-quality evidence
    """
    
    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        dynamic_retriever: MultimodalRetriever,
        device: str = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            knowledge_base: Static medical knowledge base
            dynamic_retriever: Dynamic caption cloud retriever
            device: Torch device
        """
        self.kb = knowledge_base
        self.dynamic_retriever = dynamic_retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Initialized HybridRetriever")
    
    def retrieve(
        self,
        image_path: str,
        q_clinical: str,
        caption_cloud_path: str = None,
        kb_top_k: int = 5,
        dynamic_top_k: int = 5,
        final_top_k: int = 10,
        kb_weight: float = 0.6,
        dynamic_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval.
        
        Args:
            image_path: Query image path
            q_clinical: Clinicalized query
            caption_cloud_path: Path to caption cloud JSON
            kb_top_k: Top-K from knowledge base
            dynamic_top_k: Top-K from dynamic caption cloud
            final_top_k: Final top-K after merging
            kb_weight: Weight for KB scores
            dynamic_weight: Weight for dynamic scores
            
        Returns:
            Dictionary with KB results, dynamic results, and merged results
        """
        logger.info("="*60)
        logger.info("Hybrid Retrieval: KB + Dynamic Caption Cloud")
        logger.info("="*60)
        
        # 1. Build dynamic index if caption cloud provided
        if caption_cloud_path:
            logger.info("Building dynamic caption cloud index...")
            self.dynamic_retriever.build_index(
                image_path=image_path,
                caption_cloud_path=caption_cloud_path
            )
        
        # 2. Retrieve from dynamic caption cloud
        logger.info(f"Retrieving Top-{dynamic_top_k} from dynamic caption cloud...")
        dynamic_results = self.dynamic_retriever.retrieve(
            q_clinical=q_clinical,
            top_k=dynamic_top_k
        )
        
        # 3. Compute query embedding for KB retrieval
        logger.info(f"Retrieving Top-{kb_top_k} from knowledge base...")
        
        # Encode query image and text
        z_img = self.dynamic_retriever.image_encoder.encode(image_path)
        z_img = ensure_numpy_2d(z_img)
        
        tau_txt = self.dynamic_retriever.text_encoder.encode(q_clinical)
        tau_txt = ensure_numpy_2d(tau_txt)
        
        # Convert to tensors
        z_tensor = to_tensor(z_img, device=self.device)
        tau_tensor = to_tensor(tau_txt, device=self.device)
        
        # Fuse
        with torch.no_grad():
            phi_q = self.dynamic_retriever.fusion_model(z_tensor, tau_tensor)
        
        phi_q_np = phi_q.cpu().numpy().astype(np.float32)
        
        # 4. Retrieve from KB
        kb_results = self.kb.retrieve_from_kb(
            query_embedding=phi_q_np,
            top_k=kb_top_k
        )
        
        # 5. Merge and re-rank results
        logger.info("Merging and re-ranking results...")
        merged_results = self._merge_results(
            kb_results=kb_results,
            dynamic_results=dynamic_results,
            kb_weight=kb_weight,
            dynamic_weight=dynamic_weight,
            top_k=final_top_k
        )
        
        results = {
            "kb_results": kb_results,
            "dynamic_results": dynamic_results,
            "merged_results": merged_results,
            "kb_count": len(kb_results),
            "dynamic_count": len(dynamic_results),
            "final_count": len(merged_results)
        }
        
        logger.info(f"Retrieved {len(kb_results)} from KB, {len(dynamic_results)} from dynamic")
        logger.info(f"Final merged: {len(merged_results)} results")
        logger.info("="*60)
        
        return results
    
    def _merge_results(
        self,
        kb_results: List[Dict[str, Any]],
        dynamic_results: List[Dict[str, Any]],
        kb_weight: float,
        dynamic_weight: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Merge KB and dynamic results with weighted scoring.
        
        Strategy:
        - Normalize scores to [0, 1]
        - Apply weights
        - Remove duplicates
        - Sort by combined score
        """
        merged = []
        
        # Process KB results
        for result in kb_results:
            merged.append({
                "source": "knowledge_base",
                "score": float(result["retrieval_score"]) * kb_weight,
                "raw_score": float(result["retrieval_score"]),
                "content": result.get("caption", result.get("description", "")),
                "metadata": {
                    "image_path": result.get("image_path"),
                    "category": result.get("category"),
                    "context": result.get("context"),
                    "description": result.get("description")
                }
            })
        
        # Process dynamic results
        for result in dynamic_results:
            merged.append({
                "source": "dynamic_caption_cloud",
                "score": float(result["score"]) * dynamic_weight,
                "raw_score": float(result["score"]),
                "content": result["caption"],
                "metadata": {
                    "model": result["meta"].get("model"),
                    "prompt": result["meta"].get("prompt"),
                    "seed": result["meta"].get("seed")
                }
            })
        
        # Sort by combined score
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-K
        return merged[:top_k]