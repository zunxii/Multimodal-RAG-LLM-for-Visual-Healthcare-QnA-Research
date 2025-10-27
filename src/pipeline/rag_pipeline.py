"""
Complete end-to-end pipeline for Dynamic Multimodal RAG-LLM
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from ..captioning import DynamicCaptionCloud
from ..embeddings import CLIPImageEncoder, CLIPTextEncoder
from ..fusion import FusionMLP
from ..clinicalization import clinicalize_query
from ..retrieval import MultimodalRetriever
from ..utils import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete Dynamic Multimodal RAG-LLM pipeline.
    
    Orchestrates the full workflow from Algorithm 1:
    1. Generate dynamic caption cloud
    2. Build temporary multimodal index
    3. Clinicalize query and retrieve
    4. (Future) Generate and fuse answers
    """

    def __init__(
        self,
        caption_output_dir: str = "data/captions",
        device: str = None
    ):
        """
        Initialize the pipeline with all components.
        
        Args:
            caption_output_dir: Directory to store caption clouds
            device: Torch device ('cuda' or 'cpu')
        """
        # Initialize components
        self.caption_cloud_builder = DynamicCaptionCloud(
            output_dir=caption_output_dir
        )
        
        self.image_encoder = CLIPImageEncoder(device=device)
        self.text_encoder = CLIPTextEncoder(device=device)
        
        # Initialize fusion model with correct dimensions
        dv = 512  # CLIP ViT-B-32
        dt = 512
        self.fusion_model = FusionMLP(dv=dv, dt=dt, d_out=512)
        
        self.retriever = MultimodalRetriever(
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            fusion_model=self.fusion_model,
            device=device
        )
        
        logger.info("RAGPipeline initialized")

    def run(
        self,
        image_path: str,
        user_query: str,
        n_prompts: int = 4,
        n_seeds: int = 2,
        top_k: int = 5,
        use_cached_captions: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            image_path: Path to input image
            user_query: Free-text user query
            n_prompts: Number of prompts per VLM
            n_seeds: Number of seeds per prompt
            top_k: Number of neighbors to retrieve
            use_cached_captions: Use existing caption cloud if available
            
        Returns:
            Dictionary with results including neighbors and metadata
        """
        image_path = Path(image_path)
        
        logger.info("="*60)
        logger.info("Starting Dynamic Multimodal RAG Pipeline")
        logger.info("="*60)
        logger.info(f"Image: {image_path}")
        logger.info(f"Query: {user_query}")
        
        # Step 1: Generate or load caption cloud
        caption_cloud_path = (
            self.caption_cloud_builder.output_dir / f"{image_path.stem}.json"
        )
        
        if use_cached_captions and caption_cloud_path.exists():
            logger.info(f"Using cached caption cloud: {caption_cloud_path}")
        else:
            logger.info("Generating dynamic caption cloud...")
            caption_cloud_path = self.caption_cloud_builder.build_cloud(
                str(image_path),
                n_prompts=n_prompts,
                n_seeds=n_seeds
            )
        
        # Load caption cloud for statistics
        with open(caption_cloud_path, 'r') as f:
            caption_cloud = json.load(f)
        
        # Step 2: Build temporary index
        logger.info("Building temporary multimodal index...")
        self.retriever.build_index(
            image_path=str(image_path),
            caption_cloud_path=str(caption_cloud_path)
        )
        
        # Step 3: Clinicalize query
        logger.info("Clinicalizing user query...")
        q_clinical = clinicalize_query(user_query)
        
        # Step 4: Retrieve neighbors
        logger.info("Retrieving evidence from dynamic database...")
        neighbors = self.retriever.retrieve(
            q_clinical=q_clinical,
            top_k=top_k
        )
        
        # Compile results
        results = {
            "image_path": str(image_path),
            "user_query": user_query,
            "clinical_query": q_clinical,
            "caption_cloud_stats": {
                "total_captions": len(caption_cloud),
                "vlm_models": list(set(c["model"] for c in caption_cloud)),
                "unique_prompts": len(set(c["prompt"] for c in caption_cloud))
            },
            "neighbors": neighbors,
            "retrieval_scores": [n["score"] for n in neighbors]
        }
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully")
        logger.info(f"Retrieved {len(neighbors)} neighbors")
        logger.info(f"Top score: {neighbors[0]['score']:.4f}" if neighbors else "No neighbors")
        logger.info("="*60)
        
        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print pipeline results"""
        print("\n" + "="*70)
        print("DYNAMIC MULTIMODAL RAG RESULTS")
        print("="*70)
        
        print(f"\n Image: {results['image_path']}")
        print(f"\n User Query: {results['user_query']}")
        print(f"\n Clinical Query: {results['clinical_query']}")
        
        stats = results['caption_cloud_stats']
        print(f"\n Caption Cloud Statistics:")
        print(f"   • Total captions: {stats['total_captions']}")
        print(f"   • VLM models: {', '.join(stats['vlm_models'])}")
        print(f"   • Unique prompts: {stats['unique_prompts']}")
        
        print(f"\n Top-{len(results['neighbors'])} Retrieved Neighbors:")
        print("-"*70)
        
        for i, neighbor in enumerate(results['neighbors'], 1):
            print(f"\n[Rank {i}] Score: {neighbor['score']:.4f}")
            print(f"Model: {neighbor['meta']['model']} | Seed: {neighbor['meta']['seed']}")
            print(f"Prompt: {neighbor['meta']['prompt'][:60]}...")
            print(f"Caption: {neighbor['caption'][:200]}...")
        
        print("\n" + "="*70 + "\n")
