"""
Build and manage the medical knowledge base from ClipSyntel dataset.
Fixed to work with CSV format.
"""

import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pickle

from ..embeddings import CLIPImageEncoder, CLIPTextEncoder
from ..fusion import FusionMLP
from ..utils import ensure_numpy_2d, to_tensor, get_logger
from sklearn.preprocessing import normalize
import torch

logger = get_logger(__name__)


class MedicalKnowledgeBase:
    """
    Static knowledge base built from ClipSyntel dataset.
    Stores pre-computed multimodal embeddings for retrieval.
    """
    
    def __init__(
        self,
        kb_path: str = "data/knowledge_base",
        image_encoder: CLIPImageEncoder = None,
        text_encoder: CLIPTextEncoder = None,
        fusion_model: FusionMLP = None,
        device: str = None
    ):
        """
        Initialize knowledge base manager.
        
        Args:
            kb_path: Directory to store knowledge base files
            image_encoder: E_v for encoding images
            text_encoder: E_t for encoding text
            fusion_model: F_m for multimodal fusion
            device: Torch device
        """
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize encoders
        self.image_encoder = image_encoder or CLIPImageEncoder(device=self.device)
        self.text_encoder = text_encoder or CLIPTextEncoder(device=self.device)
        
        # Initialize fusion model
        if fusion_model is None:
            dv, dt = 512, 512
            self.fusion_model = FusionMLP(dv=dv, dt=dt, d_out=512)
        else:
            self.fusion_model = fusion_model
            
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()
        
        # Knowledge base state
        self.index: Optional[faiss.Index] = None
        self.cases: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Initialized MedicalKnowledgeBase at {self.kb_path}")
    
    def build_from_clipsyntel(
        self,
        dataset_path: str,
        save_name: str = "clipsyntel_kb",
        use_caption_cloud: bool = False,
        n_prompts: int = 3,
        n_seeds: int = 2,
        image_base_path: str = "data/images"
    ) -> None:
        """
        Build knowledge base from ClipSyntel CSV dataset.
        
        Args:
            dataset_path: Path to ClipSyntel CSV file
            save_name: Name for saved knowledge base files
            use_caption_cloud: Whether to generate caption clouds for KB images
            n_prompts: Number of prompts per VLM (if using caption cloud)
            n_seeds: Number of seeds per prompt (if using caption cloud)
            image_base_path: Base path for images
        """
        logger.info("="*60)
        logger.info("Building Knowledge Base from ClipSyntel CSV Dataset")
        logger.info("="*60)
        
        # Load CSV dataset
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Reading CSV: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        logger.info(f"Loaded {len(df)} cases from ClipSyntel")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Process each case
        all_embeddings = []
        processed_cases = []
        
        image_base = Path(image_base_path)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
            try:
                # Extract case information
                # CSV headers: Question,Question_summ,image_path,category,context,description
                image_filename = row['image_path']
                question = row.get('Question', '')
                question_summ = row.get('Question_summ', '')
                description = row.get('description', '')
                category = row.get('category', '')
                context = row.get('context', '')
                
                # Construct full image path
                image_path = image_base / image_filename
                
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}, skipping...")
                    continue
                
                # Use question or description as caption
                caption = question_summ if pd.notna(question_summ) else question
                if not caption or pd.isna(caption):
                    caption = description if pd.notna(description) else "Medical image"
                
                if use_caption_cloud:
                    # Generate caption cloud for this KB image
                    caption = self._generate_kb_caption_cloud(
                        str(image_path), 
                        n_prompts=n_prompts, 
                        n_seeds=n_seeds
                    )
                
                # Encode image
                z_img = self.image_encoder.encode(str(image_path))
                z_img = ensure_numpy_2d(z_img)
                
                # Encode caption
                tau_txt = self.text_encoder.encode(str(caption))
                tau_txt = ensure_numpy_2d(tau_txt)
                
                # Convert to tensors
                z_tensor = to_tensor(z_img, device=self.device)
                tau_tensor = to_tensor(tau_txt, device=self.device)
                
                # Fuse: φ_kb = F_m(W_v·z, W_t·τ)
                with torch.no_grad():
                    phi = self.fusion_model(z_tensor, tau_tensor)
                
                phi_np = phi.cpu().numpy().astype(np.float32)
                phi_np = normalize(phi_np, axis=1)
                
                all_embeddings.append(phi_np[0])
                
                # Store case metadata
                processed_cases.append({
                    "image_path": str(image_path),
                    "image_filename": image_filename,
                    "caption": str(caption),
                    "description": str(description) if pd.notna(description) else "",
                    "category": str(category) if pd.notna(category) else "",
                    "context": str(context) if pd.notna(context) else "",
                    "question": str(question) if pd.notna(question) else "",
                    "question_summ": str(question_summ) if pd.notna(question_summ) else ""
                })
                
            except Exception as e:
                logger.error(f"Failed to process case at index {idx}: {e}")
                continue
        
        if len(all_embeddings) == 0:
            raise ValueError("No cases were successfully processed!")
        
        # Build FAISS index
        embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
        d = embeddings_matrix.shape[1]
        
        logger.info(f"Building FAISS index with {len(embeddings_matrix)} vectors of dim {d}")
        
        index = faiss.IndexFlatIP(d)  # Inner product for normalized vectors
        index.add(embeddings_matrix)
        
        # Save knowledge base
        self.index = index
        self.cases = processed_cases
        self.embeddings = embeddings_matrix
        
        self._save_kb(save_name)
        
        logger.info("="*60)
        logger.info(f"Knowledge Base Built: {len(processed_cases)} cases indexed")
        logger.info(f"Saved to: {self.kb_path / save_name}")
        logger.info("="*60)
    
    def _generate_kb_caption_cloud(
        self, 
        image_path: str, 
        n_prompts: int, 
        n_seeds: int
    ) -> str:
        """
        Generate caption cloud for a knowledge base image.
        Returns a combined caption from multiple VLM generations.
        """
        from ..captioning import DynamicCaptionCloud
        
        caption_builder = DynamicCaptionCloud()
        caption_path = caption_builder.build_cloud(
            image_path,
            n_prompts=n_prompts,
            n_seeds=n_seeds
        )
        
        with open(caption_path, 'r') as f:
            captions = json.load(f)
        
        # Combine captions (take most informative or concatenate)
        combined = " ".join([c["text"][:200] for c in captions[:3]])
        return combined
    
    def load_kb(self, save_name: str = "clipsyntel_kb") -> None:
        """Load existing knowledge base from disk."""
        logger.info(f"Loading knowledge base: {save_name}")
        
        index_path = self.kb_path / f"{save_name}.index"
        cases_path = self.kb_path / f"{save_name}_cases.json"
        embeddings_path = self.kb_path / f"{save_name}_embeddings.npy"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load cases metadata
        with open(cases_path, 'r', encoding='utf-8') as f:
            self.cases = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(embeddings_path)
        
        logger.info(f"Loaded {len(self.cases)} cases from knowledge base")
    
    def _save_kb(self, save_name: str) -> None:
        """Save knowledge base to disk."""
        logger.info(f"Saving knowledge base: {save_name}")
        
        index_path = self.kb_path / f"{save_name}.index"
        cases_path = self.kb_path / f"{save_name}_cases.json"
        embeddings_path = self.kb_path / f"{save_name}_embeddings.npy"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save cases metadata
        with open(cases_path, 'w', encoding='utf-8') as f:
            json.dump(self.cases, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        np.save(embeddings_path, self.embeddings)
        
        logger.info(f"Knowledge base saved to {self.kb_path}")
    
    def retrieve_from_kb(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K similar cases from knowledge base.
        
        Args:
            query_embedding: Query embedding φ_q of shape (1, d)
            top_k: Number of neighbors to retrieve
            
        Returns:
            List of retrieved cases with scores
        """
        if self.index is None:
            raise RuntimeError("Knowledge base not loaded. Call load_kb() first.")
        
        query_embedding = ensure_numpy_2d(query_embedding)
        query_embedding = normalize(query_embedding, axis=1)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.cases):
                continue
            
            case = self.cases[idx].copy()
            case["retrieval_score"] = float(score)
            case["kb_index"] = int(idx)
            results.append(case)
        
        return results