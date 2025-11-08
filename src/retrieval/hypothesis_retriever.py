"""

Retrieve using multiple medical hypotheses and aggregate evidence.
Implements diagnostic consensus from multiple retrieval perspectives.
"""

import numpy as np
import torch
from typing import List, Dict, Any
from collections import Counter

from .knowledge_base import MedicalKnowledgeBase
from .clinical_relevance import ClinicalRelevanceScorer
from ..captioning.medical_hypothesis_generator import MedicalHypothesis
from ..utils import ensure_numpy_2d, to_tensor, get_logger

logger = get_logger(__name__)


class MultiHypothesisRetriever:
    """
    Retrieve evidence using multiple medical hypotheses.
    
    Each hypothesis targets different diagnostic pathways.
    Aggregates results to determine diagnostic consensus.
    """
    
    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        relevance_scorer: ClinicalRelevanceScorer = None,
        device: str = None
    ):
        """
        Initialize multi-hypothesis retriever.
        
        Args:
            knowledge_base: Medical KB for retrieval
            relevance_scorer: Clinical relevance scorer
            device: Torch device
        """
        self.kb = knowledge_base
        self.relevance_scorer = relevance_scorer or ClinicalRelevanceScorer()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Initialized MultiHypothesisRetriever")
    
    def retrieve_with_hypotheses(
        self,
        image_path: str,
        hypotheses: List[MedicalHypothesis],
        top_k_per_hypothesis: int = 10,
        relevance_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Retrieve evidence for each hypothesis and aggregate.
        
        Args:
            image_path: Query image path
            hypotheses: List of medical hypotheses
            top_k_per_hypothesis: Top-K to retrieve per hypothesis
            relevance_threshold: Minimum clinical relevance
            
        Returns:
            Aggregated retrieval results with diagnostic consensus
        """
        logger.info(f"Retrieving for {len(hypotheses)} hypotheses...")
        
        # Retrieve for each hypothesis
        hypothesis_evidence = {}
        
        for i, hypothesis in enumerate(hypotheses):
            logger.info(
                f"Hypothesis {i+1}/{len(hypotheses)}: {hypothesis.diagnosis}"
            )
            
            # Encode hypothesis as structured query
            query_embedding = self._encode_hypothesis(
                image_path,
                hypothesis
            )
            
            # Retrieve from KB
            candidates = self.kb.retrieve_from_kb(
                query_embedding=query_embedding,
                top_k=top_k_per_hypothesis
            )
            
            # Filter by clinical relevance
            query_features = self._hypothesis_to_features(hypothesis)
            
            relevant_cases = self.relevance_scorer.filter_by_relevance(
                candidates=candidates,
                query_features=query_features,
                threshold=relevance_threshold
            )
            
            hypothesis_evidence[hypothesis.diagnosis] = {
                "hypothesis": hypothesis,
                "cases": relevant_cases,
                "count": len(relevant_cases)
            }
        
        # Aggregate evidence across hypotheses
        aggregation = self._aggregate_evidence(hypothesis_evidence)
        
        return {
            "hypothesis_evidence": hypothesis_evidence,
            "aggregation": aggregation,
            "num_hypotheses": len(hypotheses)
        }
    
    def _encode_hypothesis(
        self,
        image_path: str,
        hypothesis: MedicalHypothesis
    ) -> np.ndarray:
        """
        Encode hypothesis as multimodal query embedding.
        
        Combines:
        - Image encoding
        - Diagnosis + clinical context text encoding
        - Visual features as additional text
        """
        # Encode image
        z_img = self.kb.image_encoder.encode(image_path)
        z_img = ensure_numpy_2d(z_img)
        
        # Build text from hypothesis
        hypothesis_text = self._build_hypothesis_text(hypothesis)
        
        # Encode text
        tau_txt = self.kb.text_encoder.encode(hypothesis_text)
        tau_txt = ensure_numpy_2d(tau_txt)
        
        # Convert to tensors
        z_tensor = to_tensor(z_img, device=self.device)
        tau_tensor = to_tensor(tau_txt, device=self.device)
        
        # Fuse
        with torch.no_grad():
            phi = self.kb.fusion_model(z_tensor, tau_tensor)
        
        return phi.cpu().numpy().astype(np.float32)
    
    def _build_hypothesis_text(self, hypothesis: MedicalHypothesis) -> str:
        """Build text representation of hypothesis for encoding"""
        
        parts = [
            f"Diagnosis: {hypothesis.diagnosis}",
            f"Mechanism: {hypothesis.mechanism}",
            f"Visual features: {', '.join(hypothesis.visual_features)}",
            f"Clinical context: {hypothesis.clinical_context}"
        ]
        
        return ". ".join(parts)
    
    def _hypothesis_to_features(
        self,
        hypothesis: MedicalHypothesis
    ) -> Dict[str, Any]:
        """Convert hypothesis to feature dict for relevance scoring"""
        
        return {
            "diagnosis": hypothesis.diagnosis,
            "body_regions": hypothesis.body_regions,
            "keywords": hypothesis.keywords,
            "visual_features": hypothesis.visual_features,
            "urgency": hypothesis.urgency
        }
    
    def _aggregate_evidence(
        self,
        hypothesis_evidence: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate evidence across hypotheses for diagnostic consensus.
        
        Determines:
        - Primary diagnosis (most evidence support)
        - Confidence based on evidence quality & quantity
        - Differential diagnoses with probabilities
        """
        diagnosis_scores = {}
        
        for diagnosis, evidence in hypothesis_evidence.items():
            cases = evidence["cases"]
            
            if not cases:
                continue
            
            # Evidence quality (mean combined score)
            quality = np.mean([
                c.get("combined_score", 0.0) for c in cases
            ]) if cases else 0.0
            
            # Evidence quantity (normalized)
            quantity = len(cases) / 10.0  # Normalize to [0, 1]
            quantity = min(quantity, 1.0)
            
            # Evidence diversity (unique categories)
            categories = set(c.get("category", "") for c in cases)
            diversity = len(categories) / 5.0  # Normalize
            diversity = min(diversity, 1.0)
            
            # Combined confidence score
            confidence = (
                0.4 * quality +
                0.3 * quantity +
                0.3 * diversity
            )
            
            diagnosis_scores[diagnosis] = {
                "quality": float(quality),
                "quantity": float(quantity),
                "diversity": float(diversity),
                "confidence": float(confidence),
                "case_count": len(cases),
                "supporting_cases": cases[:5]  # Top 5
            }
        
        if not diagnosis_scores:
            return {
                "primary_diagnosis": "Insufficient evidence",
                "confidence": 0.0,
                "differential": []
            }
        
        # Sort by confidence
        ranked = sorted(
            diagnosis_scores.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True
        )
        
        primary = ranked[0]
        differential = ranked[1:4] if len(ranked) > 1 else []
        
        return {
            "primary_diagnosis": primary[0],
            "confidence": primary[1]["confidence"],
            "supporting_cases": primary[1]["supporting_cases"],
            "case_count": primary[1]["case_count"],
            "differential": [
                {
                    "diagnosis": diag,
                    "probability": scores["confidence"],
                    "case_count": scores["case_count"]
                }
                for diag, scores in differential
            ],
            "all_scores": diagnosis_scores
        }