"""
Complete RAG pipeline using medical hypothesis-based retrieval.
Replaces caption cloud approach with diagnostic reasoning.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..captioning.medical_hypothesis_generator import (
    MedicalHypothesisGenerator,
    MedicalHypothesis
)
from ..embeddings import CLIPImageEncoder, CLIPTextEncoder
from ..fusion import FusionMLP
from ..retrieval.knowledge_base import MedicalKnowledgeBase
from ..retrieval.clinical_relevance import ClinicalRelevanceScorer
from ..retrieval.hypothesis_retriever import MultiHypothesisRetriever
from ..generator import AnswerGenerator
from ..evaluation import RAGEvaluator
from ..utils import get_logger

logger = get_logger(__name__)


class HypothesisRAGPipeline:
    """
    Medical RAG pipeline using hypothesis-based retrieval.
    
    Flow:
    1. Generate medical hypotheses (not captions)
    2. Retrieve evidence for each hypothesis
    3. Aggregate with diagnostic consensus
    4. Generate grounded answer
    5. Evaluate
    """
    
    def __init__(
        self,
        kb_path: str = "data/knowledge_base",
        device: str = None
    ):
        """
        Initialize hypothesis-based RAG pipeline.
        
        Args:
            kb_path: Path to knowledge base
            device: Torch device
        """
        self.device = device
        
        logger.info("Initializing HypothesisRAGPipeline...")
        
        # Initialize encoders and fusion
        self.image_encoder = CLIPImageEncoder(device=device)
        self.text_encoder = CLIPTextEncoder(device=device)
        self.fusion_model = FusionMLP(dv=512, dt=512, d_out=512)
        
        # Initialize hypothesis generator
        self.hypothesis_generator = MedicalHypothesisGenerator()
        
        # Initialize knowledge base
        self.knowledge_base = MedicalKnowledgeBase(
            kb_path=kb_path,
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            fusion_model=self.fusion_model,
            device=device
        )
        
        # Initialize clinical relevance scorer
        self.relevance_scorer = ClinicalRelevanceScorer()
        
        # Initialize multi-hypothesis retriever
        self.hypothesis_retriever = MultiHypothesisRetriever(
            knowledge_base=self.knowledge_base,
            relevance_scorer=self.relevance_scorer,
            device=device
        )
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator()
        
        # Initialize evaluator
        self.evaluator = RAGEvaluator()
        
        logger.info("HypothesisRAGPipeline initialized")
    
    def build_knowledge_base(
        self,
        dataset_path: str,
        save_name: str = "medical_kb",
        image_base_path: str = "data/images"
    ) -> None:
        """Build knowledge base from dataset"""
        
        self.knowledge_base.build_from_clipsyntel(
            dataset_path=dataset_path,
            save_name=save_name,
            use_caption_cloud=False,
            image_base_path=image_base_path
        )
    
    def load_knowledge_base(self, save_name: str = "medical_kb") -> None:
        """Load existing knowledge base"""
        
        self.knowledge_base.load_kb(save_name=save_name)
    
    def run(
        self,
        image_path: str,
        user_query: str,
        ground_truth_answer: str = None,
        n_hypotheses: int = 5,
        top_k_per_hypothesis: int = 10,
        relevance_threshold: float = 0.3,
        use_cached_hypotheses: bool = True,
        generate_answer: bool = True,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete hypothesis-based RAG pipeline.
        
        Args:
            image_path: Query image path
            user_query: User's clinical question
            ground_truth_answer: Reference answer for evaluation
            n_hypotheses: Number of medical hypotheses to generate
            top_k_per_hypothesis: Top-K cases per hypothesis
            relevance_threshold: Minimum clinical relevance score
            use_cached_hypotheses: Use cached hypotheses if available
            generate_answer: Whether to generate answer
            evaluate: Whether to run evaluation
            
        Returns:
            Complete results dictionary
        """
        logger.info("="*70)
        logger.info("HYPOTHESIS-BASED RAG PIPELINE")
        logger.info("="*70)
        logger.info(f"Image: {image_path}")
        logger.info(f"Query: {user_query}")
        
        results = {
            "image_path": str(image_path),
            "user_query": user_query,
            "pipeline_config": {
                "approach": "hypothesis_based",
                "n_hypotheses": n_hypotheses,
                "top_k_per_hypothesis": top_k_per_hypothesis,
                "relevance_threshold": relevance_threshold
            }
        }
        
        # Step 1: Generate medical hypotheses
        hypotheses = self._get_hypotheses(
            image_path=image_path,
            clinical_query=user_query,
            n_hypotheses=n_hypotheses,
            use_cached=use_cached_hypotheses
        )
        
        results["hypotheses"] = [
            {
                "diagnosis": h.diagnosis,
                "mechanism": h.mechanism,
                "urgency": h.urgency,
                "keywords": h.keywords
            }
            for h in hypotheses
        ]
        
        # Step 2: Retrieve evidence for each hypothesis
        logger.info("Retrieving evidence for hypotheses...")
        retrieval_results = self.hypothesis_retriever.retrieve_with_hypotheses(
            image_path=str(image_path),
            hypotheses=hypotheses,
            top_k_per_hypothesis=top_k_per_hypothesis,
            relevance_threshold=relevance_threshold
        )
        
        results["retrieval"] = {
            "hypothesis_evidence": {
                diag: {
                    "case_count": evidence["count"],
                    "top_cases": evidence["cases"][:3]
                }
                for diag, evidence in retrieval_results["hypothesis_evidence"].items()
            },
            "aggregation": retrieval_results["aggregation"]
        }
        
        # Step 3: Generate answer
        if generate_answer:
            logger.info("Generating evidence-grounded answer...")
            
            # Prepare evidence for generation
            aggregation = retrieval_results["aggregation"]
            evidence_for_generation = self._prepare_evidence(
                aggregation,
                retrieval_results["hypothesis_evidence"]
            )
            
            generation_result = self.answer_generator.generate_answer(
                query=user_query,
                retrieved_evidence=evidence_for_generation,
                image_path=str(image_path),
                include_provenance=True
            )
            
            results["generation"] = generation_result
            logger.info(
                f"Answer generated (confidence: {generation_result['confidence']:.2f})"
            )
        
        # Step 4: Evaluate
        if evaluate and ground_truth_answer and generate_answer:
            logger.info("Running evaluation...")
            
            eval_results = self.evaluator.evaluate(
                prediction=generation_result["answer"],
                reference=ground_truth_answer,
                metrics=["BLEU", "ROUGE", "BERTScore"]
            )
            
            results["evaluation"] = {
                "answer_quality": eval_results,
                "ground_truth": ground_truth_answer
            }
        
        logger.info("="*70)
        logger.info("Pipeline completed")
        logger.info("="*70)
        
        return results
    
    def _get_hypotheses(
        self,
        image_path: str,
        clinical_query: str,
        n_hypotheses: int,
        use_cached: bool
    ) -> List[MedicalHypothesis]:
        """Generate or load cached hypotheses"""
        
        image_path = Path(image_path)
        cache_path = Path("data/hypotheses") / f"{image_path.stem}.json"
        
        if use_cached and cache_path.exists():
            logger.info(f"Using cached hypotheses: {cache_path}")
            return self.hypothesis_generator.load_hypotheses(cache_path)
        
        logger.info("Generating medical hypotheses...")
        hypotheses = self.hypothesis_generator.generate_hypotheses(
            image_path=str(image_path),
            clinical_query=clinical_query,
            n_hypotheses=n_hypotheses
        )
        
        # Cache hypotheses
        self.hypothesis_generator.save_hypotheses(hypotheses, cache_path)
        
        return hypotheses
    
    def _prepare_evidence(
        self,
        aggregation: Dict[str, Any],
        hypothesis_evidence: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare evidence for answer generation"""
        
        evidence = []
        
        # Add primary diagnosis evidence
        primary = aggregation["primary_diagnosis"]
        supporting_cases = aggregation.get("supporting_cases", [])
        
        for case in supporting_cases:
            evidence.append({
                "source": "knowledge_base",
                "diagnosis": primary,
                "score": case.get("combined_score", 0.0),
                "content": case.get("description", ""),
                "metadata": case
            })
        
        # Add differential diagnosis evidence
        for diff in aggregation.get("differential", []):
            diag = diff["diagnosis"]
            if diag in hypothesis_evidence:
                cases = hypothesis_evidence[diag]["cases"][:2]
                for case in cases:
                    evidence.append({
                        "source": "knowledge_base",
                        "diagnosis": diag,
                        "score": case.get("combined_score", 0.0),
                        "content": case.get("description", ""),
                        "metadata": case
                    })
        
        return evidence
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print results"""
        
        print("\n" + "="*70)
        print("HYPOTHESIS-BASED RAG RESULTS")
        print("="*70)
        
        print(f"\nImage: {results['image_path']}")
        print(f"Query: {results['user_query']}")
        
        # Hypotheses
        print(f"\nGenerated Hypotheses:")
        for i, h in enumerate(results.get("hypotheses", []), 1):
            print(f"  {i}. {h['diagnosis']} (urgency: {h['urgency']})")
            print(f"     Mechanism: {h['mechanism']}")
        
        # Retrieval aggregation
        if "retrieval" in results:
            agg = results["retrieval"]["aggregation"]
            print(f"\nDiagnostic Consensus:")
            print(f"  Primary: {agg['primary_diagnosis']}")
            print(f"  Confidence: {agg['confidence']:.2f}")
            print(f"  Supporting cases: {agg['case_count']}")
            
            if agg.get("differential"):
                print(f"\n  Differential diagnoses:")
                for diff in agg["differential"]:
                    print(
                        f"    - {diff['diagnosis']} "
                        f"(probability: {diff['probability']:.2f})"
                    )
        
        # Generated answer
        if "generation" in results:
            gen = results["generation"]
            print(f"\nGenerated Answer:")
            print(f"  Confidence: {gen['confidence']:.2f}")
            print(f"  {gen['answer'][:200]}...")
        
        # Evaluation
        if "evaluation" in results:
            print(f"\nEvaluation Scores:")
            eval_res = results["evaluation"]["answer_quality"]
            for metric_type, scores in eval_res.items():
                if isinstance(scores, dict) and "error" not in scores:
                    print(f"\n  {metric_type}:")
                    for sub_metric, score in list(scores.items())[:3]:
                        print(f"    {sub_metric}: {score:.4f}")
        
        print("\n" + "="*70 + "\n")