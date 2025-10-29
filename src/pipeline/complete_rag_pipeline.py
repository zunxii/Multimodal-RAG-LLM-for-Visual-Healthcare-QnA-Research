"""
Complete RAG Pipeline with:
1. Knowledge Base Retrieval
2. Dynamic Caption Cloud Retrieval
3. Answer Generation
4. Evaluation (BLEU, ROUGE, BERTScore)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..captioning import DynamicCaptionCloud
from ..embeddings import CLIPImageEncoder, CLIPTextEncoder
from ..fusion import FusionMLP
from ..clinicalization import clinicalize_query
from ..retrieval import MultimodalRetriever
from ..retrieval.knowledge_base import MedicalKnowledgeBase
from ..retrieval.hybrid_retriever import HybridRetriever
from ..generator import AnswerGenerator
from ..evaluation import RAGEvaluator, RetrievalEvaluator
from ..utils import get_logger

logger = get_logger(__name__)


class CompleteRAGPipeline:
    """
    End-to-end RAG pipeline with evaluation.
    
    Features:
    - Build/load knowledge base from ClipSyntel
    - Generate dynamic caption clouds
    - Hybrid retrieval (KB + dynamic)
    - LLM answer generation
    - Multi-metric evaluation
    """
    
    def __init__(
        self,
        kb_path: str = "data/knowledge_base",
        caption_output_dir: str = "data/captions",
        device: str = None,
        use_kb: bool = True
    ):
        """
        Initialize complete RAG pipeline.
        
        Args:
            kb_path: Path to knowledge base
            caption_output_dir: Directory for caption clouds
            device: Torch device
            use_kb: Whether to use knowledge base retrieval
        """
        self.device = device
        self.use_kb = use_kb
        
        # Initialize encoders and fusion
        logger.info("Initializing encoders and fusion model...")
        self.image_encoder = CLIPImageEncoder(device=device)
        self.text_encoder = CLIPTextEncoder(device=device)
        
        dv, dt = 512, 512
        self.fusion_model = FusionMLP(dv=dv, dt=dt, d_out=512)
        
        # Initialize caption cloud builder
        self.caption_cloud_builder = DynamicCaptionCloud(
            output_dir=caption_output_dir
        )
        
        # Initialize dynamic retriever
        self.dynamic_retriever = MultimodalRetriever(
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
            fusion_model=self.fusion_model,
            device=device
        )
        
        # Initialize knowledge base (if using)
        self.knowledge_base = None
        self.hybrid_retriever = None
        
        if use_kb:
            logger.info("Initializing knowledge base...")
            self.knowledge_base = MedicalKnowledgeBase(
                kb_path=kb_path,
                image_encoder=self.image_encoder,
                text_encoder=self.text_encoder,
                fusion_model=self.fusion_model,
                device=device
            )
            
            self.hybrid_retriever = HybridRetriever(
                knowledge_base=self.knowledge_base,
                dynamic_retriever=self.dynamic_retriever,
                device=device
            )
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator()
        
        # Initialize evaluators
        self.rag_evaluator = RAGEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()
        
        logger.info("CompleteRAGPipeline initialized")
    
    def build_knowledge_base(
        self,
        dataset_path: str,
        save_name: str = "clipsyntel_kb",
        use_caption_cloud: bool = False
    ) -> None:
        """
        Build knowledge base from ClipSyntel dataset.
        
        Args:
            dataset_path: Path to ClipSyntel JSON
            save_name: Name for KB files
            use_caption_cloud: Generate caption clouds for KB images
        """
        if not self.use_kb or self.knowledge_base is None:
            raise RuntimeError("Knowledge base not enabled. Set use_kb=True")
        
        self.knowledge_base.build_from_clipsyntel(
            dataset_path=dataset_path,
            save_name=save_name,
            use_caption_cloud=use_caption_cloud
        )
    
    def load_knowledge_base(self, save_name: str = "clipsyntel_kb") -> None:
        """Load existing knowledge base."""
        if not self.use_kb or self.knowledge_base is None:
            raise RuntimeError("Knowledge base not enabled. Set use_kb=True")
        
        self.knowledge_base.load_kb(save_name=save_name)
    
    def run(
        self,
        image_path: str,
        user_query: str,
        ground_truth_answer: str = None,
        n_prompts: int = 4,
        n_seeds: int = 2,
        kb_top_k: int = 5,
        dynamic_top_k: int = 5,
        final_top_k: int = 10,
        use_cached_captions: bool = True,
        generate_answer: bool = True,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete RAG pipeline with evaluation.
        
        Args:
            image_path: Query image path
            user_query: User's question
            ground_truth_answer: Reference answer for evaluation
            n_prompts: Prompts per VLM for caption cloud
            n_seeds: Seeds per prompt
            kb_top_k: Top-K from KB
            dynamic_top_k: Top-K from dynamic captions
            final_top_k: Final top-K after merging
            use_cached_captions: Use cached caption cloud
            generate_answer: Whether to generate answer
            evaluate: Whether to run evaluation
            
        Returns:
            Complete results dictionary
        """
        logger.info("="*70)
        logger.info("COMPLETE RAG PIPELINE")
        logger.info("="*70)
        logger.info(f"Image: {image_path}")
        logger.info(f"Query: {user_query}")
        
        results = {
            "image_path": str(image_path),
            "user_query": user_query,
            "pipeline_config": {
                "use_kb": self.use_kb,
                "n_prompts": n_prompts,
                "n_seeds": n_seeds,
                "kb_top_k": kb_top_k,
                "dynamic_top_k": dynamic_top_k,
                "final_top_k": final_top_k
            }
        }
        
        # Step 1: Generate/load caption cloud
        caption_cloud_path = self._get_caption_cloud(
            image_path=image_path,
            n_prompts=n_prompts,
            n_seeds=n_seeds,
            use_cached=use_cached_captions
        )
        
        with open(caption_cloud_path, 'r') as f:
            caption_cloud = json.load(f)
        
        results["caption_cloud_stats"] = {
            "total_captions": len(caption_cloud),
            "vlm_models": list(set(c["model"] for c in caption_cloud)),
            "unique_prompts": len(set(c["prompt"] for c in caption_cloud))
        }
        
        # Step 2: Clinicalize query
        logger.info("Clinicalizing query...")
        q_clinical = clinicalize_query(user_query)
        results["clinical_query"] = q_clinical
        
        # Step 3: Retrieve evidence
        if self.use_kb and self.hybrid_retriever:
            # Hybrid retrieval
            logger.info("Performing hybrid retrieval (KB + Dynamic)...")
            retrieval_results = self.hybrid_retriever.retrieve(
                image_path=str(image_path),
                q_clinical=q_clinical,
                caption_cloud_path=str(caption_cloud_path),
                kb_top_k=kb_top_k,
                dynamic_top_k=dynamic_top_k,
                final_top_k=final_top_k
            )
            
            results["retrieval"] = {
                "kb_results": retrieval_results["kb_results"],
                "dynamic_results": retrieval_results["dynamic_results"],
                "merged_results": retrieval_results["merged_results"],
                "retrieval_type": "hybrid"
            }
            
            evidence_for_generation = retrieval_results["merged_results"]
        else:
            # Dynamic-only retrieval
            logger.info("Performing dynamic caption cloud retrieval...")
            self.dynamic_retriever.build_index(
                image_path=str(image_path),
                caption_cloud_path=str(caption_cloud_path)
            )
            
            dynamic_results = self.dynamic_retriever.retrieve(
                q_clinical=q_clinical,
                top_k=final_top_k
            )
            
            results["retrieval"] = {
                "dynamic_results": dynamic_results,
                "retrieval_type": "dynamic_only"
            }
            
            # Convert to evidence format
            evidence_for_generation = [
                {
                    "source": "dynamic_caption_cloud",
                    "score": r["score"],
                    "content": r["caption"],
                    "metadata": r["meta"]
                }
                for r in dynamic_results
            ]
        
        # Step 4: Generate answer
        if generate_answer:
            logger.info("Generating answer...")
            generation_result = self.answer_generator.generate_answer(
                query=q_clinical,
                retrieved_evidence=evidence_for_generation,
                image_path=str(image_path),
                include_provenance=True
            )
            
            results["generation"] = generation_result
            logger.info(f"Answer generated (confidence: {generation_result['confidence']:.2f})")
        
        # Step 5: Evaluate
        if evaluate and ground_truth_answer and generate_answer:
            logger.info("Running evaluation...")
            
            # Answer quality evaluation
            eval_results = self.rag_evaluator.evaluate(
                prediction=generation_result["answer"],
                reference=ground_truth_answer,
                metrics=["BLEU", "ROUGE", "BERTScore"]
            )
            
            results["evaluation"] = {
                "answer_quality": eval_results,
                "ground_truth": ground_truth_answer
            }
            
            # Log scores
            logger.info("Evaluation Scores:")
            for metric_type, scores in eval_results.items():
                if isinstance(scores, dict) and "error" not in scores:
                    for sub_metric, score in scores.items():
                        logger.info(f"  {metric_type}_{sub_metric}: {score:.4f}")
        
        logger.info("="*70)
        logger.info("Pipeline completed successfully")
        logger.info("="*70)
        
        return results
    
    def _get_caption_cloud(
        self,
        image_path: str,
        n_prompts: int,
        n_seeds: int,
        use_cached: bool
    ) -> Path:
        """Get or generate caption cloud."""
        image_path = Path(image_path)
        caption_cloud_path = (
            self.caption_cloud_builder.output_dir / f"{image_path.stem}.json"
        )
        
        if use_cached and caption_cloud_path.exists():
            logger.info(f"Using cached caption cloud: {caption_cloud_path}")
        else:
            logger.info("Generating dynamic caption cloud...")
            caption_cloud_path = self.caption_cloud_builder.build_cloud(
                str(image_path),
                n_prompts=n_prompts,
                n_seeds=n_seeds
            )
        
        return caption_cloud_path
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline on multiple test cases.
        
        Args:
            test_cases: List of dicts with 'image_path', 'query', 'ground_truth'
            output_path: Optional path to save results
            
        Returns:
            Aggregated evaluation results
        """
        logger.info(f"Evaluating pipeline on {len(test_cases)} test cases...")
        
        all_results = []
        predictions = []
        references = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\nProcessing test case {i}/{len(test_cases)}")
            
            result = self.run(
                image_path=test_case["image_path"],
                user_query=test_case["query"],
                ground_truth_answer=test_case.get("ground_truth"),
                generate_answer=True,
                evaluate=True
            )
            
            all_results.append(result)
            
            if "generation" in result:
                predictions.append(result["generation"]["answer"])
            if "ground_truth" in test_case:
                references.append(test_case["ground_truth"])
        
        # Aggregate evaluation
        if predictions and references:
            aggregated_eval = self.rag_evaluator.evaluate_batch(
                predictions=predictions,
                references=references
            )
        else:
            aggregated_eval = {}
        
        batch_results = {
            "num_cases": len(test_cases),
            "per_case_results": all_results,
            "aggregated_evaluation": aggregated_eval
        }
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch results saved to: {output_path}")
        
        # Print summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _print_batch_summary(self, batch_results: Dict[str, Any]) -> None:
        """Print batch evaluation summary."""
        print("\n" + "="*70)
        print("BATCH EVALUATION SUMMARY")
        print("="*70)
        print(f"Number of test cases: {batch_results['num_cases']}")
        
        if "aggregated_evaluation" in batch_results:
            agg = batch_results["aggregated_evaluation"].get("aggregated", {})
            if agg:
                print("\nAggregated Scores:")
                print("-"*70)
                for metric, score in sorted(agg.items()):
                    print(f"{metric:30s}: {score:.4f}")
        
        print("\n" + "="*70 + "\n")
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print pipeline results."""
        print("\n" + "="*70)
        print("RAG PIPELINE RESULTS")
        print("="*70)
        
        print(f"\n Image: {results['image_path']}")
        print(f"\n User Query: {results['user_query']}")
        print(f"\n Clinical Query: {results.get('clinical_query', 'N/A')}")
        
        # Caption cloud stats
        if "caption_cloud_stats" in results:
            stats = results["caption_cloud_stats"]
            print(f"\n Caption Cloud:")
            print(f"   • Total captions: {stats['total_captions']}")
            print(f"   • VLM models: {', '.join(stats['vlm_models'])}")
        
        # Retrieval results
        if "retrieval" in results:
            ret = results["retrieval"]
            print(f"\n Retrieval ({ret['retrieval_type']}):")
            
            if "kb_results" in ret:
                print(f"   • KB results: {len(ret['kb_results'])}")
            if "dynamic_results" in ret:
                print(f"   • Dynamic results: {len(ret['dynamic_results'])}")
            if "merged_results" in ret:
                print(f"   • Merged results: {len(ret['merged_results'])}")
                
                print(f"\n Top-3 Retrieved Evidence:")
                for i, item in enumerate(ret["merged_results"][:3], 1):
                    print(f"\n   [{i}] {item['source']} (score: {item['score']:.3f})")
                    print(f"       {item['content'][:150]}...")
        
        # Generated answer
        if "generation" in results:
            gen = results["generation"]
            print(f"\n Generated Answer:")
            print(f"   Confidence: {gen['confidence']:.2f}")
            print(f"   {gen['answer']}")
            
            if gen.get("provenance"):
                print(f"\n   Sources cited: {len(gen['provenance'])}")
        
        # Evaluation scores
        if "evaluation" in results:
            print(f"\n Evaluation Scores:")
            eval_res = results["evaluation"]["answer_quality"]
            
            for metric_type, scores in eval_res.items():
                if isinstance(scores, dict) and "error" not in scores:
                    print(f"\n   {metric_type}:")
                    for sub_metric, score in scores.items():
                        print(f"     {sub_metric:20s}: {score:.4f}")
        
        print("\n" + "="*70 + "\n")