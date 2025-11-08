"""
main_hypothesis.py

Main entry point for hypothesis-based medical RAG system.
"""

import argparse
from pathlib import Path

from src.pipeline.hypothesis_rag_pipeline import HypothesisRAGPipeline
from src.utils import get_logger

logger = get_logger("main_hypothesis")


def main():
    """Run hypothesis-based RAG pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Hypothesis-Based Medical RAG System"
    )
    
    # Input arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to medical image"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Clinical question"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Ground truth answer for evaluation"
    )
    
    # Knowledge base arguments
    parser.add_argument(
        "--kb_path",
        type=str,
        default="data/knowledge_base",
        help="Path to knowledge base"
    )
    parser.add_argument(
        "--load_kb",
        type=str,
        default=None,
        help="Load existing KB (name)"
    )
    parser.add_argument(
        "--build_kb",
        type=str,
        default=None,
        help="Build KB from dataset (path to CSV)"
    )
    
    # Hypothesis arguments
    parser.add_argument(
        "--n_hypotheses",
        type=int,
        default=5,
        help="Number of medical hypotheses to generate"
    )
    parser.add_argument(
        "--top_k_per_hypothesis",
        type=int,
        default=10,
        help="Top-K cases to retrieve per hypothesis"
    )
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=0.3,
        help="Minimum clinical relevance score"
    )
    parser.add_argument(
        "--use_cached",
        action="store_true",
        help="Use cached hypotheses if available"
    )
    
    # Pipeline control
    parser.add_argument(
        "--no_generation",
        action="store_true",
        help="Skip answer generation"
    )
    parser.add_argument(
        "--no_evaluation",
        action="store_true",
        help="Skip evaluation"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="results/hypothesis_rag_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Verify image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    # Initialize pipeline
    logger.info("Initializing Hypothesis-Based RAG Pipeline...")
    pipeline = HypothesisRAGPipeline(
        kb_path=args.kb_path,
        device=args.device
    )
    
    # Build or load knowledge base
    if args.build_kb:
        logger.info(f"Building knowledge base from {args.build_kb}...")
        pipeline.build_knowledge_base(
            dataset_path=args.build_kb,
            save_name="medical_kb",
            image_base_path="data/images"
        )
    
    if args.load_kb:
        logger.info(f"Loading knowledge base: {args.load_kb}...")
        pipeline.load_knowledge_base(save_name=args.load_kb)
    
    # Run pipeline
    try:
        results = pipeline.run(
            image_path=str(image_path),
            user_query=args.query,
            ground_truth_answer=args.ground_truth,
            n_hypotheses=args.n_hypotheses,
            top_k_per_hypothesis=args.top_k_per_hypothesis,
            relevance_threshold=args.relevance_threshold,
            use_cached_hypotheses=args.use_cached,
            generate_answer=not args.no_generation,
            evaluate=not args.no_evaluation and args.ground_truth is not None
        )
        
        # Print results
        pipeline.print_results(results)
        
        # Save results
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print evaluation summary if available
        if "evaluation" in results:
            print("\n" + "="*70)
            print("EVALUATION SUMMARY")
            print("="*70)
            
            eval_res = results["evaluation"]["answer_quality"]
            for metric_type, scores in eval_res.items():
                if isinstance(scores, dict) and "error" not in scores:
                    print(f"\n{metric_type}:")
                    for sub_metric, score in scores.items():
                        try:
                            print(f"  {sub_metric:20s}: {float(score):.4f}")
                        except:
                            print(f"  {sub_metric:20s}: {score}")
            
            print("\n" + "="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()