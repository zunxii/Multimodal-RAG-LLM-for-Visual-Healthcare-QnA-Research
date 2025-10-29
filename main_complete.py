"""
Main entry point for Complete RAG System with evaluation.
Supports both KB and dynamic caption cloud retrieval.
"""

import argparse
from pathlib import Path

from src.pipeline import CompleteRAGPipeline
from src.utils import get_logger

logger = get_logger("main_complete")


def main():
    """Run complete RAG pipeline with evaluation"""
    
    parser = argparse.ArgumentParser(
        description="Complete Dynamic Multimodal RAG-LLM with Evaluation"
    )
    
    # Input arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input medical image"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query about the image"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Ground truth answer for evaluation"
    )
    
    # Knowledge base arguments
    parser.add_argument(
        "--use_kb",
        action="store_true",
        help="Use knowledge base retrieval"
    )
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
        help="Build KB from ClipSyntel dataset (path to JSON)"
    )
    
    # Caption cloud arguments
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=3,
        help="Number of prompts per VLM"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=2,
        help="Number of seeds per prompt"
    )
    parser.add_argument(
        "--use_cached",
        action="store_true",
        help="Use cached caption cloud if available"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--kb_top_k",
        type=int,
        default=5,
        help="Top-K from knowledge base"
    )
    parser.add_argument(
        "--dynamic_top_k",
        type=int,
        default=5,
        help="Top-K from dynamic caption cloud"
    )
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=10,
        help="Final top-K after merging"
    )
    
    # Generation and evaluation
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
        default="results/complete_rag_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Verify image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    # Initialize pipeline
    logger.info("Initializing Complete RAG Pipeline...")
    pipeline = CompleteRAGPipeline(
        kb_path=args.kb_path,
        device=args.device,
        use_kb=args.use_kb
    )
    
    # Build or load knowledge base
    if args.build_kb:
        logger.info(f"Building knowledge base from {args.build_kb}...")
        pipeline.build_knowledge_base(
            dataset_path=args.build_kb,
            save_name="clipsyntel_kb",
            use_caption_cloud=False  # Set to True for better quality
        )
    
    if args.use_kb and args.load_kb:
        logger.info(f"Loading knowledge base: {args.load_kb}...")
        pipeline.load_knowledge_base(save_name=args.load_kb)
    
    # Run pipeline
    try:
        results = pipeline.run(
            image_path=str(image_path),
            user_query=args.query,
            ground_truth_answer=args.ground_truth,
            n_prompts=args.n_prompts,
            n_seeds=args.n_seeds,
            kb_top_k=args.kb_top_k,
            dynamic_top_k=args.dynamic_top_k,
            final_top_k=args.final_top_k,
            use_cached_captions=args.use_cached,
            generate_answer=not args.no_generation,
            evaluate=not args.no_evaluation and args.ground_truth is not None
        )
        
        # Display results
        pipeline.print_results(results)
        
        # Save results
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print evaluation summary
        if "evaluation" in results:
            print("\n" + "="*70)
            print("EVALUATION SUMMARY")
            print("="*70)
            
            eval_res = results["evaluation"]["answer_quality"]
            for metric_type, scores in eval_res.items():
                if isinstance(scores, dict) and "error" not in scores:
                    print(f"\n{metric_type}:")
                    for sub_metric, score in scores.items():
                        print(f"  {sub_metric:20s}: {score:.4f}")
            
            print("\n" + "="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()