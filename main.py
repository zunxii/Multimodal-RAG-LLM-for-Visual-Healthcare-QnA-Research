"""
Main entry point for the Dynamic Multimodal RAG-LLM system.
Demonstrates the complete pipeline from caption generation to retrieval.
"""

import argparse
from pathlib import Path

from src.pipeline import RAGPipeline
from src.utils import get_logger

logger = get_logger("main")


def main():
    """Run the complete pipeline"""
    
    # Configuration
    parser = argparse.ArgumentParser(
        description="Dynamic Multimodal RAG-LLM for Visual QnA"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/cyanosis_Image_1.jpg",
        help="Path to input medical image"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What condition is shown in this image?",
        help="User query about the image"
    )
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
        "--top_k",
        type=int,
        default=5,
        help="Number of neighbors to retrieve"
    )
    parser.add_argument(
        "--use_cached",
        action="store_true",
        help="Use cached caption cloud if available"
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
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        caption_output_dir="data/captions",
        device=args.device
    )
    
    # Run pipeline
    try:
        results = pipeline.run(
            image_path=str(image_path),
            user_query=args.query,
            n_prompts=args.n_prompts,
            n_seeds=args.n_seeds,
            top_k=args.top_k,
            use_cached_captions=args.use_cached
        )
        
        # Display results
        pipeline.print_results(results)
        
        # Optionally save results to JSON
        output_file = Path("results") / f"{image_path.stem}_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
