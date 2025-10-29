"""
Script to build knowledge base from ClipSyntel dataset.
Run this once to create the static KB for retrieval.
"""

import argparse
from pathlib import Path

from src.pipeline import CompleteRAGPipeline
from src.utils import get_logger

logger = get_logger("build_kb")


def main():
    """Build knowledge base from ClipSyntel dataset"""
    
    parser = argparse.ArgumentParser(
        description="Build Medical Knowledge Base from ClipSyntel"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to ClipSyntel dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/knowledge_base",
        help="Output directory for knowledge base"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="clipsyntel_kb",
        help="Name for the knowledge base"
    )
    parser.add_argument(
        "--use_caption_cloud",
        action="store_true",
        help="Generate caption clouds for KB images (slower but better quality)"
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=3,
        help="Number of prompts per VLM (if using caption cloud)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=2,
        help="Number of seeds per prompt (if using caption cloud)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    logger.info("="*70)
    logger.info("BUILDING MEDICAL KNOWLEDGE BASE")
    logger.info("="*70)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {args.output_dir}/{args.name}")
    logger.info(f"Use caption cloud: {args.use_caption_cloud}")
    
    # Initialize pipeline
    logger.info("\nInitializing pipeline...")
    pipeline = CompleteRAGPipeline(
        kb_path=args.output_dir,
        device=args.device,
        use_kb=True
    )
    
    # Build knowledge base
    try:
        pipeline.build_knowledge_base(
            dataset_path=str(dataset_path),
            save_name=args.name,
            use_caption_cloud=args.use_caption_cloud
        )
        
        logger.info("="*70)
        logger.info("KNOWLEDGE BASE BUILT SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Saved to: {args.output_dir}/{args.name}")
        logger.info("\nYou can now use this KB with:")
        logger.info(f"  python main_complete.py --use_kb --load_kb {args.name} ...")
        
    except Exception as e:
        logger.error(f"Failed to build knowledge base: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()