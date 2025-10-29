"""
Test script for complete RAG pipeline with evaluation.
"""

from pathlib import Path
from src.pipeline import CompleteRAGPipeline
from src.utils import get_logger

logger = get_logger("test_complete")


def test_dynamic_only():
    """Test with dynamic caption cloud only (no KB)"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Dynamic Caption Cloud Only")
    logger.info("="*70)
    
    image_path = "data/images/cyanosis_Image_1.jpg"
    user_query = "What medical condition is visible in these fingers?"
    ground_truth = (
        "The image shows cyanosis, characterized by bluish discoloration of the "
        "fingertips, which indicates poor circulation or inadequate oxygenation "
        "of the blood. This can be associated with conditions like Raynaud's "
        "phenomenon or cardiovascular issues."
    )
    
    if not Path(image_path).exists():
        logger.error(f"Test image not found: {image_path}")
        return False
    
    try:
        # Initialize pipeline (no KB)
        pipeline = CompleteRAGPipeline(
            use_kb=False,
            device="cpu"
        )
        
        # Run pipeline
        results = pipeline.run(
            image_path=image_path,
            user_query=user_query,
            ground_truth_answer=ground_truth,
            n_prompts=2,
            n_seeds=1,
            final_top_k=5,
            use_cached_captions=True,
            generate_answer=True,
            evaluate=True
        )
        
        # Print results
        pipeline.print_results(results)
        
        # Verify structure
        assert "retrieval" in results
        assert "generation" in results
        assert "evaluation" in results
        
        logger.info("✅ Dynamic-only test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic-only test FAILED: {e}", exc_info=True)
        return False


def test_with_kb():
    """Test with hybrid retrieval (KB + dynamic)"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Hybrid Retrieval (KB + Dynamic)")
    logger.info("="*70)
    
    # This test requires a pre-built knowledge base
    kb_name = "clipsyntel_kb"
    kb_path = "data/knowledge_base"
    
    kb_index = Path(kb_path) / f"{kb_name}.index"
    
    if not kb_index.exists():
        logger.warning(
            f"Knowledge base not found: {kb_index}\n"
            "Skipping KB test. Build KB first with:\n"
            "  python build_kb.py --dataset path/to/clipsyntel.json"
        )
        return True  # Skip, not fail
    
    image_path = "data/images/cyanosis_Image_1.jpg"
    user_query = "What condition is shown in this image?"
    ground_truth = "Cyanosis with bluish discoloration of fingertips."
    
    try:
        # Initialize pipeline with KB
        pipeline = CompleteRAGPipeline(
            kb_path=kb_path,
            use_kb=True,
            device="cpu"
        )
        
        # Load KB
        pipeline.load_knowledge_base(save_name=kb_name)
        
        # Run pipeline
        results = pipeline.run(
            image_path=image_path,
            user_query=user_query,
            ground_truth_answer=ground_truth,
            n_prompts=2,
            n_seeds=1,
            kb_top_k=3,
            dynamic_top_k=3,
            final_top_k=5,
            use_cached_captions=True,
            generate_answer=True,
            evaluate=True
        )
        
        # Print results
        pipeline.print_results(results)
        
        # Verify structure
        assert "retrieval" in results
        assert results["retrieval"]["retrieval_type"] == "hybrid"
        assert "kb_results" in results["retrieval"]
        assert "dynamic_results" in results["retrieval"]
        assert "generation" in results
        assert "evaluation" in results
        
        logger.info("✅ Hybrid retrieval test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid retrieval test FAILED: {e}", exc_info=True)
        return False


def test_batch_evaluation():
    """Test batch evaluation on multiple images"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Batch Evaluation")
    logger.info("="*70)
    
    test_cases = [
        {
            "image_path": "data/images/cyanosis_Image_1.jpg",
            "query": "What condition is shown?",
            "ground_truth": "Cyanosis with bluish fingertips indicating poor oxygenation."
        }
        # Add more test cases here
    ]
    
    # Filter for existing images
    valid_cases = [
        case for case in test_cases 
        if Path(case["image_path"]).exists()
    ]
    
    if not valid_cases:
        logger.warning("No valid test images found. Skipping batch test.")
        return True
    
    try:
        # Initialize pipeline
        pipeline = CompleteRAGPipeline(
            use_kb=False,
            device="cpu"
        )
        
        # Run batch evaluation
        batch_results = pipeline.evaluate_batch(
            test_cases=valid_cases,
            output_path="results/batch_evaluation.json"
        )
        
        # Verify structure
        assert "aggregated_evaluation" in batch_results
        assert "per_case_results" in batch_results
        
        logger.info("✅ Batch evaluation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch evaluation test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPLETE RAG PIPELINE TESTS")
    print("="*70)
    
    tests = [
        ("Dynamic Only", test_dynamic_only),
        ("Hybrid (KB + Dynamic)", test_with_kb),
        ("Batch Evaluation", test_batch_evaluation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = "CRASHED"
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, status in results.items():
        symbol = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
        print(f"{symbol} {test_name:30s}: {status}")
    
    print("="*70 + "\n")
    
    # Exit code
    if all(s == "PASSED" for s in results.values()):
        print("All tests passed! 🎉")
        exit(0)
    else:
        print("Some tests failed. See logs above.")
        exit(1)


if __name__ == "__main__":
    main()