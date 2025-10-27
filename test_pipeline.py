from src.pipeline import RAGPipeline
from pathlib import Path


def test_pipeline():
    """Test the complete pipeline with sample data"""
    
    # Test configuration
    image_path = "data/images/cyanosis_Image_1.jpg"
    user_query = "What medical condition is visible in these fingers?"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f" Test image not found: {image_path}")
        print("Please ensure the image exists in the data/images directory")
        return False
    
    print(" Testing Dynamic Multimodal RAG Pipeline")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(device="cpu")  # Use CPU for testing
        
        # Run with minimal settings for quick test
        results = pipeline.run(
            image_path=image_path,
            user_query=user_query,
            n_prompts=2,  # Fewer prompts for testing
            n_seeds=1,    # Single seed for testing
            top_k=3,      # Fewer neighbors
            use_cached_captions=False  # Generate fresh
        )
        
        # Display results
        pipeline.print_results(results)
        
        # Verify results structure
        assert "neighbors" in results
        assert len(results["neighbors"]) > 0
        assert "clinical_query" in results
        
        print("✅ Pipeline test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)
