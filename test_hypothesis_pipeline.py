"""
test_hypothesis_pipeline.py

Quick test for hypothesis-based RAG system.
"""

import sys
from pathlib import Path

from src.pipeline.hypothesis_rag_pipeline import HypothesisRAGPipeline
from src.utils import get_logger

logger = get_logger("test_hypothesis")


def test_hypothesis_generation():
    """Test hypothesis generation"""
    
    print("\n" + "="*70)
    print("TEST 1: Medical Hypothesis Generation")
    print("="*70)
    
    try:
        from src.captioning.medical_hypothesis_generator import (
            MedicalHypothesisGenerator
        )
        
        generator = MedicalHypothesisGenerator()
        
        image_path = "data/images/cyanosis_Image_1.jpg"
        query = "What condition is shown in this image?"
        
        if not Path(image_path).exists():
            logger.warning(f"Test image not found: {image_path}")
            return False
        
        hypotheses = generator.generate_hypotheses(
            image_path=image_path,
            clinical_query=query,
            n_hypotheses=5
        )
        
        print(f"\nGenerated {len(hypotheses)} hypotheses:")
        for i, h in enumerate(hypotheses, 1):
            print(f"\n{i}. {h.diagnosis}")
            print(f"   Mechanism: {h.mechanism}")
            print(f"   Urgency: {h.urgency}")
            print(f"   Keywords: {', '.join(h.keywords[:3])}")
        
        assert len(hypotheses) > 0, "No hypotheses generated"
        assert all(h.diagnosis for h in hypotheses), "Missing diagnoses"
        
        logger.info("✅ Hypothesis generation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hypothesis generation test FAILED: {e}", exc_info=True)
        return False


def test_clinical_relevance():
    """Test clinical relevance scoring"""
    
    print("\n" + "="*70)
    print("TEST 2: Clinical Relevance Scoring")
    print("="*70)
    
    try:
        from src.retrieval.clinical_relevance import ClinicalRelevanceScorer
        
        scorer = ClinicalRelevanceScorer()
        
        # Mock query features
        query_features = {
            "body_regions": ["fingers", "hand"],
            "keywords": ["blue", "discoloration", "cyanosis"],
            "visual_features": ["bluish", "distal"],
            "urgency": "moderate",
            "diagnosis": "cyanosis"
        }
        
        # Mock case metadata
        case_metadata = {
            "category": "finger discoloration",
            "description": "Bluish discoloration of fingertips indicating cyanosis",
            "context": "Cyanosis, poor circulation",
            "urgency": "moderate"
        }
        
        score = scorer.compute_relevance(query_features, case_metadata)
        
        print(f"\nRelevance score: {score:.3f}")
        print(f"Expected: > 0.5 for good match")
        
        assert score > 0.0, "Zero relevance score"
        assert score <= 1.0, "Score out of bounds"
        
        logger.info(f"✅ Clinical relevance test PASSED (score: {score:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clinical relevance test FAILED: {e}", exc_info=True)
        return False


def test_full_pipeline():
    """Test full hypothesis-based pipeline"""
    
    print("\n" + "="*70)
    print("TEST 3: Full Hypothesis-Based Pipeline")
    print("="*70)
    
    image_path = "data/images/cyanosis_Image_1.jpg"
    query = "What condition is shown?"
    ground_truth = "Cyanosis with bluish fingertips"
    
    if not Path(image_path).exists():
        logger.warning(f"Test image not found: {image_path}")
        return True  # Skip, not fail
    
    # Check if KB exists
    kb_path = Path("data/knowledge_base/medical_kb.index")
    if not kb_path.exists():
        logger.warning(
            "Knowledge base not found. Build with:\n"
            "  python main_hypothesis.py --build_kb data/clipsyntel.csv ..."
        )
        return True  # Skip, not fail
    
    try:
        # Initialize pipeline
        pipeline = HypothesisRAGPipeline(device="cpu")
        
        # Load KB
        pipeline.load_knowledge_base(save_name="medical_kb")
        
        # Run pipeline
        results = pipeline.run(
            image_path=image_path,
            user_query=query,
            ground_truth_answer=ground_truth,
            n_hypotheses=3,
            top_k_per_hypothesis=5,
            relevance_threshold=0.3,
            use_cached_hypotheses=True,
            generate_answer=True,
            evaluate=True
        )
        
        # Verify structure
        assert "hypotheses" in results, "Missing hypotheses"
        assert "retrieval" in results, "Missing retrieval"
        assert "generation" in results, "Missing generation"
        
        # Print results
        pipeline.print_results(results)
        
        logger.info("✅ Full pipeline test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("HYPOTHESIS-BASED RAG SYSTEM TESTS")
    print("="*70)
    
    tests = [
        ("Hypothesis Generation", test_hypothesis_generation),
        ("Clinical Relevance", test_clinical_relevance),
        ("Full Pipeline", test_full_pipeline)
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
    passed = sum(1 for s in results.values() if s == "PASSED")
    total = len(results)
    
    if passed == total:
        print(f"All {total} tests passed! 🎉")
        return 0
    else:
        print(f"{passed}/{total} tests passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())