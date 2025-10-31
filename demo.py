"""
Simple demo script - minimal working example
Tests the core pipeline without full complexity
"""

import sys
from pathlib import Path
import pandas as pd

# Download NLTK data if needed
try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass


def simple_demo():
    """Run a simple end-to-end demo"""
    
    print("\n" + "="*70)
    print("SIMPLE MULTIMODAL RAG DEMO")
    print("="*70)
    
    # Step 1: Check dataset
    print("\n[1/5] Checking dataset...")
    dataset_path = Path("data/clipsyntel.csv")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure data/clipsyntel.csv exists")
        return False
    
    df = pd.read_csv(dataset_path)
    print(f"✓ Dataset loaded: {len(df)} cases")
    
    # Step 2: Find a test image
    print("\n[2/5] Finding test image...")
    test_image = None
    test_question = None
    
    for idx, row in df.head(20).iterrows():
        img_path = Path("data/images") / row['image_path']
        if img_path.exists():
            test_image = str(img_path)
            test_question = row.get('Question_summ', row.get('Question', 'What is shown?'))
            print(f"✓ Using: {img_path.name}")
            print(f"  Question: {test_question}")
            break
    
    if not test_image:
        print("❌ No valid test image found in first 20 rows")
        return False
    
    # Step 3: Initialize encoders
    print("\n[3/5] Initializing encoders...")
    try:
        from src.embeddings import CLIPImageEncoder, CLIPTextEncoder
        
        img_encoder = CLIPImageEncoder(device="cpu")
        txt_encoder = CLIPTextEncoder(device="cpu")
        print("✓ Encoders initialized")
    except Exception as e:
        print(f"❌ Failed to initialize encoders: {e}")
        return False
    
    # Step 4: Generate caption cloud (minimal)
    print("\n[4/5] Generating caption cloud...")
    try:
        from src.captioning import DynamicCaptionCloud
        
        caption_builder = DynamicCaptionCloud(output_dir="data/captions")
        caption_path = caption_builder.build_cloud(
            test_image,
            n_prompts=1,  # Minimal for speed
            n_seeds=1
        )
        
        import json
        with open(caption_path) as f:
            captions = json.load(f)
        
        print(f"✓ Generated {len(captions)} captions")
        print(f"  Sample: {captions[0]['text'][:100]}...")
    except Exception as e:
        print(f"❌ Caption generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Run retrieval
    print("\n[5/5] Running retrieval...")
    try:
        from src.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(device="cpu")
        
        results = pipeline.run(
            image_path=test_image,
            user_query=test_question,
            n_prompts=1,
            n_seeds=1,
            top_k=3,
            use_cached_captions=True
        )
        
        print(f"✓ Retrieval complete")
        print(f"  Clinical query: {results['clinical_query']}")
        print(f"  Retrieved: {len(results['neighbors'])} neighbors")
        
        if results['neighbors']:
            print(f"\n  Top result (score: {results['neighbors'][0]['score']:.4f}):")
            print(f"  {results['neighbors'][0]['caption'][:150]}...")
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nThe system is working! Next steps:")
    print("1. Build knowledge base: python build_kb.py")
    print("2. Run full pipeline: python main_complete.py --image <path> --query <text>")
    print("3. Check FIXES_APPLIED.md for detailed documentation")
    
    return True


def main():
    try:
        success = simple_demo()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())