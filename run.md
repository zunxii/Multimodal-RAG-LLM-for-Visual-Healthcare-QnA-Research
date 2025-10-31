# Multimodal RAG System - Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT-4V)
- Google API key (for Gemini)
- Dataset: `data/clipsyntel.csv` with images in `data/images/`

---

## Step-by-Step Setup

### 1. Install Dependencies (2 min)
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys (1 min)
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

### 3. Verify Everything Works (2 min)
```bash
python simple_demo.py
```

This will:
- ✓ Check dataset exists
- ✓ Find a test image
- ✓ Initialize encoders
- ✓ Generate captions
- ✓ Run retrieval

If you see "✅ DEMO COMPLETED SUCCESSFULLY!", you're good to go!

---

## Usage Examples

### Example 1: Dynamic Retrieval (No KB)
```bash
python main_complete.py \
  --image data/images/cyanosis_Image_1.jpg \
  --query "What medical condition is visible?" \
  --n_prompts 2 \
  --n_seeds 1 \
  --final_top_k 5
```

### Example 2: Build Knowledge Base
```bash
# Quick build (no caption clouds)
python build_kb.py \
  --dataset data/clipsyntel.csv \
  --images data/images \
  --name clipsyntel_kb

# High quality build (with caption clouds - slower)
python build_kb.py \
  --dataset data/clipsyntel.csv \
  --images data/images \
  --name clipsyntel_kb \
  --use_caption_cloud \
  --n_prompts 3 \
  --n_seeds 2
```

### Example 3: Hybrid Retrieval (KB + Dynamic)
```bash
python main_complete.py \
  --use_kb \
  --load_kb clipsyntel_kb \
  --image data/images/your_image.jpg \
  --query "Diagnose this condition" \
  --kb_top_k 5 \
  --dynamic_top_k 5 \
  --final_top_k 10
```

### Example 4: With Evaluation
```bash
python main_complete.py \
  --image data/images/test_image.jpg \
  --query "What condition is this?" \
  --ground_truth "Cyanosis with bluish discoloration" \
  --use_kb \
  --load_kb clipsyntel_kb
```

---

## Key Files Modified/Added

### ✨ New/Fixed Files:
1. **`src/retrieval/knowledge_base.py`** - CSV support, proper image paths
2. **`build_kb.py`** - Updated for CSV dataset
3. **`requirements.txt`** - Added pandas
4. **`simple_demo.py`** - Quick test script
5. **`quick_test.py`** - Comprehensive test suite

### 🗑️ Redundant Files (Can Remove):
- `src/retrieval/build_db.py`
- `src/retrieval/search_db.py`
- `src/retrieval/pipeline.py`
- `src/generator/evidence_generator.py`
- `src/clinicalization/llm_call.py`

---

## Common Commands

### Quick Test
```bash
python simple_demo.py
```

### Full Test Suite
```bash
python quick_test.py
```

### Build KB (Fast)
```bash
python build_kb.py --dataset data/clipsyntel.csv --images data/images
```

### Run Pipeline (Dynamic Only)
```bash
python main_complete.py --image <path> --query <text>
```

### Run Pipeline (With KB)
```bash
python main_complete.py --use_kb --load_kb clipsyntel_kb --image <path> --query <text>
```

---

## Troubleshooting

### Error: "Dataset not found"
**Fix:** Ensure `data/clipsyntel.csv` exists
```bash
ls -l data/clipsyntel.csv
```

### Error: "Image not found"
**Fix:** Check image paths in CSV match files in `data/images/`
```bash
# List images
ls data/images/

# Check CSV paths
python -c "import pandas as pd; df=pd.read_csv('data/clipsyntel.csv'); print(df['image_path'].head())"
```

### Error: "NLTK punkt_tab not found"
**Fix:** Download NLTK data
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
```

### Error: "API key not set"
**Fix:** Create `.env` file with your keys
```bash
echo "OPENAI_API_KEY=your-key" > .env
echo "GOOGLE_API_KEY=your-key" >> .env
```

### Error: "CUDA out of memory"
**Fix:** Use CPU instead
```bash
python main_complete.py --device cpu --image <path> --query <text>
```

---

## Understanding the Output

### Sample Output:
```
==================================================================
COMPLETE RAG PIPELINE
==================================================================
Image: data/images/cyanosis_Image_1.jpg
Query: What condition is shown?
Clinical Query: Differential diagnosis of cyanotic fingertips.

Caption Cloud:
  • Total captions: 12
  • VLM models: GPT-4V, Gemini-2.5

Retrieval (dynamic_only):
  • Dynamic results: 10

Top-3 Retrieved Evidence:

  [1] dynamic_caption_cloud (score: 0.995)
      Abnormal. The blue coloration of the fingertips suggests 
      cyanosis, which is an abnormal finding...

  [2] dynamic_caption_cloud (score: 0.994)
      The fingers appear to have a bluish discoloration, which 
      could suggest a circulation issue...

  [3] dynamic_caption_cloud (score: 0.994)
      The fingertips are showing signs of cyanosis, which is 
      the bluish discoloration of the skin...

Generated Answer:
  Confidence: 0.95
  The fingertips in the image show a bluish discoloration [Source 1, 
  Source 2, Source 3], suggesting cyanosis. Cyanosis is the bluish 
  discoloration of the skin due to poor circulation or inadequate 
  oxygenation of the blood [Source 1, Source 3]...

Evaluation Scores:
  ROUGE-1: 0.6234
  ROUGE-L: 0.5821
  BERTScore-F1: 0.8456
==================================================================
```

---

## Performance Tips

### For Speed:
- Use `--use_cached` to reuse caption clouds
- Set `--n_prompts 1 --n_seeds 1` for minimal VLM calls
- Use `--device cpu` if GPU causes issues
- Pre-build knowledge base once, reuse many times

### For Quality:
- Use `--use_caption_cloud` when building KB
- Increase `--n_prompts 4 --n_seeds 2` for caption diversity
- Use hybrid retrieval (`--use_kb`) for better evidence
- Increase `--final_top_k 15` for more context

---

## System Architecture

```
Medical Image → Caption Cloud (GPT-4V + Gemini) → Embeddings (CLIP)
       ↓                                               ↓
Clinical Query → Clinicalization (LLM) → Query Embedding
       ↓                                               ↓
Knowledge Base (Optional) ←──────────→ Dynamic Index
       ↓                                               ↓
    Hybrid Retrieval (Multimodal Fusion + FAISS)
       ↓
Retrieved Evidence (Top-K with scores)
       ↓
Answer Generation (LLM with Evidence Grounding)
       ↓
Evaluation (BLEU, ROUGE, BERTScore)
```

---

## Next Steps

1. ✅ Run `simple_demo.py` to verify setup
2. ✅ Build knowledge base with `build_kb.py`
3. ✅ Test on your medical images
4. ✅ Adjust hyperparameters for your use case
5. ✅ Add more VLM models for diversity
6. ✅ Implement custom evaluation metrics
7. ✅ Scale to larger datasets

---

## Support & Documentation

- **Detailed fixes**: See `FIXES_APPLIED.md`
- **Test suite**: Run `quick_test.py`
- **Simple demo**: Run `simple_demo.py`
- **Full pipeline**: Check `main_complete.py --help`
- **KB building**: Check `build_kb.py --help`

---

## Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] API keys set in `.env` file
- [ ] Dataset at `data/clipsyntel.csv`
- [ ] Images in `data/images/`
- [ ] `simple_demo.py` runs successfully
- [ ] Knowledge base built (optional)
- [ ] First retrieval works

If all checkboxes are ✓, you're ready to go! 🚀