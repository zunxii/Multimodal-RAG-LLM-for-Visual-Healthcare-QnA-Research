# Dynamic Multimodal RAG-LLM for Visual QnA in Healthcare

Implementation of the research proposal: "A Dynamic Caption-Cloud + Joint 
Multimodal-Embedding Retrieval Framework for Clinically-Grounded Visual Answers"

## Architecture Overview

This system implements a retrieval-first multimodal approach for clinical 
visual question answering with the following key innovations:

1. **Dynamic Caption Cloud Generation**: For each input image, generate 
   multiple diverse captions using multiple VLMs (GPT-4V, Gemini) with 
   various prompt templates and random seeds.

2. **Temporary Multimodal Index**: Create a personalized vector database 
   containing joint image+caption embeddings specific to the input image.

3. **Multimodal Retrieval**: Query the dynamic database using joint 
   image+clinicalized-query embeddings in a shared latent space.

## Project Structure

```
.
├── src/
│   ├── captioning/          # Caption generation module
│   │   ├── dynamic_caption_cloud.py
│   │   ├── prompt_bank.py
│   │   └── vlm_adapters/
│   │       ├── base_adapter.py
│   │       ├── gpt4v_adapter.py
│   │       └── gemini_adapter.py
│   ├── embeddings/          # Image and text encoders
│   │   ├── clip_image_encoder.py
│   │   └── clip_text_encoder.py
│   ├── fusion/              # Multimodal fusion
│   │   └── fusion_mlp.py
│   ├── clinicalization/     # Query normalization
│   │   └── clinicalize_query.py
│   ├── retrieval/           # Retrieval and indexing
│   │   └── multimodal_retriever.py
│   ├── pipeline/            # End-to-end pipeline
│   │   └── rag_pipeline.py
│   └── utils/               # Utility functions
│       ├── ensure_numpy_2d.py
│       ├── to_tensor.py
│       └── logger.py
├── data/
│   ├── images/              # Input medical images
│   └── captions/            # Generated caption clouds (JSON)
├── results/                 # Pipeline outputs
├── main.py                  # Main entry point
├── test_pipeline.py         # Test script
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Usage

### Basic Usage

```bash
python main.py \
    --image data/images/cyanosis_Image_1.jpg \
    --query "What condition is shown in this image?" \
    --n_prompts 3 \
    --n_seeds 2 \
    --top_k 5
```

### Using Cached Captions

```bash
python main.py \
    --image data/images/cyanosis_Image_1.jpg \
    --query "What medical condition is visible?" \
    --use_cached
```

### Programmatic Usage

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(device="cuda")

# Run pipeline
results = pipeline.run(
    image_path="data/images/sample.jpg",
    user_query="What is this condition?",
    n_prompts=4,
    n_seeds=2,
    top_k=5
)

# Display results
pipeline.print_results(results)
```

## Testing

Run the test script to verify installation:

```bash
python test_pipeline.py
```

## Key Components

### 1. Caption Cloud Generation
- Multiple VLMs (GPT-4V, Gemini)
- Diverse prompt templates
- Multiple random seeds for variation

### 2. Multimodal Embeddings
- CLIP image encoder (E_v)
- CLIP text encoder (E_t)
- MLP fusion module (F_m)

### 3. Temporary Index
- FAISS-based similarity search
- Image-specific caption vectors
- Cosine similarity retrieval

### 4. Query Processing
- LLM-based clinicalization
- Joint image+query embedding
- Top-K retrieval

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FAISS
- sentence-transformers
- OpenAI API key (for GPT-4V)
- Google API key (for Gemini)

## Future Work

- [ ] Implement answer generation module
- [ ] Add fusion strategies (centroid, atomic-fact)
- [ ] Implement grounding score calculation
- [ ] Add evaluation metrics
- [ ] Support for additional VLMs (LLaVA-Med, Med-Flamingo)
- [ ] Web interface

## Citation

```bibtex
@article{multimodal_rag_2025,
  title={Multimodal RAG-LLM for Visual QnA in Healthcare},
  author={Junaid, Faqre Alam},
  year={2025}
}
```

## License

MIT License
"""
