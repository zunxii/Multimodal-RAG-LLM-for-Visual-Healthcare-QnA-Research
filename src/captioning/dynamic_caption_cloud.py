import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

from prompt_bank import PromptBank
# from vlm_adapters.llava_med_adapter import LLaVAMedAdapter
from vlm_adapters.gpt4v_adapter import GPT4VAdapter
# from vlm_adapters.medflamingo_adapter import MedFlamingoAdapter
from vlm_adapters.gemini_adapter import GeminiAdapter

class DynamicCaptionCloud:
    """
    Builds dynamic caption cloud C(I)^dyn by sampling across
    multiple VLMs, prompts, and random seeds.
    """

    def __init__(self, output_dir="data/captions", embed_model="all-MiniLM-L6-v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_bank = PromptBank()
        self.models = [
            # LLaVAMedAdapter(),
            GPT4VAdapter(),
            # MedFlamingoAdapter(),
            GeminiAdapter()
        ]
        # self.embedder = SentenceTransformer(embed_model)

    def build_cloud(self, image_path: str, n_prompts: int = 4, n_seeds: int = 2, dedup_threshold: float = 0.9):
        all_captions = []

        for model in self.models:
            for _ in range(n_prompts):
                prompt = self.prompt_bank.sample_prompt()
                for seed in range(n_seeds):
                    caption = model.generate_caption(image_path, prompt, seed)
                    all_captions.append({
                        "text": caption,
                        "model": model.name,
                        "prompt": prompt,
                        "seed": seed
                    })

        deduped = self._deduplicate(all_captions, threshold=dedup_threshold)
        out_path = self.output_dir / f"{Path(image_path).stem}.json"
        with open(out_path, "w") as f:
            json.dump(deduped, f, indent=2)
        return out_path

    def _deduplicate(self, captions, threshold=0.9):
        texts = [c["text"] for c in captions]
        emb = self.embedder.encode(texts, normalize_embeddings=True)
        kept_indices = []
        for i in range(len(emb)):
            if not any(np.dot(emb[i], emb[j]) > threshold for j in kept_indices):
                kept_indices.append(i)
        return [captions[i] for i in kept_indices]


if __name__ == "__main__":
    cloud = DynamicCaptionCloud()
    result = cloud.build_cloud("data/images/cyanosis_Image_1.jpg", n_prompts=3, n_seeds=2)
    print(f"Caption cloud saved to: {result}")
