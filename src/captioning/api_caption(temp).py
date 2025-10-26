import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from prompt_bank import PromptBank
from vlm_adapters.gpt4v_adapter import GPT4VAdapter
from vlm_adapters.gemini_adapter import GeminiAdapter


class DynamicCaptionCloud:
    """
    Builds dynamic caption cloud C(I)^dyn by sampling across
    multiple VLMs, prompts, and random seeds.
    """

    def __init__(self, output_dir="data/captions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_bank = PromptBank()
        self.models = [
            GPT4VAdapter(),
            GeminiAdapter()
        ]

    def build_cloud(self, image_path: str, n_prompts: int = 4, n_seeds: int = 2):
        """
        Generate captions for the given image using multiple prompts and seeds.
        Only GPT-4V and Gemini adapters are active.
        """
        all_captions = []

        for model in self.models:
            for _ in range(n_prompts):
                prompt = self.prompt_bank.sample_prompt()
                for seed in range(n_seeds):
                    try:
                        caption = model.generate_caption(image_path, prompt, seed)
                        all_captions.append({
                            "text": caption,
                            "model": model.name,
                            "prompt": prompt,
                            "seed": seed
                        })
                    except Exception as e:
                        print(f"[ERROR] {model.name} failed for seed {seed}: {e}")

        out_path = self.output_dir / f"{Path(image_path).stem}.json"
        with open(out_path, "w") as f:
            json.dump(all_captions, f, indent=2)

        return out_path


if __name__ == "__main__":
    cloud = DynamicCaptionCloud()
    result = cloud.build_cloud("data/images/cyanosis_Image_1.jpg", n_prompts=3, n_seeds=2)
    print(f" Caption cloud saved to: {result}")
