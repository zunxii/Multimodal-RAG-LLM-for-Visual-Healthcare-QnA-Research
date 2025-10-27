import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .prompt_bank import PromptBank
from .vlm_adapters import GPT4VAdapter, GeminiAdapter
from ..utils import get_logger

logger = get_logger(__name__)


class DynamicCaptionCloud:
    """
    Builds dynamic caption cloud C(I)^dyn by sampling across
    multiple VLMs (M), prompts (Π), and random seeds (S).
    
    Implements Algorithm 1 lines 3-11 from the paper.
    """

    def __init__(self, output_dir: str = "data/captions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_bank = PromptBank()
        
        # Initialize VLM ensemble M = {M1, M2, ...}
        self.models = [
            GPT4VAdapter(),
            GeminiAdapter()
        ]
        
        logger.info(f"Initialized DynamicCaptionCloud with {len(self.models)} VLMs")

    def build_cloud(
        self, 
        image_path: str, 
        n_prompts: int = 4, 
        n_seeds: int = 2
    ) -> Path:
        """
        Generate caption cloud for a single image.
        
        Implements: C(I)^dyn = ∪(j=1..V) ∪(k=1..P) ∪(s=1..S) {c_j,k,s}
        where c_j,k,s = M_j(I; πk, s)
        
        Args:
            image_path: Path to input image I
            n_prompts: Number of prompts to sample per VLM
            n_seeds: Number of random seeds per prompt
            
        Returns:
            Path to saved caption cloud JSON
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        all_captions: List[Dict[str, Any]] = []
        
        logger.info(f"Building caption cloud for {image_path.name}")
        logger.info(f"Configuration: {len(self.models)} VLMs × {n_prompts} prompts × {n_seeds} seeds")
        
        # Iterate over VLMs (models)
        for model in tqdm(self.models, desc="VLMs"):
            # Sample prompts for this model
            for prompt_idx in range(n_prompts):
                prompt = self.prompt_bank.sample_prompt()
                
                # Generate with different seeds
                for seed in range(n_seeds):
                    try:
                        caption_text = model.generate_caption(
                            str(image_path), 
                            prompt, 
                            seed
                        )
                        
                        # Store with full provenance (meta_i)
                        all_captions.append({
                            "text": caption_text,
                            "model": model.name,
                            "prompt": prompt,
                            "seed": seed
                        })
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to generate caption with {model.name} "
                            f"(prompt_idx={prompt_idx}, seed={seed}): {e}"
                        )
        
        # Save caption cloud
        output_path = self.output_dir / f"{image_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_captions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(all_captions)} captions → {output_path}")
        return output_path
