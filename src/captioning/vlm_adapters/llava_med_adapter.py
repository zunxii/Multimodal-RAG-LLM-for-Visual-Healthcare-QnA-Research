from .base_adapter import BaseVLMAdapter
from ...api.kaggle_client import KaggleInferenceClient
from typing import List, Dict
import os

class LLaVAMedAdapter(BaseVLMAdapter):
    """Adapter for LLaVA-Med via Kaggle"""
    
    def __init__(self, mode: str = "kaggle"):
        super().__init__(model_name="llava-med")
        self.mode = mode
        
        if mode == "kaggle":
            endpoint = os.getenv("KAGGLE_ENDPOINT", "http://localhost:5000")
            api_key = os.getenv("KAGGLE_API_KEY")
            self.client = KaggleInferenceClient(endpoint, api_key)
        elif mode == "local":
            # For future local GPU use
            self._load_local_model()
    
    def generate_caption(self, image_path: str, prompt: str, 
                        temperature: float = 0.7) -> Dict:
        """Generate single caption"""
        if self.mode == "kaggle":
            result = self.client.generate_caption(image_path, prompt)
            return {
                "caption": result["caption"],
                "model": "llava-med",
                "prompt": prompt,
                "metadata": result.get("metadata", {})
            }
        else:
            return self._generate_local(image_path, prompt)
    
    def batch_generate(self, image_path: str, prompts: List[str]) -> List[Dict]:
        """Batch generation for efficiency"""
        if self.mode == "kaggle":
            captions = self.client.batch_generate(image_path, prompts)
            return [
                {
                    "caption": cap,
                    "model": "llava-med",
                    "prompt": prompts[i],
                    "metadata": {}
                }
                for i, cap in enumerate(captions)
            ]
        else:
            return [self._generate_local(image_path, p) for p in prompts]
    
    def _load_local_model(self):
        """Placeholder for future local deployment"""
        raise NotImplementedError("Local mode not yet implemented")