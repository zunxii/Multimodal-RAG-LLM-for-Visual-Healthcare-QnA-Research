import requests
import json
from typing import Dict, Any, Optional
import base64
from PIL import Image
import io

class KaggleInferenceClient:
    """Client to communicate with Kaggle notebook serving LLaVA-Med"""
    
    def __init__(self, kaggle_endpoint: str, api_key: Optional[str] = None):
        self.endpoint = kaggle_endpoint
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
    
    def generate_caption(self, image_path: str, prompt: str, 
                        model: str = "llava-med") -> Dict[str, Any]:
        """Generate caption using remote Kaggle GPU"""
        payload = {
            "image": self.encode_image(image_path),
            "prompt": prompt,
            "model": model
        }
        
        response = self.session.post(
            f"{self.endpoint}/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def batch_generate(self, image_path: str, prompts: list) -> list:
        """Batch caption generation"""
        payload = {
            "image": self.encode_image(image_path),
            "prompts": prompts,
            "model": "llava-med"
        }
        
        response = self.session.post(
            f"{self.endpoint}/batch_generate",
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        return response.json()["captions"]