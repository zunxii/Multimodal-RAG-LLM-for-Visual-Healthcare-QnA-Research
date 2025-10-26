from .base_adapter import BaseVLMAdapter
import google.generativeai as genai
import os
from dotenv import load_dotenv

class GeminiAdapter(BaseVLMAdapter):
    """Adapter for Google Gemini 2.5 Vision."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__("Gemini-2.5")
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_caption(self, image_path: str, prompt: str, seed: int = 0) -> str:
        """Generate caption using Gemini 2.5 Vision."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = self.model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt,
            ]
        )
        return response.text.strip()
