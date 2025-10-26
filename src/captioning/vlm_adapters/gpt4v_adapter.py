from .base_adapter import BaseVLMAdapter
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

class GPT4VAdapter(BaseVLMAdapter):
    """Adapter for OpenAI GPT-4o / GPT-4V Vision models."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__("GPT-4V")
        load_dotenv()
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_caption(self, image_url: str, prompt: str, seed: int = 0) -> str:
        """Generate caption using OpenAI GPT-4o / GPT-4V with a public image URL."""
        base64_image = self.encode_image_to_base64(image_url)
        response = self.client.chat.completions.create(
            model=self.model_name,
                        messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
