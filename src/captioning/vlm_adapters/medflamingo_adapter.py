from .base_adapter import BaseVLMAdapter
import random

class MedFlamingoAdapter(BaseVLMAdapter):
    def __init__(self):
        super().__init__("Med-Flamingo")

    def generate_caption(self, image_path: str, prompt: str, seed: int = 0) -> str:
        random.seed(seed)
        # TODO: connect to Med-Flamingo checkpoint
        return f"[Med-Flamingo] Caption for {image_path} using prompt '{prompt}'"
