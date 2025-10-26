from abc import ABC, abstractmethod

class BaseVLMAdapter(ABC):
    """
    Abstract base class for Vision-Language Model (VLM) adapters.
    Each adapter must implement the generate_caption method.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_caption(self, image_path: str, prompt: str, seed: int = 0) -> str:
        """
        Generate a caption for an image given a textual prompt and seed.
        Must return a plain text caption.
        """
        pass
