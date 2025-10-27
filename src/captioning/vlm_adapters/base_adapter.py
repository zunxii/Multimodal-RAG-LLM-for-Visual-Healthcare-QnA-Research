from abc import ABC, abstractmethod

class BaseVLMAdapter(ABC):
    """
    Abstract base class for Vision-Language Model (VLM) adapters.
    Each adapter implements the M_j(I; πk, s) function from the paper.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_caption(self, image_path: str, prompt: str, seed: int = 0) -> str:
        """
        Generate a caption for an image given a textual prompt and seed.
        
        Args:
            image_path: Path to the input image
            prompt: Textual prompt template
            seed: Random seed for reproducibility
            
        Returns:
            Generated caption text
        """
        pass
