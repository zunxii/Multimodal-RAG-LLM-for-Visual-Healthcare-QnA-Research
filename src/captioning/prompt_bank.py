import random
from typing import Optional, List

class PromptBank:
    """
    Manages prompt templates for caption generation across different categories.
    Implements the prompt template bank Π from the paper.
    """

    def __init__(self):
        self.prompts = {
            "clinical": [
                "Describe any visible abnormalities in this medical image.",
                "Identify potential pathologies or irregularities.",
                "Provide a detailed medical description including anatomical location, "
                "morphological characteristics, and associated features."
            ],
            "attribute": [
                "List key visual attributes like texture, opacity, and shape.",
                "Describe color, size, and structural characteristics in the image.",
                "What are the color, texture, and surface characteristics?"
            ],
            "question": [
                "What might explain the visible patterns in this scan?",
                "Are there signs of infection, lesion, or fracture?",
                "What are the key diagnostic features for differential diagnosis?"
            ],
            "triage": [
                "Is there any feature requiring urgent clinical attention?",
                "Indicate whether this image suggests normal or abnormal findings.",
                "From a clinical triage perspective, identify any immediate red flags or concerning features."
            ]
        }

    def sample_prompt(self, category: Optional[str] = None) -> str:
        """
        Randomly sample a prompt from specified category or all categories.
        """
        if category and category in self.prompts:
            return random.choice(self.prompts[category])
        
        all_prompts = sum(self.prompts.values(), [])
        return random.choice(all_prompts)

    def all_prompts(self) -> List[str]:
        """Return all prompt templates flattened."""
        return sum(self.prompts.values(), [])

