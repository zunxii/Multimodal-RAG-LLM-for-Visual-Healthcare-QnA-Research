import random

class PromptBank:
    """
    Stores and manages prompt templates across categories:
    clinical, attribute, question-based, differential, triage-oriented, etc.
    """

    def __init__(self):
        self.prompts = {
            "clinical": [
                "Describe any visible abnormalities in this medical image.",
                "Identify potential pathologies or irregularities.",
                "Summarize diagnostic findings relevant to this scan."
            ],
            "attribute": [
                "List key visual attributes like texture, opacity, and shape.",
                "Describe color, size, and structural anomalies in the image.",
            ],
            "question": [
                "What might explain the visible patterns in this scan?",
                "Are there signs of infection, lesion, or fracture?"
            ],
            "triage": [
                "Is there any feature requiring urgent clinical attention?",
                "Indicate whether this image suggests normal or abnormal findings."
            ]
        }

    def sample_prompt(self, category=None) -> str:
        """
        Randomly sample a prompt. Optionally specify a category.
        """
        if category and category in self.prompts:
            return random.choice(self.prompts[category])
        all_prompts = sum(self.prompts.values(), [])
        return random.choice(all_prompts)

    def all_prompts(self):
        """Return all prompt templates flattened."""
        return sum(self.prompts.values(), [])
