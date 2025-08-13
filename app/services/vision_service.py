from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from app.config import settings
from app.services.llm_service import llm_service

class VisionService:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(settings.VISION_CAPTION_MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(settings.VISION_CAPTION_MODEL_NAME)
        print(f"Vision Service initialized with model {settings.VISION_CAPTION_MODEL_NAME}.")

    def _get_initial_caption(self, image: Image.Image) -> str:
        """Generates a single, basic caption for the image."""
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def generate_doctor_descriptions(self, image: Image.Image) -> list[str]:
        """
        Generates 100 doctor-style descriptions for an image.
        """
        initial_caption = self._get_initial_caption(image)
        print(f"Initial image caption: '{initial_caption}'")

        prompt = f"""
        An initial observation of a medical symptom from an image is: "{initial_caption}".

        Your task is to expand this single observation into 100 diverse, detailed, clinical descriptions.
        Each description should be a plausible medical interpretation of what could be in the image.
        Phrase them as if written by different medical professionals (e.g., a dermatologist, a general practitioner, an allergist, a pediatrician).
        Use precise medical terminology where appropriate. Focus on visual characteristics like morphology, color, distribution, and texture.

        Examples of phrasing:
        - "Lesion appears as an erythematous maculopapular rash with well-defined borders."
        - "Multiple vesicular eruptions noted on the distal forearm, consistent with contact dermatitis."
        - "The patient presents with a solitary, annular plaque with central clearing and a raised, scaly edge."
        - "Evidence of urticarial wheals with surrounding erythema, suggestive of an acute allergic reaction."

        Generate exactly 100 unique descriptions, each on a new line, based on the initial observation: "{initial_caption}".
        """
        
        print("Generating 100 doctor-style descriptions with LLM...")
        generated_text = llm_service.generate_text(prompt)
        
        # Split the response by newlines and filter out any empty lines
        descriptions = [desc.strip() for desc in generated_text.split('\n') if desc.strip()]
        
        print(f"Generated {len(descriptions)} descriptions.")
        return descriptions[:100] # Ensure we don't exceed 100

# Singleton instance
vision_service = VisionService()