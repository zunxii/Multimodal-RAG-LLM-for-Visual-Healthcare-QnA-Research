import google.generativeai as genai
from app.config import settings

class LLMService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        print("LLM Service (Gemini) initialized.")

    def generate_text(self, prompt: str) -> str:
        """Generates text using the configured Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error during LLM text generation: {e}")
            return f"An error occurred while generating the text: {str(e)}"

# Singleton instance
llm_service = LLMService()