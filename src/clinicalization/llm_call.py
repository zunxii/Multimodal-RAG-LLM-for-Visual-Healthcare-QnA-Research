import os
import google.generativeai as genai
from dotenv import load_dotenv


class LLM_call():
    """llm basic all"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__("Gemini-2.5")
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def llm_call_fn(self, query: str,) -> str:
        """Generate caption using Gemini 2.5 Vision."""
    
        response = self.model.generate_content(
            [
                query,
            ]
        )
        return response.text.strip()
