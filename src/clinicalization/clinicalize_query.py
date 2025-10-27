import os
import google.generativeai as genai
from dotenv import load_dotenv
from ..utils import get_logger

logger = get_logger(__name__)


class QueryClinicalizer:
    """
    Converts free-text user queries into clinicalized medical search strings.
    Implements the g(·) function: q_user → q_clin
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        
        self.system_prompt = (
            "You are a medical query normalizer. Rewrite the user's free-text query "
            "into a concise, clinical search string suitable for medical retrieval. "
            "Use standard medical terminology, keep it short (one sentence), "
            "and focus on the key clinical concepts. "
            "Do not add explanations, just output the clinicalized query."
        )

    def clinicalize(self, q_user: str) -> str:
        """
        Clinicalize user query.
        
        Args:
            q_user: Free-text user query
            
        Returns:
            q_clin: Clinicalized medical query string
        """
        prompt = f"{self.system_prompt}\n\nUser query: {q_user}\n\nClinicalized query:"
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=100
                )
            )
            q_clin = response.text.strip()
            logger.info(f"Clinicalized: '{q_user}' → '{q_clin}'")
            return q_clin
            
        except Exception as e:
            logger.error(f"Clinicalization failed: {e}. Using original query.")
            return q_user


# Convenience function
_clinicalizer = None

def clinicalize_query(q_user: str) -> str:
    """Convenience function for query clinicalization"""
    global _clinicalizer
    if _clinicalizer is None:
        _clinicalizer = QueryClinicalizer()
    return _clinicalizer.clinicalize(q_user)

