from pydantic import BaseModel, Field

class SymptomAnalysisRequest(BaseModel):
    text_description: str = Field(
        ...,
        description="User's freeform text description of their symptom.",
        example="I have a red, itchy rash on my arm that has been there for two days. It has small bumps."
    )

class SymptomAnalysisResponse(BaseModel):
    summarized_user_text: str
    retrieved_doctor_interpretation: str
    composed_llm_question: str
    final_answer: str
    disclaimer: str = "DISCLAIMER: This is an AI-generated analysis and not a substitute for professional medical advice. Please consult a qualified healthcare provider for any health concerns."