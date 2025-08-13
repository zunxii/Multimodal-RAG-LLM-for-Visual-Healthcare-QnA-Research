from PIL import Image
import numpy as np

from app.services.vision_service import vision_service
from app.services.embedding_service import embedding_service
from app.services.vector_db_service import VectorDBService
from app.services.llm_service import llm_service
from app.models import SymptomAnalysisResponse

def run_symptom_analysis_pipeline(image_file, user_text: str) -> SymptomAnalysisResponse:
    """
    Executes the full 6-step analysis pipeline.
    """
    # ----------------------------------------------------------------------
    # 1. User Input (Handled by FastAPI, passed as arguments)
    # ----------------------------------------------------------------------
    print("Pipeline Step 1: Processing User Input.")
    image = Image.open(image_file.file).convert("RGB")

    # ----------------------------------------------------------------------
    # 2. Doctor-style Text Expansion for the Image & DB Storage
    # ----------------------------------------------------------------------
    print("\nPipeline Step 2: Generating Doctor-style Image Descriptions.")
    doctor_descriptions = vision_service.generate_doctor_descriptions(image)
    
    if not doctor_descriptions:
        raise ValueError("Failed to generate doctor descriptions from the image.")

    print("Embedding generated descriptions...")
    # Here, we only embed the text descriptions for simplicity and speed.
    # The prompt already fused the image context into the text.
    description_embeddings = embedding_service.embed_texts(doctor_descriptions)

    print("Storing embeddings in Vector DB...")
    vector_db = VectorDBService(dimension=embedding_service.embedding_dim)
    vector_db.build_index(description_embeddings, doctor_descriptions)

    # ----------------------------------------------------------------------
    # 3. Doctor-style Summary of User Text
    # ----------------------------------------------------------------------
    print("\nPipeline Step 3: Summarizing User Text.")
    summary_prompt = f"""
    Summarize the following user's description of their medical symptom into a concise, structured, clinical summary.
    Focus on key details like onset, duration, location, symptoms (itch, pain), and appearance.

    User Description: "{user_text}"

    Clinical Summary:
    """
    summarized_text = llm_service.generate_text(summary_prompt).strip()
    print(f"Summarized Text: '{summarized_text}'")

    print("Embedding summarized text...")
    summary_embedding = embedding_service.embed_texts([summarized_text])

    # ----------------------------------------------------------------------
    # 4. Retrieval
    # ----------------------------------------------------------------------
    print("\nPipeline Step 4: Retrieving Best Image Interpretation.")
    _, retrieved_interpretation = vector_db.search(summary_embedding)
    print(f"Retrieved Interpretation: '{retrieved_interpretation}'")

    # ----------------------------------------------------------------------
    # 5. Composition for LLM
    # ----------------------------------------------------------------------
    print("\nPipeline Step 5: Composing Final Question for LLM.")
    final_question_prompt = f"""
    You are a helpful medical information AI. Based on the combined information below, provide a detailed analysis.

    CONTEXT:
    1.  **Clinical Observation from Image:** "{retrieved_interpretation}"
    2.  **Patient's Own Description (Summarized):** "{summarized_text}"

    TASK:
    Based *only* on the provided context, generate a helpful response that includes:
    1.  A list of potential conditions that might align with these observations.
    2.  A brief, easy-to-understand explanation for each potential condition.
    3.  A set of relevant questions a doctor might ask the patient to get more clarity.
    4.  General advice on next steps (e.g., "It may be helpful to monitor...", "Consider seeing a healthcare professional...").

    Start your response with a clear synthesis of the information. Do not add any information not derivable from the context.
    Do NOT provide a diagnosis.
    """
    print(f"Composed Final Prompt:\n---\n{final_question_prompt}\n---")

    # ----------------------------------------------------------------------
    # 6. Final QnA
    # ----------------------------------------------------------------------
    print("\nPipeline Step 6: Generating Final Answer.")
    final_answer = llm_service.generate_text(final_question_prompt)
    
    # Construct the final response object
    response = SymptomAnalysisResponse(
        summarized_user_text=summarized_text,
        retrieved_doctor_interpretation=retrieved_interpretation,
        composed_llm_question=final_question_prompt,
        final_answer=final_answer
    )

    return response