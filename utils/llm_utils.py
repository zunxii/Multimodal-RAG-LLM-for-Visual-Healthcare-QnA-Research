import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
llm = genai.GenerativeModel('gemini-2.5')  # type: ignore

def generate_final_answer(query, context):
    prompt = f"""
    Using this context below, answer the medical question carefully.
    
    Context:
    {context}

    Question:
    {query}

    Answer ONLY based on this context, do not hallucinate.
    """
    response = llm.generate_content(prompt)
    return response.text.strip()
