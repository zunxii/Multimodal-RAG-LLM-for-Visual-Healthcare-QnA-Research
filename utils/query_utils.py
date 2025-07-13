import numpy as np
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY) # type: ignore
text_model = genai.GenerativeModel('gemini-1.5-pro-latest') # type: ignore

def get_query_embedding(query):
    # Mock embedding for demo (replace this with Gemini embedding API call)
    return np.random.rand(512)

def generate_summary(chunks):
    context = "\n".join(chunks)
    prompt = f"Summarize the following medical records into one concise paragraph:\n{context}"
    response = text_model.generate_content(prompt)
    return response.text.strip()
