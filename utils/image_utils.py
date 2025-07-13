import numpy as np
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY) # type: ignore
vision_model = genai.GenerativeModel('gemini-1.5-pro-latest') # type: ignore

def get_image_embedding(image_file):
    image_bytes = Image.open(BytesIO(image_file.read()))
    response = vision_model.generate_content(["Give me a vector representation of this image for search indexing.", image_bytes])
    # Mock embedding for demo (replace this with Gemini vision embedding extraction)
    return np.random.rand(512)
