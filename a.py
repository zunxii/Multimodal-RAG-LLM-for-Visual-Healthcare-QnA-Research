import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a Gemini model instance
m = genai.GenerativeModel("gemini-pro")

# Test message
response = m.generate_content("Say hello")

print("Gemini ✅", response.text)
