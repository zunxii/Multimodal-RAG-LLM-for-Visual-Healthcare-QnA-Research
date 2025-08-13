import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # Model configuration
    # Using a general-purpose sentence-transformer is a good starting point.
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Using BLIP for base image captioning.
    VISION_CAPTION_MODEL_NAME: str = "Salesforce/blip-image-captioning-large"
    
    # Using Gemini Pro for all text generation tasks.
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"

# Instantiate settings
settings = Settings()

# Basic validation
if not settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file.")