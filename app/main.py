from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import io

from app.models import SymptomAnalysisResponse
from app.services.pipeline import run_symptom_analysis_pipeline

app = FastAPI(
    title="Medical Symptom Analyzer API",
    description="Analyzes user-provided symptom images and text using a multi-step AI pipeline.",
    version="1.0.0"
)

@app.post("/analyze", response_model=SymptomAnalysisResponse)
async def analyze_symptoms(
    text_description: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Accepts an image and text description of a medical symptom,
    and returns a detailed AI-powered analysis.
    """
    # Validate image file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # The main call to your pipeline
        result = run_symptom_analysis_pipeline(
            image_file=image,
            user_text=text_description
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Return a more generic server error for unexpected issues
        raise HTTPException(status_code=500, detail="An internal server error occurred during analysis.")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Medical Symptom Analyzer API is running. See /docs for usage."}