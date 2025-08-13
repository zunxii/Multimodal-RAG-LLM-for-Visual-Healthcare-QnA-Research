# Medical Symptom Analyzer API 🏥

A sophisticated AI-powered API that analyzes medical symptoms using image and text input through an intelligent RAG pipeline.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/zunxii/Multimodal-RAG-LLM-for-Visual-Healthcare-QnA-Research
cd Multimodal-RAG-LLM-for-Visual-Healthcare-QnA-Research

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo 'GOOGLE_API_KEY="your_api_key_here"' > .env

# Run the server
uvicorn app.main:app --reload
```

## 🔬 How It Works

The system processes user input through a unique 6-step pipeline:

1. **User Input**: Image + text description
2. **Image Analysis**: Generate 100 doctor-style interpretations
3. **Text Analysis**: Summarize user description clinically
4. **Smart Matching**: Find best image interpretation using vector similarity
5. **Prompt Composition**: Combine matched interpretation with user summary
6. **Final Analysis**: Generate comprehensive medical analysis with disclaimer

## 📡 API Usage

### Interactive Documentation
Visit `http://127.0.0.1:8000/docs` for the Swagger UI interface.

### cURL Example
```bash
curl -X POST 'http://127.0.0.1:8000/analyze' \
  -H 'Content-Type: multipart/form-data' \
  -F 'text_description=Red itchy rash on arm, appeared after gardening' \
  -F 'image=@sample_image.jpg'
```

## 📁 Project Structure

```
medical-symptom-analyzer/
|── data/  #have some test data  
├── app/
│   ├── main.py              # FastAPI routes
│   ├── config.py            # Configuration
│   ├── models.py            # Data models
│   └── services/            # Core services
│       ├── pipeline.py      # Main processing flow
│       ├── llm_service.py   # Gemini integration
│       ├── vision_service.py
│       ├── embedding_service.py
│       └── vector_db_service.py
├── requirements.txt
├── .env                     # API keys (create this)
└── README.md
```

## 🔧 Prerequisites

- Python 3.9+
- Google AI Studio API key

## ⚠️ Disclaimer

This is an AI-generated analysis for educational purposes only. Not a substitute for professional medical advice. Always consult qualified healthcare providers for health concerns.