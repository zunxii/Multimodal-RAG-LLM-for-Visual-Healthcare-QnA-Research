"""
Generate structured medical hypotheses instead of caption clouds.
Replaces DynamicCaptionCloud with diagnostic diversity.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import google.generativeai as genai
from dotenv import load_dotenv

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class MedicalHypothesis:
    """Structured medical hypothesis"""
    diagnosis: str
    mechanism: str
    visual_features: List[str]
    keywords: List[str]
    urgency: str  # low/moderate/high/emergency
    body_regions: List[str]
    clinical_context: str
    confidence: float = 0.0


class MedicalHypothesisGenerator:
    """
    Generate structured medical hypotheses for differential diagnosis.
    
    Replaces caption cloud with clinically distinct diagnostic hypotheses.
    Each hypothesis represents a different pathological mechanism.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        logger.info("Initialized MedicalHypothesisGenerator")
    
    def generate_hypotheses(
        self,
        image_path: str,
        clinical_query: str,
        n_hypotheses: int = 5
    ) -> List[MedicalHypothesis]:
        """
        Generate N distinct medical hypotheses from image + query.
        
        Args:
            image_path: Path to medical image
            clinical_query: User's clinical question
            n_hypotheses: Number of distinct hypotheses to generate
            
        Returns:
            List of structured medical hypotheses
        """
        logger.info(f"Generating {n_hypotheses} medical hypotheses...")
        
        # Read image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Build structured prompt
        prompt = self._build_hypothesis_prompt(clinical_query, n_hypotheses)
        
        # Generate with VLM
        response = self.model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt
            ],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        # Parse structured output
        hypotheses = self._parse_hypotheses(response.text)
        
        logger.info(f"Generated {len(hypotheses)} distinct hypotheses")
        return hypotheses
    
    def _build_hypothesis_prompt(
        self,
        clinical_query: str,
        n_hypotheses: int
    ) -> str:
        """Build structured prompt for hypothesis generation"""
        
        prompt = f"""Analyze this medical image and generate {n_hypotheses} CLINICALLY DISTINCT differential diagnoses.

User Query: {clinical_query}

For each hypothesis, provide:
1. Diagnosis name (specific condition)
2. Pathophysiological mechanism (how it causes the observed features)
3. Key visual features to look for
4. Clinical keywords for retrieval
5. Urgency level (low/moderate/high/emergency)
6. Body regions typically affected
7. Brief clinical context

CRITICAL: Make hypotheses CLINICALLY DISTINCT:
- Hypothesis 1: Most likely primary diagnosis
- Hypothesis 2: Alternative with different mechanism
- Hypothesis 3: Emergency condition to rule out
- Hypothesis 4: Chronic underlying condition
- Hypothesis 5: Benign variant (if applicable)

Output as JSON array:
[
  {{
    "diagnosis": "Exact condition name",
    "mechanism": "Pathophysiological mechanism",
    "visual_features": ["feature1", "feature2", ...],
    "keywords": ["keyword1", "keyword2", ...],
    "urgency": "low|moderate|high|emergency",
    "body_regions": ["region1", "region2", ...],
    "clinical_context": "Brief clinical description"
  }},
  ...
]

JSON array only, no other text:"""
        
        return prompt
    
    def _parse_hypotheses(self, response_text: str) -> List[MedicalHypothesis]:
        """Parse JSON response into MedicalHypothesis objects"""
        
        try:
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_text = response_text[json_start:json_end]
            hypotheses_data = json.loads(json_text)
            
            # Convert to MedicalHypothesis objects
            hypotheses = []
            for data in hypotheses_data:
                hypothesis = MedicalHypothesis(
                    diagnosis=data.get("diagnosis", "Unknown"),
                    mechanism=data.get("mechanism", ""),
                    visual_features=data.get("visual_features", []),
                    keywords=data.get("keywords", []),
                    urgency=data.get("urgency", "moderate"),
                    body_regions=data.get("body_regions", []),
                    clinical_context=data.get("clinical_context", "")
                )
                hypotheses.append(hypothesis)
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Failed to parse hypotheses: {e}")
            logger.debug(f"Response text: {response_text}")
            
            # Fallback: create single generic hypothesis
            return [MedicalHypothesis(
                diagnosis="Unspecified condition",
                mechanism="Unknown",
                visual_features=["Visible abnormality"],
                keywords=["medical", "condition"],
                urgency="moderate",
                body_regions=["Unknown"],
                clinical_context="Requires clinical evaluation"
            )]
    
    def save_hypotheses(
        self,
        hypotheses: List[MedicalHypothesis],
        output_path: Path
    ) -> None:
        """Save hypotheses to JSON file"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(h) for h in hypotheses],
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"Saved hypotheses to {output_path}")
    
    def load_hypotheses(self, file_path: Path) -> List[MedicalHypothesis]:
        """Load hypotheses from JSON file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [MedicalHypothesis(**h) for h in data]