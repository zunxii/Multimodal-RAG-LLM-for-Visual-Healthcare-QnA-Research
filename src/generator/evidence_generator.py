"""
src/generation/evidence_generator.py

STEP 5: Generate Answer Grounded in Retrieved Medical Evidence
"""

import os
import google.generativeai as genai
from typing import Dict, Any, List
from dotenv import load_dotenv

from ..utils import get_logger

logger = get_logger(__name__)


class EvidenceGroundedGenerator:
    """
    Generate answers grounded in retrieved medical evidence.
    
    Uses retrieved similar cases from medical KB to produce
    clinically-informed answers with explicit citations.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        
        self.system_prompt = """You are a medical AI assistant providing evidence-based answers.

Your task:
1. Answer the user's medical question about their image
2. Ground your answer in the retrieved similar medical cases provided
3. Cite specific cases using [Case ID] when making claims
4. Be clear about diagnostic uncertainty
5. Always recommend consulting healthcare professionals

Format your response as:
**Answer**: [Your answer grounded in evidence]

**Evidence Support**:
- [Claim 1] → Supported by [Case ID]
- [Claim 2] → Supported by [Case ID]

**Confidence**: [Low/Medium/High] based on evidence quality

**Recommendation**: [Clinical recommendation]
"""
    
    def generate_answer(
        self,
        query_text: str,
        retrieved_evidence: Dict[str, Any],
        query_image_captions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer grounded in retrieved evidence.
        
        Args:
            query_text: User's clinical query
            retrieved_evidence: Evidence from medical KB retrieval
            query_image_captions: Captions describing query image
            
        Returns:
            Dictionary with generated answer and grounding information
        """
        logger.info("Generating evidence-grounded answer...")
        
        # Build prompt with evidence
        prompt = self._build_prompt(query_text, retrieved_evidence, query_image_captions)
        
        # Generate answer
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024
            )
        )
        
        answer_text = response.text.strip()
        
        # Parse structured output
        parsed = self._parse_answer(answer_text)
        
        # Compute grounding score
        grounding_score = self._compute_grounding_score(
            parsed,
            retrieved_evidence
        )
        
        result = {
            'answer': parsed.get('answer', answer_text),
            'evidence_support': parsed.get('evidence_support', []),
            'confidence': parsed.get('confidence', 'Unknown'),
            'recommendation': parsed.get('recommendation', ''),
            'grounding_score': grounding_score,
            'retrieved_cases_used': [
                c['case_id'] for c in retrieved_evidence['retrieved_cases']
            ],
            'full_response': answer_text
        }
        
        logger.info(f"✅ Generated answer (confidence: {result['confidence']})")
        return result
    
    def _build_prompt(
        self,
        query_text: str,
        retrieved_evidence: Dict[str, Any],
        query_image_captions: List[str] = None
    ) -> str:
        """Build generation prompt with evidence"""
        prompt_parts = [
            self.system_prompt,
            "\n## USER QUERY:",
            f'"{query_text}"',
            "\n## QUERY IMAGE DESCRIPTIONS:",
        ]
        
        if query_image_captions:
            for i, caption in enumerate(query_image_captions[:3], 1):
                prompt_parts.append(f"{i}. {caption}")
        else:
            prompt_parts.append("(No captions available)")
        
        prompt_parts.extend([
            "\n## RETRIEVED SIMILAR MEDICAL CASES:",
            retrieved_evidence['evidence_text'],
            "\n## YOUR TASK:",
            "Based on the query image descriptions and retrieved similar cases, "
            "provide an evidence-grounded answer. Cite case IDs explicitly."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_answer(self, answer_text: str) -> Dict[str, Any]:
        """Parse structured answer from LLM response"""
        parsed = {
            'answer': '',
            'evidence_support': [],
            'confidence': 'Unknown',
            'recommendation': ''
        }
        
        # Simple parsing (can be improved with regex)
        sections = answer_text.split('**')
        
        for i, section in enumerate(sections):
            section_lower = section.lower()
            
            if 'answer' in section_lower and i + 1 < len(sections):
                parsed['answer'] = sections[i + 1].strip()
            elif 'confidence' in section_lower and i + 1 < len(sections):
                conf_text = sections[i + 1].strip()
                if 'high' in conf_text.lower():
                    parsed['confidence'] = 'High'
                elif 'medium' in conf_text.lower():
                    parsed['confidence'] = 'Medium'
                elif 'low' in conf_text.lower():
                    parsed['confidence'] = 'Low'
            elif 'recommendation' in section_lower and i + 1 < len(sections):
                parsed['recommendation'] = sections[i + 1].strip()
        
        # If parsing fails, use full text as answer
        if not parsed['answer']:
            parsed['answer'] = answer_text
        
        return parsed
    
    def _compute_grounding_score(
        self,
        parsed_answer: Dict[str, Any],
        retrieved_evidence: Dict[str, Any]
    ) -> float:
        """
        Compute grounding score based on:
        - Mean retrieval score
        - Confidence level
        - Number of cases cited
        """
        # Base score from retrieval
        retrieval_score = retrieved_evidence.get('mean_score', 0.0)
        
        # Confidence adjustment
        confidence_map = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4, 'Unknown': 0.5}
        confidence_factor = confidence_map.get(parsed_answer['confidence'], 0.5)
        
        # Combine (weighted average)
        grounding_score = 0.6 * retrieval_score + 0.4 * confidence_factor
        
        return float(grounding_score)


class MultiCandidateGenerator:
    """
    Generate multiple candidate answers from Top-K retrieved cases,
    then fuse using consensus strategy.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.generator = EvidenceGroundedGenerator(model_name)
    
    def generate_with_fusion(
        self,
        query_text: str,
        retrieved_cases: List[Dict[str, Any]],
        query_image_captions: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate multiple candidate answers (one per top case),
        then fuse using majority voting or confidence weighting.
        """
        logger.info(f"Generating {top_k} candidate answers...")
        
        candidates = []
        for case in retrieved_cases[:top_k]:
            # Create single-case evidence
            single_evidence = {
                'retrieved_cases': [case],
                'evidence_text': f"""
[{case['case_id']}] (Score: {case['score']:.3f})
Diagnosis: {case['diagnosis']}
Findings: {case['findings']}
                """.strip(),
                'mean_score': case['score']
            }
            
            # Generate answer for this case
            result = self.generator.generate_answer(
                query_text,
                single_evidence,
                query_image_captions
            )
            
            candidates.append({
                'answer': result['answer'],
                'case_id': case['case_id'],
                'diagnosis': case['diagnosis'],
                'confidence': result['confidence'],
                'grounding_score': result['grounding_score']
            })
        
        # Fuse candidates (simple majority voting on diagnosis)
        from collections import Counter
        diagnosis_votes = Counter([c['diagnosis'] for c in candidates])
        consensus_diagnosis = diagnosis_votes.most_common(1)[0][0]
        
        # Select best answer (highest grounding score)
        best_candidate = max(candidates, key=lambda x: x['grounding_score'])
        
        return {
            'final_answer': best_candidate['answer'],
            'consensus_diagnosis': consensus_diagnosis,
            'candidates': candidates,
            'agreement_rate': diagnosis_votes[consensus_diagnosis] / len(candidates),
            'mean_grounding_score': sum(c['grounding_score'] for c in candidates) / len(candidates)
        }