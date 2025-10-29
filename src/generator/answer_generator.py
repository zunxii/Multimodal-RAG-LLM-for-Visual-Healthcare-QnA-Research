"""
Answer generation module using LLM with retrieved evidence.
Implements STEP 5 from the architecture.
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

from ..utils import get_logger

logger = get_logger(__name__)


class AnswerGenerator:
    """
    Generate answers grounded in retrieved evidence.
    Uses LLM to synthesize information from multiple sources.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize answer generator.
        
        Args:
            model_name: LLM model to use
        """
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        logger.info(f"Initialized AnswerGenerator with {model_name}")
    
    def generate_answer(
        self,
        query: str,
        retrieved_evidence: List[Dict[str, Any]],
        image_path: str = None,
        include_provenance: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer based on retrieved evidence.
        
        Args:
            query: User's clinical query
            retrieved_evidence: List of retrieved evidence (KB + dynamic)
            image_path: Optional image path for visual context
            include_provenance: Whether to include source citations
            
        Returns:
            Dictionary with answer, confidence, and provenance
        """
        logger.info("Generating answer from retrieved evidence...")
        
        # Build prompt with evidence
        prompt = self._build_prompt(
            query=query,
            evidence=retrieved_evidence,
            include_provenance=include_provenance
        )
        
        # Generate answer
        try:
            if image_path and os.path.exists(image_path):
                # Multimodal generation with image
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                
                response = self.model.generate_content(
                    [
                        {"mime_type": "image/jpeg", "data": image_bytes},
                        prompt
                    ],
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024
                    )
                )
            else:
                # Text-only generation
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024
                    )
                )
            
            answer_text = response.text.strip()
            
            # Parse answer and extract provenance
            parsed = self._parse_answer(answer_text, retrieved_evidence)
            
            logger.info("Answer generated successfully")
            
            return {
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "provenance": parsed["provenance"],
                "raw_answer": answer_text,
                "evidence_count": len(retrieved_evidence)
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": "Unable to generate answer due to an error.",
                "confidence": 0.0,
                "provenance": [],
                "error": str(e)
            }
    
    def _build_prompt(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        include_provenance: bool
    ) -> str:
        """Build prompt for LLM with evidence."""
        
        # System instruction
        system = (
            "You are a medical AI assistant. Generate a clinically accurate answer "
            "based ONLY on the provided evidence. Do not add information not present "
            "in the evidence. If the evidence is insufficient, state that clearly.\n\n"
        )
        
        if include_provenance:
            system += (
                "IMPORTANT: Cite sources using [Source N] notation when making claims. "
                "Each claim should reference the evidence it comes from.\n\n"
            )
        
        # Format evidence
        evidence_text = "Retrieved Evidence:\n\n"
        for i, item in enumerate(evidence, 1):
            source = item.get("source", "unknown")
            content = item.get("content", "")
            score = item.get("score", 0.0)
            
            evidence_text += f"[Source {i}] ({source}, score: {score:.3f})\n"
            evidence_text += f"{content}\n\n"
        
        # User query
        query_text = f"Clinical Query: {query}\n\n"
        
        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Synthesize information from the evidence above\n"
            "2. Provide a concise, medically accurate answer\n"
            "3. Cite sources for each claim using [Source N]\n"
            "4. If evidence is conflicting, mention all perspectives\n"
            "5. If evidence is insufficient, state what is missing\n"
            "6. End with a confidence score (0.0-1.0) and reasoning\n\n"
            "Answer:\n"
        )
        
        prompt = system + evidence_text + query_text + instructions
        return prompt
    
    def _parse_answer(
        self,
        answer_text: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse answer to extract confidence and provenance.
        
        Returns:
            Dictionary with parsed answer, confidence, and provenance
        """
        # Simple parsing - extract confidence if mentioned
        import re
        
        confidence = 0.7  # Default
        confidence_match = re.search(
            r'confidence[:\s]+([0-9.]+)',
            answer_text.lower()
        )
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except:
                pass
        
        # Extract source citations
        provenance = []
        citations = re.findall(r'\[Source (\d+)\]', answer_text)
        for cite_num in set(citations):
            idx = int(cite_num) - 1
            if 0 <= idx < len(evidence):
                provenance.append({
                    "source_num": int(cite_num),
                    "source_type": evidence[idx].get("source", "unknown"),
                    "content": evidence[idx].get("content", "")[:100] + "..."
                })
        
        return {
            "answer": answer_text,
            "confidence": confidence,
            "provenance": provenance
        }
    
    def generate_batch_answers(
        self,
        query: str,
        evidence_per_neighbor: List[Dict[str, Any]],
        image_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple candidate answers from different evidence subsets.
        Implements per-neighbor answer generation from the paper.
        
        Args:
            query: Clinical query
            evidence_per_neighbor: List of evidence bundles (one per neighbor)
            image_path: Optional image path
            
        Returns:
            List of candidate answers with confidence scores
        """
        candidates = []
        
        for i, evidence_bundle in enumerate(evidence_per_neighbor):
            logger.info(f"Generating candidate answer {i+1}/{len(evidence_per_neighbor)}")
            
            # Generate answer for this evidence
            result = self.generate_answer(
                query=query,
                retrieved_evidence=[evidence_bundle],
                image_path=image_path,
                include_provenance=True
            )
            
            candidates.append({
                "candidate_id": i,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "evidence": evidence_bundle
            })
        
        return candidates
    
    def fuse_answers(
        self,
        candidates: List[Dict[str, Any]],
        strategy: str = "weighted_vote"
    ) -> Dict[str, Any]:
        """
        Fuse multiple candidate answers into final answer.
        
        Args:
            candidates: List of candidate answers
            strategy: Fusion strategy ('weighted_vote', 'highest_confidence', 'llm_synthesis')
            
        Returns:
            Final fused answer
        """
        if not candidates:
            return {
                "answer": "No candidates to fuse.",
                "confidence": 0.0,
                "strategy": strategy
            }
        
        if strategy == "highest_confidence":
            # Select answer with highest confidence
            best = max(candidates, key=lambda x: x["confidence"])
            return {
                "answer": best["answer"],
                "confidence": best["confidence"],
                "strategy": strategy,
                "selected_candidate": best["candidate_id"]
            }
        
        elif strategy == "weighted_vote":
            # Weight answers by confidence and vote
            # For simplicity, return highest confidence
            # In production, implement proper voting
            best = max(candidates, key=lambda x: x["confidence"])
            return {
                "answer": best["answer"],
                "confidence": best["confidence"],
                "strategy": strategy,
                "num_candidates": len(candidates)
            }
        
        elif strategy == "llm_synthesis":
            # Use LLM to synthesize all candidates
            synthesis_prompt = self._build_synthesis_prompt(candidates)
            
            try:
                response = self.model.generate_content(
                    synthesis_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1024
                    )
                )
                
                return {
                    "answer": response.text.strip(),
                    "confidence": sum(c["confidence"] for c in candidates) / len(candidates),
                    "strategy": strategy,
                    "num_candidates": len(candidates)
                }
            except Exception as e:
                logger.error(f"LLM synthesis failed: {e}")
                # Fallback to highest confidence
                return self.fuse_answers(candidates, strategy="highest_confidence")
        
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
    
    def _build_synthesis_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        """Build prompt for synthesizing multiple candidate answers."""
        
        prompt = (
            "You are a medical AI assistant. Multiple candidate answers have been "
            "generated for a clinical query. Synthesize them into a single, "
            "coherent, and accurate final answer.\n\n"
            "Candidate Answers:\n\n"
        )
        
        for i, cand in enumerate(candidates, 1):
            prompt += f"Candidate {i} (confidence: {cand['confidence']:.2f}):\n"
            prompt += f"{cand['answer']}\n\n"
        
        prompt += (
            "Instructions:\n"
            "1. Identify common themes across candidates\n"
            "2. Resolve contradictions by favoring higher-confidence answers\n"
            "3. Create a comprehensive final answer\n"
            "4. Maintain clinical accuracy\n\n"
            "Final Synthesized Answer:\n"
        )
        
        return prompt