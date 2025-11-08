"""

Compute clinical relevance between query and retrieved cases.
Goes beyond embedding similarity to structured medical matching.
"""

from typing import Dict, List, Set, Any
from ..utils import get_logger

logger = get_logger(__name__)


class ClinicalRelevanceScorer:
    """
    Score clinical relevance between query features and case metadata.
    
    Combines multiple medical dimensions:
    - Body region matching
    - Symptom overlap
    - Visual pattern matching
    - Urgency alignment
    - Diagnostic category relevance
    """
    
    # Anatomical region relationships
    RELATED_REGIONS = {
        "fingers": ["hand", "upper_extremity", "digits"],
        "hand": ["fingers", "wrist", "upper_extremity"],
        "foot": ["toes", "ankle", "lower_extremity"],
        "leg": ["knee", "foot", "lower_extremity"],
        "knee": ["leg", "lower_extremity"],
        "ankle": ["foot", "lower_extremity"],
    }
    
    def __init__(self):
        logger.info("Initialized ClinicalRelevanceScorer")
    
    def compute_relevance(
        self,
        query_features: Dict[str, Any],
        case_metadata: Dict[str, Any]
    ) -> float:
        """
        Compute clinical relevance score [0, 1].
        
        Args:
            query_features: Features from medical hypothesis
            case_metadata: Metadata from KB case
            
        Returns:
            Relevance score combining multiple factors
        """
        score = 0.0
        
        # 1. Body region match (30%)
        region_score = self._body_region_score(
            query_features.get("body_regions", []),
            case_metadata.get("category", "")
        )
        score += 0.30 * region_score
        
        # 2. Symptom/keyword overlap (30%)
        symptom_score = self._symptom_overlap(
            query_features.get("keywords", []),
            case_metadata.get("description", ""),
            case_metadata.get("context", "")
        )
        score += 0.30 * symptom_score
        
        # 3. Visual pattern match (20%)
        visual_score = self._visual_pattern_match(
            query_features.get("visual_features", []),
            case_metadata.get("description", "")
        )
        score += 0.20 * visual_score
        
        # 4. Urgency alignment (10%)
        urgency_score = self._urgency_alignment(
            query_features.get("urgency", "moderate"),
            case_metadata.get("urgency", "moderate")
        )
        score += 0.10 * urgency_score
        
        # 5. Diagnostic category (10%)
        category_score = self._category_relevance(
            query_features.get("diagnosis", ""),
            case_metadata.get("category", "")
        )
        score += 0.10 * category_score
        
        return float(score)
    
    def _body_region_score(
        self,
        query_regions: List[str],
        case_category: str
    ) -> float:
        """Score body region matching"""
        
        if not query_regions or not case_category:
            return 0.0
        
        case_category_lower = case_category.lower()
        
        # Exact match
        for region in query_regions:
            region_lower = region.lower()
            if region_lower in case_category_lower:
                return 1.0
        
        # Related region match
        for region in query_regions:
            region_lower = region.lower()
            related = self.RELATED_REGIONS.get(region_lower, [])
            for rel in related:
                if rel in case_category_lower:
                    return 0.6
        
        return 0.0
    
    def _symptom_overlap(
        self,
        query_keywords: List[str],
        case_description: str,
        case_context: str
    ) -> float:
        """Compute symptom/keyword overlap using Jaccard similarity"""
        
        if not query_keywords:
            return 0.0
        
        # Combine case text
        case_text = f"{case_description} {case_context}".lower()
        
        # Count matches
        matches = sum(
            1 for keyword in query_keywords
            if keyword.lower() in case_text
        )
        
        # Jaccard-like score
        return matches / len(query_keywords) if query_keywords else 0.0
    
    def _visual_pattern_match(
        self,
        query_features: List[str],
        case_description: str
    ) -> float:
        """Match visual features (color, pattern, distribution)"""
        
        if not query_features or not case_description:
            return 0.0
        
        case_desc_lower = case_description.lower()
        
        matches = sum(
            1 for feature in query_features
            if feature.lower() in case_desc_lower
        )
        
        return matches / len(query_features) if query_features else 0.0
    
    def _urgency_alignment(
        self,
        query_urgency: str,
        case_urgency: str
    ) -> float:
        """Score urgency level alignment"""
        
        urgency_map = {
            "low": 1,
            "moderate": 2,
            "high": 3,
            "emergency": 4
        }
        
        q_level = urgency_map.get(query_urgency.lower(), 2)
        c_level = urgency_map.get(case_urgency.lower(), 2)
        
        # Perfect match
        if q_level == c_level:
            return 1.0
        
        # Adjacent levels
        if abs(q_level - c_level) == 1:
            return 0.5
        
        return 0.0
    
    def _category_relevance(
        self,
        query_diagnosis: str,
        case_category: str
    ) -> float:
        """Check if diagnosis matches case category"""
        
        if not query_diagnosis or not case_category:
            return 0.0
        
        # Simple keyword matching
        q_words = set(query_diagnosis.lower().split())
        c_words = set(case_category.lower().split())
        
        intersection = q_words & c_words
        union = q_words | c_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def filter_by_relevance(
        self,
        candidates: List[Dict[str, Any]],
        query_features: Dict[str, Any],
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by clinical relevance threshold.
        
        Args:
            candidates: Retrieved cases from KB
            query_features: Query hypothesis features
            threshold: Minimum relevance score
            
        Returns:
            Filtered and scored candidates
        """
        scored_candidates = []
        
        for candidate in candidates:
            relevance = self.compute_relevance(
                query_features,
                candidate
            )
            
            if relevance >= threshold:
                candidate_copy = candidate.copy()
                candidate_copy["clinical_relevance"] = relevance
                candidate_copy["combined_score"] = (
                    0.6 * candidate.get("retrieval_score", 0.0) +
                    0.4 * relevance
                )
                scored_candidates.append(candidate_copy)
        
        # Sort by combined score
        scored_candidates.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        logger.info(
            f"Filtered {len(scored_candidates)}/{len(candidates)} "
            f"candidates (threshold={threshold})"
        )
        
        return scored_candidates