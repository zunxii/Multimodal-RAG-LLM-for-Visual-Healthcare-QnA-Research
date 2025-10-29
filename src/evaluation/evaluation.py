"""
Evaluation module for RAG system.
Implements BLEU, ROUGE, and BERTScore metrics.
"""

import numpy as np
from typing import List, Dict, Any, Union
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

from ..utils import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Evaluation suite for RAG system.
    Computes BLEU, ROUGE, and BERTScore metrics.
    """
    
    def __init__(self):
        """Initialize evaluator with available metrics."""
        self.available_metrics = []
        
        if NLTK_AVAILABLE:
            self.available_metrics.append("BLEU")
            self.smoothing = SmoothingFunction().method1
        
        if ROUGE_AVAILABLE:
            self.available_metrics.append("ROUGE")
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        
        if BERTSCORE_AVAILABLE:
            self.available_metrics.append("BERTScore")
        
        logger.info(f"Initialized RAGEvaluator with metrics: {self.available_metrics}")
        
        if not self.available_metrics:
            logger.warning(
                "No evaluation metrics available! Install: "
                "pip install nltk rouge-score bert-score"
            )
    
    def evaluate(
        self,
        prediction: str,
        reference: str,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate prediction against reference using specified metrics.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            metrics: List of metrics to compute (default: all available)
            
        Returns:
            Dictionary with metric scores
        """
        if metrics is None:
            metrics = self.available_metrics
        
        results = {}
        
        # BLEU Score
        if "BLEU" in metrics and NLTK_AVAILABLE:
            try:
                bleu_scores = self.compute_bleu(prediction, reference)
                results["BLEU"] = bleu_scores
            except Exception as e:
                logger.error(f"BLEU computation failed: {e}")
                results["BLEU"] = {"error": str(e)}
        
        # ROUGE Scores
        if "ROUGE" in metrics and ROUGE_AVAILABLE:
            try:
                rouge_scores = self.compute_rouge(prediction, reference)
                results["ROUGE"] = rouge_scores
            except Exception as e:
                logger.error(f"ROUGE computation failed: {e}")
                results["ROUGE"] = {"error": str(e)}
        
        # BERTScore
        if "BERTScore" in metrics and BERTSCORE_AVAILABLE:
            try:
                bert_scores = self.compute_bertscore(prediction, reference)
                results["BERTScore"] = bert_scores
            except Exception as e:
                logger.error(f"BERTScore computation failed: {e}")
                results["BERTScore"] = {"error": str(e)}
        
        return results
    
    def compute_bleu(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4).
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            Dictionary with BLEU scores
        """
        if not NLTK_AVAILABLE:
            return {"error": "NLTK not available"}
        
        # Tokenize
        pred_tokens = word_tokenize(prediction.lower())
        ref_tokens = word_tokenize(reference.lower())
        
        # Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4
        bleu_scores = {}
        
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=weights,
                smoothing_function=self.smoothing
            )
            bleu_scores[f"BLEU-{n}"] = float(score)
        
        # Average BLEU
        bleu_scores["BLEU-avg"] = float(np.mean([
            bleu_scores[f"BLEU-{n}"] for n in range(1, 5)
        ]))
        
        return bleu_scores
    
    def compute_rouge(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            Dictionary with ROUGE F1 scores
        """
        if not ROUGE_AVAILABLE:
            return {"error": "rouge-score not available"}
        
        scores = self.rouge_scorer.score(reference, prediction)
        
        return {
            "ROUGE-1": float(scores['rouge1'].fmeasure),
            "ROUGE-2": float(scores['rouge2'].fmeasure),
            "ROUGE-L": float(scores['rougeL'].fmeasure),
        }
    
    def compute_bertscore(
        self,
        prediction: Union[str, List[str]],
        reference: Union[str, List[str]],
        model_type: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, float]:
        """
        Compute BERTScore (Precision, Recall, F1).
        
        Args:
            prediction: Generated text or list of texts
            reference: Reference text or list of texts
            model_type: BERT model to use
            
        Returns:
            Dictionary with BERTScore metrics
        """
        if not BERTSCORE_AVAILABLE:
            return {"error": "bert-score not available"}
        
        # Convert to lists
        if isinstance(prediction, str):
            prediction = [prediction]
        if isinstance(reference, str):
            reference = [reference]
        
        # Compute BERTScore
        P, R, F1 = bert_score(
            prediction,
            reference,
            model_type=model_type,
            verbose=False
        )
        
        return {
            "BERTScore-P": float(P.mean().item()),
            "BERTScore-R": float(R.mean().item()),
            "BERTScore-F1": float(F1.mean().item()),
        }
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple predictions against references.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            metrics: Metrics to compute
            
        Returns:
            Dictionary with averaged scores and per-sample scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")
        
        all_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.evaluate(pred, ref, metrics=metrics)
            all_scores.append(scores)
        
        # Aggregate scores
        aggregated = self._aggregate_scores(all_scores)
        
        return {
            "aggregated": aggregated,
            "per_sample": all_scores,
            "num_samples": len(predictions)
        }
    
    def _aggregate_scores(
        self,
        all_scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate scores across multiple samples."""
        aggregated = {}
        
        # Get all metric types
        metric_types = set()
        for scores in all_scores:
            metric_types.update(scores.keys())
        
        for metric_type in metric_types:
            # Get all sub-metrics for this type
            sub_metrics = set()
            for scores in all_scores:
                if metric_type in scores and isinstance(scores[metric_type], dict):
                    sub_metrics.update(scores[metric_type].keys())
            
            if sub_metrics:
                # Average each sub-metric
                for sub_metric in sub_metrics:
                    values = []
                    for scores in all_scores:
                        if (metric_type in scores and 
                            isinstance(scores[metric_type], dict) and
                            sub_metric in scores[metric_type]):
                            val = scores[metric_type][sub_metric]
                            if isinstance(val, (int, float)):
                                values.append(val)
                    
                    if values:
                        key = f"{metric_type}_{sub_metric}"
                        aggregated[key] = float(np.mean(values))
        
        return aggregated
    
    def print_evaluation(self, results: Dict[str, Any]) -> None:
        """Pretty print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        if "aggregated" in results:
            # Batch evaluation
            print("\nAggregated Scores:")
            print("-"*60)
            for metric, score in sorted(results["aggregated"].items()):
                print(f"{metric:30s}: {score:.4f}")
            print(f"\nNumber of samples: {results['num_samples']}")
        else:
            # Single evaluation
            for metric_type, scores in results.items():
                if isinstance(scores, dict) and "error" not in scores:
                    print(f"\n{metric_type}:")
                    print("-"*60)
                    for sub_metric, score in scores.items():
                        print(f"  {sub_metric:25s}: {score:.4f}")
                elif "error" in scores:
                    print(f"\n{metric_type}: {scores['error']}")
        
        print("\n" + "="*60 + "\n")


class RetrievalEvaluator:
    """
    Evaluate retrieval quality (independent of answer generation).
    """
    
    def __init__(self):
        logger.info("Initialized RetrievalEvaluator")
    
    def evaluate_retrieval(
        self,
        retrieved_items: List[Dict[str, Any]],
        relevant_items: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval with Precision@K, Recall@K, and MRR.
        
        Args:
            retrieved_items: List of retrieved items with 'content' field
            relevant_items: List of relevant content strings
            k_values: K values for P@K and R@K
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not retrieved_items or not relevant_items:
            return {"error": "Empty retrieved or relevant items"}
        
        # Extract retrieved content
        retrieved_content = [
            item.get("content", "").lower() for item in retrieved_items
        ]
        relevant_set = set(r.lower() for r in relevant_items)
        
        metrics = {}
        
        # Precision@K and Recall@K
        for k in k_values:
            retrieved_k = retrieved_content[:k]
            relevant_in_k = sum(1 for r in retrieved_k if any(rel in r for rel in relevant_set))
            
            precision = relevant_in_k / k if k > 0 else 0
            recall = relevant_in_k / len(relevant_set) if relevant_set else 0
            
            metrics[f"P@{k}"] = float(precision)
            metrics[f"R@{k}"] = float(recall)
        
        # Mean Reciprocal Rank (MRR)
        for i, content in enumerate(retrieved_content, 1):
            if any(rel in content for rel in relevant_set):
                metrics["MRR"] = float(1.0 / i)
                break
        else:
            metrics["MRR"] = 0.0
        
        # Mean Average Precision (MAP)
        avg_precision = self._compute_average_precision(retrieved_content, relevant_set)
        metrics["MAP"] = float(avg_precision)
        
        return metrics
    
    def _compute_average_precision(
        self,
        retrieved: List[str],
        relevant: set
    ) -> float:
        """Compute Average Precision."""
        num_relevant = 0
        sum_precision = 0.0
        
        for i, item in enumerate(retrieved, 1):
            if any(rel in item for rel in relevant):
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precision += precision_at_i
        
        if num_relevant == 0:
            return 0.0
        
        return sum_precision / len(relevant)