"""
Main entry point for Complete RAG System with evaluation.
Supports both KB and dynamic caption cloud retrieval.
"""

import argparse
from pathlib import Path
import json

from src.pipeline import CompleteRAGPipeline
from src.utils import get_logger
from src.llm_compare import compare_llms

# metrics
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logger = get_logger("main_complete")

from dotenv import load_dotenv
load_dotenv()


def build_context_from_results(results: dict, fallback: str = "") -> str:
    """
    Tries to build a textual context from whatever structure pipeline returns.
    Looks for lists of dicts with a 'text' key across common fields.
    """
    text_chunks = []

    # 1) direct
    if isinstance(results, dict):
        # common shapes:
        # results["retrieval"]["final"] -> list[{"text": "..."}]
        # results["retrieval"]["merged_top_k"]
        # results["top_k"] / results["evidence"]
        candidates = []

        for key in ["retrieval", "evidence", "final_evidence", "merged_top_k", "final"]:
            if key in results and isinstance(results[key], list):
                candidates.extend(results[key])

        if "retrieval" in results and isinstance(results["retrieval"], dict):
            for key in ["final", "merged_top_k", "dynamic", "kb"]:
                if key in results["retrieval"] and isinstance(results["retrieval"][key], list):
                    candidates.extend(results["retrieval"][key])

        for item in candidates:
            if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                text_chunks.append(item["text"])

    context = "\n".join(ch for ch in text_chunks if ch)
    if not context:
        context = fallback
    return context


def compute_text_metrics(reference: str, candidate: str) -> dict:
    """
    ROUGE-1/ROUGE-L and BERTScore-F1 for one candidate.
    """
    ref = reference or ""
    cand = candidate or ""
    r = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge = r.score(ref, cand)
    P, R, F = bert_score([cand], [ref], lang="en", verbose=False)
    return {
        "rouge1": rouge["rouge1"].fmeasure,
        "rougeL": rouge["rougeL"].fmeasure,
        "bertscore_f1": float(F.mean()),
    }


def main():
    """Run complete RAG pipeline with evaluation"""

    parser = argparse.ArgumentParser(
        description="Complete Dynamic Multimodal RAG-LLM with Evaluation"
    )

    # Input arguments
    parser.add_argument("--image", type=str, required=True, help="Path to input medical image")
    parser.add_argument("--query", type=str, required=True, help="User query about the image")
    parser.add_argument("--ground_truth", type=str, default=None, help="Ground truth answer for evaluation")

    # Knowledge base arguments
    parser.add_argument("--use_kb", action="store_true", help="Use knowledge base retrieval")
    parser.add_argument("--kb_path", type=str, default="data/knowledge_base", help="Path to knowledge base")
    parser.add_argument("--load_kb", type=str, default=None, help="Load existing KB (name)")
    parser.add_argument("--build_kb", type=str, default=None, help="Build KB from ClipSyntel dataset (path to JSON)")

    # Caption cloud arguments
    parser.add_argument("--n_prompts", type=int, default=3, help="Number of prompts per VLM")
    parser.add_argument("--n_seeds", type=int, default=2, help="Number of seeds per prompt")
    parser.add_argument("--use_cached", action="store_true", help="Use cached caption cloud if available")

    # Retrieval arguments
    parser.add_argument("--kb_top_k", type=int, default=5, help="Top-K from knowledge base")
    parser.add_argument("--dynamic_top_k", type=int, default=5, help="Top-K from dynamic caption cloud")
    parser.add_argument("--final_top_k", type=int, default=10, help="Final top-K after merging")

    # Generation and evaluation
    parser.add_argument("--no_generation", action="store_true", help="Skip answer generation")
    parser.add_argument("--no_evaluation", action="store_true", help="Skip evaluation")

    # Output
    parser.add_argument("--output", type=str, default="results/complete_rag_results.json", help="Output path for results")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Verify image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    # Initialize pipeline
    logger.info("Initializing Complete RAG Pipeline...")
    pipeline = CompleteRAGPipeline(kb_path=args.kb_path, device=args.device, use_kb=args.use_kb)

    # Build or load knowledge base
    if args.build_kb:
        logger.info(f"Building knowledge base from {args.build_kb}...")
        pipeline.build_knowledge_base(
            dataset_path=args.build_kb,
            save_name="clipsyntel_kb",
            use_caption_cloud=False  # Set True for higher-quality KB (slower)
        )

    if args.use_kb and args.load_kb:
        logger.info(f"Loading knowledge base: {args.load_kb}...")
        pipeline.load_knowledge_base(save_name=args.load_kb)

    # Run pipeline (retrieval + optional generator/eval)
    try:
        results = pipeline.run(
            image_path=str(image_path),
            user_query=args.query,
            ground_truth_answer=args.ground_truth,
            n_prompts=args.n_prompts,
            n_seeds=args.n_seeds,
            kb_top_k=args.kb_top_k,
            dynamic_top_k=args.dynamic_top_k,
            final_top_k=args.final_top_k,
            use_cached_captions=args.use_cached,
            generate_answer=not args.no_generation,  # keep existing generator if you like
            evaluate=not args.no_evaluation and args.ground_truth is not None
        )

        # -------------------------------
        # Multi-LLM comparison (ChatGPT vs Gemini)
        # -------------------------------
        logger.info("Running multi-LLM comparison (ChatGPT vs Gemini)...")

        # Build context from retrieved evidence (robust across shapes)
        context = build_context_from_results(results) or "No context extracted."
        compare_out = compare_llms(args.query, context)

        # Compute per-model metrics vs ground_truth if provided; else vs context
        reference = args.ground_truth if args.ground_truth else context
        chatgpt_scores = compute_text_metrics(reference, compare_out["chatgpt"])
        gemini_scores  = compute_text_metrics(reference, compare_out["gemini"])
        selected_scores = compute_text_metrics(reference, compare_out["selected_answer"])

        # Attach to results JSON
        results.setdefault("multi_llm", {})
        results["multi_llm"].update({
            "query": args.query,
            "reference_type": "ground_truth" if args.ground_truth else "context",
            "reference_text": reference,
            "context_used": context,
            "chatgpt_answer": compare_out["chatgpt"],
            "gemini_answer": compare_out["gemini"],
            "judge": compare_out["judge"],  # {"better": "A"/"B", "reason": "..."}
            "selected_model": compare_out["selected_model"],
            "selected_answer": compare_out["selected_answer"],
            "scores": {
                "chatgpt": chatgpt_scores,
                "gemini": gemini_scores,
                "selected": selected_scores
            }
        })

        # Pretty print to console
        print("\n" + "="*70)
        print("MULTI-LLM COMPARISON (ChatGPT vs Gemini)")
        print("="*70)
        print(f"\nChatGPT Answer:\n{compare_out['chatgpt']}\n")
        print(f"Gemini Answer:\n{compare_out['gemini']}\n")
        print("Judge Decision:", compare_out["judge"])
        print("\nScores (vs {}):".format("Ground Truth" if args.ground_truth else "Context"))
        print("  - ChatGPT :", chatgpt_scores)
        print("  - Gemini  :", gemini_scores)
        print("  - Selected:", selected_scores)
        print("\nSelected Model:", compare_out["selected_model"])
        print("Selected Answer:\n", compare_out["selected_answer"])
        print("\n" + "="*70 + "\n")

        # Display existing pipeline results (if your pipeline prints them)
        pipeline.print_results(results)

        # Save results
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_file}")

        # Keep your original evaluation summary if available
        if "evaluation" in results:
            print("\n" + "="*70)
            print("EVALUATION SUMMARY")
            print("="*70)
            eval_res = results["evaluation"].get("answer_quality", {})
            for metric_type, scores in eval_res.items():
                if isinstance(scores, dict) and "error" not in scores:
                    print(f"\n{metric_type}:")
                    for sub_metric, score in scores.items():
                        try:
                            print(f"  {sub_metric:20s}: {float(score):.4f}")
                        except Exception:
                            print(f"  {sub_metric:20s}: {score}")
            print("\n" + "="*70 + "\n")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
