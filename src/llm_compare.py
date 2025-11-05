# src/llm_compare.py
import os
import json
from typing import Dict, Any

# OpenAI (ChatGPT)
from openai import OpenAI

# Google (Gemini)
import google.generativeai as genai


def _get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _ensure_gemini():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=key)


def call_chatgpt(prompt: str, model: str = "gpt-4-turbo") -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def call_gemini(prompt: str, model: str = "gemini-pro") -> str:
    _ensure_gemini()
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(prompt)
    return (resp.text or "").strip()


def judge_answers(context: str, ans_a: str, ans_b: str, model: str = "gpt-4-turbo") -> Dict[str, Any]:
    """
    Uses ChatGPT as a judge. Returns JSON:
    {"better": "A"|"B", "reason": "..."}
    """
    judge_prompt = f"""
You are a clinical evaluation model. Compare Answer A vs Answer B for
accuracy, faithfulness to the context, and clinical safety. Pick ONE.

Context (evidence):
{context[:1500]}

Answer A (ChatGPT):
{ans_a}

Answer B (Gemini):
{ans_b}

Return valid JSON only:
{{"better": "A" or "B", "reason": "<one-sentence reason>"}}
"""
    raw = call_chatgpt(judge_prompt, model=model)
    try:
        return json.loads(raw)
    except Exception:
        # fallback if judge outputs text
        better = "A" if "A" in raw and "B" not in raw else "B"
        return {"better": better, "reason": raw.strip()}


def compare_llms(query: str, context: str, judge_model: str = "gpt-4-turbo") -> Dict[str, Any]:
    """
    Runs ChatGPT + Gemini on the same prompt, then judges them.
    Returns:
    {
      "chatgpt": "...",
      "gemini": "...",
      "judge": {"better": "A"/"B", "reason": "..."},
      "selected_model": "ChatGPT"/"Gemini",
      "selected_answer": "..."
    }
    """
    base_prompt = f"""Question: {query}

Context (evidence):
{context}

Task: Provide the most accurate, concise clinical answer grounded ONLY
in the context. If uncertain, state the most likely differential and why.
Answer format: 2-3 sentences, clinical tone.
"""
    ans_chatgpt = call_chatgpt(base_prompt)
    ans_gemini = call_gemini(base_prompt)

    judge = judge_answers(context, ans_chatgpt, ans_gemini, model=judge_model)
    selected_model = "ChatGPT" if judge.get("better", "A").upper() == "A" else "Gemini"
    selected_answer = ans_chatgpt if selected_model == "ChatGPT" else ans_gemini

    return {
        "chatgpt": ans_chatgpt,
        "gemini": ans_gemini,
        "judge": judge,
        "selected_model": selected_model,
        "selected_answer": selected_answer,
    }
