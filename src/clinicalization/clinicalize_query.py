from llm_call import llm_call_fn

def clinicalize_query_llm(quser: str, system_prompt: str = None) -> str:
    """
    Turn free-form user query into a clinicalized query string q_clin.
    llm_call_fn should be a function that accepts (prompt) and returns text.
    If not provided, a simple identity is used (no-op) — replace with your LLM adapter.
    """
    if llm_call_fn is None:
        # no-op (return input). In practice replace with an LLM call (Gemini/GPT4) that rewrites queries.
        return quser

    system = system_prompt or (
        "Rewrite the user query into a concise clinical query suitable for medical retrieval. "
        "Keep medical entities normalized and keep it short (one sentence)."
    )
    prompt = f"{system}\n\nUser query: {quser}\n\nClinicalized query:"
    qclin = llm_call_fn(prompt)
    return qclin.strip()