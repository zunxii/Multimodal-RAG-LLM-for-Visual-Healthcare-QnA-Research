import json
import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import normalize

# import your encoders + fusion
from src.embeddings.clip_image_encoder import CLIPImageEncoder
from src.embeddings.clip_text_encoder import CLIPTextEncoder
# from src.embeddings.blip_image_encoder import BLIPImageEncoder
# from src.embeddings.blip_text_encoder import BLIPTextEncoder

from src.fusion.fusion_mlp import FusionMLP

# ----------------------------
# Helpers: ensure types/shapes
# ----------------------------
def ensure_numpy_2d(x):
    """Return numpy array shape (N, D). If input is (D,), convert to (1,D)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    return x

def to_tensor(x, device=None):
    x = np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(x)
    if device:
        t = t.to(device)
    return t

# ----------------------------
# Build temporary DB
# ----------------------------
class TempMultimodalDB:
    """
    For a single image I and its caption-cloud JSON (list of dicts),
    build phi_i fused embeddings and a FAISS index.
    """

    def __init__(self, image_encoder, text_encoder, fusion: FusionMLP, device=None):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata: List[Dict[str, Any]] = []
        self.emb_matrix: np.ndarray = None  # (N, d_out)
        self.index = None
        self.d_out = fusion.mlp[-1].normalized_shape[0] if isinstance(fusion.mlp[-1], torch.nn.LayerNorm) else fusion.mlp[-1].out_features

    @classmethod
    def from_caption_json(cls, image_path: str, captions_json_path: str,
                          image_encoder=None, text_encoder=None, fusion: FusionMLP=None, device=None):
        """
        Build DB from caption-cloud JSON produced by dynamic_caption_cloud.
        captions_json is list of dicts with at least 'text' and provenance fields.
        """
        captions_json_path = Path(captions_json_path)
        assert captions_json_path.exists(), f"{captions_json_path} not found"

        with open(captions_json_path, "r") as f:
            captions = json.load(f)
        if len(captions) == 0:
            raise ValueError("No captions in json")

        # default instantiation if not provided (you can pass BLIP/CLIP instances)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if image_encoder is None:
            image_encoder = CLIPImageEncoder()
        if text_encoder is None:
            text_encoder = CLIPTextEncoder()
        if fusion is None:
            # infer dims by probing encoders
            img_emb = image_encoder.encode(image_path)    # (1, dv)
            txt_emb = text_encoder.encode(captions[0]["text"])  # (1, dt)
            dv = ensure_numpy_2d(img_emb).shape[1]
            dt = ensure_numpy_2d(txt_emb).shape[1]
            fusion = FusionMLP(dv=dv, dt=dt, d_out=512)

        wrapper = cls(image_encoder, text_encoder, fusion, device=device)

        # 1) compute single image vector z(I)
        zI = image_encoder.encode(image_path)               # numpy (1, dv)
        zI = ensure_numpy_2d(zI)
        # 2) compute all caption text embeddings τ_i
        texts = [c["text"] for c in captions]
        tau = text_encoder.encode(texts)                    # numpy (N, dt)
        tau = ensure_numpy_2d(tau)
        # 3) convert to torch and repeat image vector to match captions
        zI_t = to_tensor(np.repeat(zI, len(texts), axis=0), device=wrapper.device).float()
        tau_t = to_tensor(tau, device=wrapper.device).float()
        wrapper.fusion.to(wrapper.device)

        # 4) fuse: phi_i = fusion(zI_repeat, tau_i)
        with torch.no_grad():
            phi_t = wrapper.fusion(zI_t, tau_t)  # torch (N, d_out)
        phi = phi_t.cpu().numpy().astype(np.float32)

        # 5) normalize for cosine (row-wise)
        phi = normalize(phi, axis=1)

        # 6) build FAISS index (inner-product on normalized vectors == cosine)
        d = phi.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(phi)

        # 7) store metadata and return wrapper
        wrapper.metadata = captions
        wrapper.emb_matrix = phi
        wrapper.index = index
        return wrapper

    # ----------------------------
    # Query and retrieve
    # ----------------------------
    def query(self, image_path: str, qclin: str, top_k: int = 5):
        """
        Given the same image_path (for zq) and a clinicalized query string qclin,
        compute phi_q and return Top-K neighbors with scores and metadata.
        """
        # zq = Ev(I) (recompute or reuse - here recompute for simplicity)
        zq = self.image_encoder.encode(image_path)
        zq = ensure_numpy_2d(zq)
        tau_q = self.text_encoder.encode(qclin)
        tau_q = ensure_numpy_2d(tau_q)

        zq_t = to_tensor(zq, device=self.device).float()
        tau_q_t = to_tensor(tau_q, device=self.device).float()
        self.fusion.to(self.device)
        with torch.no_grad():
            phi_q_t = self.fusion(zq_t, tau_q_t)
        phi_q = phi_q_t.cpu().numpy().astype(np.float32)
        phi_q = normalize(phi_q, axis=1)

        D, I = self.index.search(phi_q, top_k)  # D: similarity scores, I: indices
        neighbors = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            neighbors.append({
                "score": float(score),
                "index": int(idx),
                "caption": self.metadata[idx]["text"],
                "meta": self.metadata[idx]
            })
        return neighbors

# ----------------------------
# Clinicalization (q_user -> q_clin)
# ----------------------------
def clinicalize_query_llm(quser: str, llm_call_fn=None, system_prompt: str = None) -> str:
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

# ----------------------------
# Evidence-to-candidate generator (placeholder)
# ----------------------------
def generate_candidate_answer(image_path: str, qclin: str, evidence: Dict[str, Any], generator_fn=None) -> Tuple[str, float, Dict]:
    """
    Given image, clinical query, and a single evidence bundle (caption + meta + score),
    return tuple (answer_text, p_confidence, grounding_map).
    generator_fn should implement your constrained generator G. If not provided, we use a naive template.
    p_confidence is a proxy (0..1).
    Γ (grounding_map) is a simple mapping {claim: evidence_caption} — you should implement stricter constraint in real system.
    """
    caption = evidence["caption"]
    score = evidence["score"]

    if generator_fn is None:
        # naive synthesis: short answer that refers to the caption
        answer = f"Based on the image and evidence: {caption}"
        p_conf = min(0.9, 0.2 + 0.15 * score)  # toy proxy
        grounding = {"supporting_caption": caption}
        return answer, float(p_conf), grounding

    return generator_fn(image_path, qclin, evidence)

# ----------------------------
# Fusion strategies
# ----------------------------
def centroid_fusion(answers: List[str], pns: List[float], sns: List[float], text_encoder=None, gamma=(1.0,1.0,1.0)) -> Dict[str, Any]:
    """
    Centroid embedding fusion (4.6.1).
    - answers: list of generated candidate answers an
    - pns: list of generator confidences pn
    - sns: list of retrieval scores sn
    - text_encoder: Et to embed answers (must return numpy (1,dt) or batched)
    gamma: weights tuple (γ1, γ2, γ3)
    Returns: dict with centroid vector, chosen_answer_index, centroid_vector, and weights
    """
    assert text_encoder is not None, "Provide a text_encoder to embed answers"
    embs = []
    for a in answers:
        emb = ensure_numpy_2d(text_encoder.encode(a))  # (1, dt)
        embs.append(emb[0])
    embs = np.stack(embs, axis=0)  # (N, dt)
    # compute weights: diversity term is placeholder (set to 0 if not computed)
    gamma1, gamma2, gamma3 = gamma
    diversity_term = np.zeros(len(answers))  # compute properly if you want
    weights = gamma1 * np.array(pns) + gamma2 * np.array(sns) + gamma3 * diversity_term
    # centroid
    weighted = (weights[:, None] * embs).sum(axis=0)
    centroid = weighted / (weights.sum() + 1e-12)
    # find nearest answer to centroid (by cosine with raw embeddings)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embs, centroid.reshape(1, -1)).flatten()
    best_idx = int(np.argmax(sims))
    return {
        "centroid": centroid,
        "best_answer_index": best_idx,
        "best_answer": answers[best_idx],
        "weights": weights
    }

def atomic_fact_intersection(answers: List[str], min_support: int = 2) -> Dict[str, Any]:
    """
    Very simple atomic-fact extraction by splitting on sentences and nouns.
    Real implementation: use a fact-extraction pipeline (dependency parsing, claim detection).
    Returns the set of facts that appear in at least min_support answers.
    """
    # naive: split by sentences, strip
    import re
    facts = []
    for a in answers:
        sents = re.split(r'[.?!]\s*', a)
        sents = [s.strip().lower() for s in sents if len(s.strip())>5]
        facts.append(sents)
    # count occurrences
    from collections import Counter
    cnt = Counter()
    for sents in facts:
        unique = set(sents)
        for f in unique:
            cnt[f] += 1
    retained = [f for f, c in cnt.items() if c >= min_support]
    return {"retained_facts": retained, "counts": cnt}

# ----------------------------
# Diversity & grounding score utilities
# ----------------------------
def diversity_score(metadatas: List[Dict[str, Any]], all_vlm_list: List[str] = None) -> float:
    """
    Compute diversity = number of distinct VLM models in metadatas / len(all_vlm_list)
    """
    if all_vlm_list is None:
        all_vlm_list = list({m.get("model") for m in metadatas})
    seen = set([m.get("model") for m in metadatas])
    return len(seen) / (len(all_vlm_list) + 1e-12)

def compute_final_grounding_score(sns: List[float], pns: List[float], consensus: float, diversity: float, weights=(0.25,0.25,0.25,0.25)) -> float:
    """
    Implements g* = eta1*s + eta2*p + eta3*consensus + eta4*diversity
    where s = mean(sn), p = mean(pn)
    """
    eta1, eta2, eta3, eta4 = weights
    s = float(np.mean(sns)) if len(sns)>0 else 0.0
    p = float(np.mean(pns)) if len(pns)>0 else 0.0
    return float(eta1*s + eta2*p + eta3*consensus + eta4*diversity)
