from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:
    CrossEncoder = None
    SentenceTransformer = None


MENTAL_CATEGORY_KEYWORDS = {
    "Anxiety-like": ["anxiety", "stress", "panic", "worry", "burnout"],
    "Depressive/Low Mood": ["depression", "sad", "hopeless", "low mood", "lonely"],
    "Trauma-related": ["trauma", "ptsd", "abuse", "flashback"],
    "Relationship/Interpersonal": ["relationship", "partner", "friend", "family", "breakup"],
    "High-risk/Crisis": ["suicide", "self-harm", "crisis", "emergency"],
    "Other": ["mental health", "mindfulness", "meditation", "wellbeing"],
}


def minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    denom = (max_v - min_v) + 1e-8
    return [(v - min_v) / denom for v in values]


@dataclass
class RerankedDocument:
    doc_id: str
    text: str
    topic: str

    retrieval_score: float
    retrieval_score_norm: float
    category_bonus: float
    cross_encoder_score: float
    cross_encoder_score_norm: float
    diversity_penalty: float
    final_score: float
    matched_category: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "topic": self.topic,
            "retrieval_score": self.retrieval_score,
            "retrieval_score_norm": self.retrieval_score_norm,
            "category_bonus": self.category_bonus,
            "cross_encoder_score": self.cross_encoder_score,
            "cross_encoder_score_norm": self.cross_encoder_score_norm,
            "diversity_penalty": self.diversity_penalty,
            "final_score": self.final_score,
            "matched_category": self.matched_category,
        }


class SoftWeightReranker:
    """
    final_score = w_r * retrieval_score_norm
                + w_c * category_bonus
                + w_x * cross_encoder_score_norm
                - w_d * diversity_penalty
    """

    def __init__(
        self,
        w_r: float = 0.55,
        w_c: float = 0.20,
        w_x: float = 0.25,
        w_d: float = 0.10,
        cross_encoder: Optional[Any] = None,
        use_embedding_diversity: bool = True,
        diversity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.w_r = w_r
        self.w_c = w_c
        self.w_x = w_x
        self.w_d = w_d
        self.cross_encoder = cross_encoder

        self.diversity_encoder = None
        if use_embedding_diversity and SentenceTransformer is not None:
            try:
                self.diversity_encoder = SentenceTransformer(diversity_model_name)
            except Exception:
                self.diversity_encoder = None

    def rerank(
        self,
        query: str,
        analyzer_results: Optional[Dict[str, Any]],
        documents: Iterable[Dict[str, Any]],
        top_k: int = 10,
        enable_diversity_penalty: bool = True,
    ) -> List[Dict[str, Any]]:
        docs = list(documents)
        if not docs:
            return []

        category_scores = self._extract_category_scores(analyzer_results)

        retrieval_raw = [float(doc.get("retrieval_score", 0.0)) for doc in docs]
        cross_raw = [self._cross_encoder_score(query, doc) for doc in docs]

        retrieval_norm = minmax_normalize(retrieval_raw)
        cross_norm = minmax_normalize(cross_raw)

        candidates: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            matched_category, category_probability = self._match_document_category(doc, category_scores)

            candidates.append(
                {
                    "doc_id": str(doc.get("doc_id", f"doc_{i}")),
                    "text": str(doc.get("text") or doc.get("content") or ""),
                    "topic": str(doc.get("topic", "")),
                    "retrieval_score": retrieval_raw[i],
                    "retrieval_score_norm": retrieval_norm[i] if i < len(retrieval_norm) else 0.0,
                    "category_bonus": float(category_probability),
                    "cross_encoder_score": cross_raw[i],
                    "cross_encoder_score_norm": cross_norm[i] if i < len(cross_norm) else 0.0,
                    "diversity_penalty": 0.0,
                    "final_score": 0.0,
                    "matched_category": matched_category,
                }
            )

        embedding_map: Dict[str, np.ndarray] = {}
        if enable_diversity_penalty and self.diversity_encoder is not None:
            texts = [c["text"] for c in candidates]
            embs = self.diversity_encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for c, emb in zip(candidates, embs):
                embedding_map[c["doc_id"]] = emb

        selected: List[Dict[str, Any]] = []
        remaining = candidates.copy()
        k = min(top_k, len(remaining))

        for _ in range(k):
            best_idx = -1
            best_score = -1e18
            best_penalty = 0.0

            for idx, cand in enumerate(remaining):
                diversity_penalty = 0.0
                if enable_diversity_penalty and self.w_d > 0:
                    diversity_penalty = self._max_similarity_penalty(cand, selected, embedding_map)

                score = (
                    self.w_r * cand["retrieval_score_norm"]
                    + self.w_c * cand["category_bonus"]
                    + self.w_x * cand["cross_encoder_score_norm"]
                    - self.w_d * diversity_penalty
                )

                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_penalty = diversity_penalty

            chosen = remaining.pop(best_idx)
            chosen["diversity_penalty"] = best_penalty
            chosen["final_score"] = best_score
            selected.append(chosen)

        result_docs: List[RerankedDocument] = []
        for item in selected:
            result_docs.append(
                RerankedDocument(
                    doc_id=item["doc_id"],
                    text=item["text"],
                    topic=item["topic"],
                    retrieval_score=item["retrieval_score"],
                    retrieval_score_norm=item["retrieval_score_norm"],
                    category_bonus=item["category_bonus"],
                    cross_encoder_score=item["cross_encoder_score"],
                    cross_encoder_score_norm=item["cross_encoder_score_norm"],
                    diversity_penalty=item["diversity_penalty"],
                    final_score=item["final_score"],
                    matched_category=item["matched_category"],
                )
            )

        return [d.to_dict() for d in result_docs]

    def _extract_category_scores(self, analyzer_results: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not analyzer_results:
            return {}

        if "categories" in analyzer_results and isinstance(analyzer_results["categories"], dict):
            return {str(k): float(v) for k, v in analyzer_results["categories"].items()}

        if "mental_state_top5" in analyzer_results:
            out: Dict[str, float] = {}
            for item in analyzer_results.get("mental_state_top5", []):
                out[str(item.get("label"))] = float(item.get("score", 0.0))
            return out

        return {}

    def _match_document_category(
        self,
        document: Dict[str, Any],
        category_scores: Dict[str, float],
    ) -> Tuple[Optional[str], float]:
        if not category_scores:
            return None, 0.0

        topic = str(document.get("topic", "")).lower()
        text = str(document.get("text") or document.get("content") or "").lower()
        haystack = f"{topic} {text}"

        best_category = None
        best_probability = 0.0

        for category, probability in category_scores.items():
            if category in MENTAL_CATEGORY_KEYWORDS:
                if any(keyword in haystack for keyword in MENTAL_CATEGORY_KEYWORDS[category]):
                    if probability > best_probability:
                        best_category = category
                        best_probability = probability
            elif category.lower() in haystack and probability > best_probability:
                best_category = category
                best_probability = probability

        return best_category, float(best_probability)

    def _cross_encoder_score(self, query: str, document: Dict[str, Any]) -> float:
        if self.cross_encoder is None:
            return 0.0

        if hasattr(self.cross_encoder, "score"):
            try:
                return float(self.cross_encoder.score(query, document))
            except TypeError:
                # backward compatibility if old signature exists
                return float(self.cross_encoder.score(query, None, document))

        if callable(self.cross_encoder):
            try:
                return float(self.cross_encoder(query, document))
            except TypeError:
                return float(self.cross_encoder(query, None, document, 0))

        return 0.0

    def _max_similarity_penalty(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
        embedding_map: Dict[str, np.ndarray],
    ) -> float:
        if not selected:
            return 0.0

        cand_id = candidate["doc_id"]

        # embedding-based cosine similarity penalty
        if cand_id in embedding_map and embedding_map:
            cand_emb = embedding_map[cand_id]
            sims = []
            for sel in selected:
                sel_id = sel["doc_id"]
                if sel_id in embedding_map:
                    sim = float(np.dot(cand_emb, embedding_map[sel_id]))
                    sims.append(max(0.0, min(1.0, sim)))
            if sims:
                return float(max(sims))

        # fallback heuristic
        cand_topic = str(candidate.get("topic", "")).strip().lower()
        cand_text = str(candidate.get("text", "")).strip().lower()

        penalties = []
        for sel in selected:
            sel_topic = str(sel.get("topic", "")).strip().lower()
            sel_text = str(sel.get("text", "")).strip().lower()

            if cand_topic and cand_topic == sel_topic:
                penalties.append(1.0)
                continue

            if cand_text and sel_text and cand_text[:120] == sel_text[:120]:
                penalties.append(0.8)
                continue

            cand_tokens = set(cand_text.split())
            sel_tokens = set(sel_text.split())
            if cand_tokens and sel_tokens:
                overlap = len(cand_tokens & sel_tokens) / max(1, len(cand_tokens | sel_tokens))
                penalties.append(float(overlap))

        return float(max(penalties)) if penalties else 0.0


class CrossEncoderScorer:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers CrossEncoder is unavailable. "
                "Please install sentence-transformers."
            )
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, document: Dict[str, Any]) -> float:
        pair_query = f"User query: {query}"
        document_text = str(document.get("text") or document.get("content") or "")
        score = self.model.predict([(pair_query, document_text)], convert_to_numpy=True)[0]
        return float(score)