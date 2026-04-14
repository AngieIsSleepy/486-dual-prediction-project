from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


MENTAL_CATEGORY_KEYWORDS = {
    "Anxiety-like": ["anxiety", "stress", "panic", "worry", "burnout"],
    "Depressive/Low Mood": ["depression", "sad", "hopeless", "low mood", "lonely"],
    "Trauma-related": ["trauma", "ptsd", "abuse", "flashback"],
    "Relationship/Interpersonal": ["relationship", "partner", "friend", "family", "breakup"],
    "High-risk/Crisis": ["suicide", "self-harm", "crisis", "emergency"],
    "Other": ["mental health", "mindfulness", "meditation", "wellbeing"],
}


@dataclass
class RerankedDocument:
    doc_id: str
    text: str
    topic: str
    retrieval_score: float
    category_bonus: float
    # personality_bonus: float
    cross_encoder_score: float
    final_score: float
    matched_category: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "topic": self.topic,
            "retrieval_score": self.retrieval_score,
            "category_bonus": self.category_bonus,
            # "personality_bonus": self.personality_bonus,
            "cross_encoder_score": self.cross_encoder_score,
            "final_score": self.final_score,
            "matched_category": self.matched_category,
        }


class SoftWeightReranker:
    """
    Member D reranking logic.

    Current responsibilities:
    - Apply soft category weighting to retrieval scores.
    - Leave room for personality-aware boosting.
    - Accept an optional cross-encoder scoring function later.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        # beta: float = 0.10,
        gamma: float = 0.0,
        cross_encoder: Optional[Any] = None,
    ) -> None:
        self.alpha = alpha
        # self.beta = beta
        self.gamma = gamma
        self.cross_encoder = cross_encoder

    def rerank(
        self,
        query: str,
        analyzer_results: Optional[Dict[str, Any]],
        documents: Iterable[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        category_scores = self._extract_category_scores(analyzer_results)
        # personality_label = self._extract_personality(analyzer_results)

        reranked_docs: List[RerankedDocument] = []
        for index, doc in enumerate(documents):
            retrieval_score = float(doc.get("retrieval_score", 0.0))
            matched_category, category_probability = self._match_document_category(doc, category_scores)
            category_bonus = self.alpha * category_probability
            # personality_bonus = self.beta * self._personality_alignment(personality_label, doc)
            cross_encoder_score = self.gamma * self._cross_encoder_score(query, doc, index)

            reranked_docs.append(
                RerankedDocument(
                    doc_id=str(doc.get("doc_id", f"doc_{index}")),
                    text=str(doc.get("text") or doc.get("content") or ""),
                    topic=str(doc.get("topic", "")),
                    retrieval_score=retrieval_score,
                    category_bonus=category_bonus,
                    # personality_bonus=personality_bonus,
                    cross_encoder_score=cross_encoder_score,
                    final_score=retrieval_score + category_bonus + cross_encoder_score,
                    matched_category=matched_category,
                )
            )

        reranked_docs.sort(key=lambda item: item.final_score, reverse=True)
        return [doc.to_dict() for doc in reranked_docs[:top_k]]

    def _extract_category_scores(self, analyzer_results: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not analyzer_results:
            return {}

        if "mental_state_top5" in analyzer_results:
            return {
                str(item.get("label")): float(item.get("score", 0.0))
                for item in analyzer_results.get("mental_state_top5", [])
            }

        if "categories" in analyzer_results:
            return {
                str(label): float(score)
                for label, score in analyzer_results.get("categories", {}).items()
            }

        return {}

    # def _extract_personality(self, analyzer_results: Optional[Dict[str, Any]]) -> Optional[str]:
    #     if not analyzer_results:
    #         return None

    #     mbti_top5 = analyzer_results.get("mbti_top5", [])
    #     if mbti_top5:
    #         return str(mbti_top5[0].get("label"))

    #     personality = analyzer_results.get("personality")
    #     if isinstance(personality, str):
    #         return personality

    #     return None

    def _match_document_category(
        self,
        document: Dict[str, Any],
        category_scores: Dict[str, float],
    ) -> tuple[Optional[str], float]:
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

        return best_category, best_probability

    # def _personality_alignment(self, personality_label: Optional[str], document: Dict[str, Any]) -> float:
    #     if not personality_label:
    #         return 0.0
    #     user_mbti = personality_label.upper()
    #     text_upper = str(document.get("text") or document.get("content") or "").upper()
    #     doc_mbti_list = []
    #     if "mbti_top5" in document:
    #         doc_mbti_list = [str(item.get("label", "")).upper() for item in document["mbti_top5"]]
    #         if user_mbti in doc_mbti_list:
    #             return 1.0
            
    #     if f'"LABEL":"{user_mbti}"' in text_upper or f"'LABEL':'{user_mbti}'" in text_upper or f'"LABEL": "{user_mbti}"' in text_upper:
    #         return 1.0
    #     return 0.0

    def _cross_encoder_score(
        self,
        query: str,
        # personality_label: Optional[str],
        document: Dict[str, Any],
        index: int,
    ) -> float:
        if self.cross_encoder is None:
            return 0.0

        if hasattr(self.cross_encoder, "score"):
            return float(self.cross_encoder.score(query, document))

        if callable(self.cross_encoder):
            return float(self.cross_encoder(query, document, index))

        return 0.0


class CrossEncoderScorer:
    """
    Lightweight wrapper for an optional cross-encoder reranking stage.

    It combines the user query with the predicted personality label so the final
    ranking can consider both the immediate problem and the user's style.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers CrossEncoder is unavailable. "
                "Please install sentence-transformers to enable cross-encoder reranking."
            )
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(
        self,
        query: str,
        # personality_label: Optional[str],
        document: Dict[str, Any],
    ) -> float:
        # personality_text = ""
        # if personality_label:
        #     personality_text = f" Personality style: {personality_label}."

        pair_query = f"User query: {query}"
        document_text = str(document.get("text") or document.get("content") or "")
        score = self.model.predict([(pair_query, document_text)], convert_to_numpy=True)[0]
        return float(score)
