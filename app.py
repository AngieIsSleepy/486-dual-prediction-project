from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_predictor import BaselinePredictor
from reranker import SoftWeightReranker


try:
    from query_analyzer import QueryAnalyzer
except Exception:
    QueryAnalyzer = None


DATA_PATH = PROJECT_ROOT / "data" / "qa_corpus_clean.csv"


@st.cache_resource
def load_baseline_predictor() -> Optional[BaselinePredictor]:
    try:
        return BaselinePredictor(
            model_path=str(PROJECT_ROOT / "baselines" / "lr_model.joblib"),
            vectorizer_path=str(PROJECT_ROOT / "baselines" / "tfidf_vectorizer.joblib"),
        )
    except FileNotFoundError:
        return None


@st.cache_resource
def load_query_analyzer() -> Optional[Any]:
    if QueryAnalyzer is None:
        return None
    try:
        return QueryAnalyzer()
    except Exception:
        return None


@st.cache_resource
def load_reranker(alpha: float) -> SoftWeightReranker:
    return SoftWeightReranker(alpha=alpha)


@st.cache_data
def load_qa_corpus() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["doc_id", "topic", "content"])
    return pd.read_csv(DATA_PATH)


def keyword_retrieve(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    df = load_qa_corpus()
    if df.empty:
        return []

    query_terms = set(query.lower().split())
    scored_docs = []
    for _, row in df.iterrows():
        content = str(row.get("content", ""))
        content_terms = set(content.lower().split())
        overlap = len(query_terms & content_terms)
        if overlap == 0:
            continue
        scored_docs.append(
            {
                "doc_id": row.get("doc_id", ""),
                "topic": row.get("topic", ""),
                "text": content,
                "retrieval_score": float(overlap),
            }
        )

    scored_docs.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return scored_docs[:top_k]


def fallback_analyzer_result(baseline_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not baseline_result:
        return {"mental_state_top5": [], "mbti_top5": []}
    return {
        "mental_state_top5": baseline_result.get("top_predictions", []),
        "mbti_top5": [],
    }


st.set_page_config(page_title="Mental Health Support Demo", layout="wide")
st.title("Mental Health Support System Demo")
st.caption("Member D integration demo: baseline classification, analysis fallback, retrieval, and reranking.")

with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Category weight (alpha)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    top_k = st.slider("Final results", min_value=3, max_value=10, value=5, step=1)

user_input = st.text_area(
    "Enter a user query",
    placeholder="I am feeling overwhelmed at work and do not know how to calm down.",
    height=140,
)

if st.button("Run Pipeline", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a query first.")
        st.stop()

    baseline_predictor = load_baseline_predictor()
    analyzer = load_query_analyzer()
    reranker = load_reranker(alpha)

    baseline_result = baseline_predictor.predict(user_input) if baseline_predictor else None
    analyzer_result = analyzer.analyze(user_input) if analyzer else fallback_analyzer_result(baseline_result)
    retrieved_docs = keyword_retrieve(user_input, top_k=top_k * 2)
    reranked_docs = reranker.rerank(user_input, analyzer_result, retrieved_docs, top_k=top_k)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline Classifier")
        if baseline_result is None:
            st.info("Baseline artifacts are missing. Run `python baselines/lr_classifier.py` first.")
        else:
            st.write(f"Predicted label: **{baseline_result['predicted_label']}**")
            if baseline_result["top_predictions"]:
                st.json(baseline_result["top_predictions"])

    with col2:
        st.subheader("Query Analyzer")
        if analyzer is None:
            st.info("Using fallback analyzer result until Member B's model folders are available.")
        else:
            st.success("Loaded Member B's QueryAnalyzer models.")
        st.json(analyzer_result)

    st.subheader("Retrieved Candidates")
    if not retrieved_docs:
        st.info("No retrieval corpus found yet. Run `python src/preprocessing.py` to create `data/qa_corpus_clean.csv`.")
    else:
        st.dataframe(pd.DataFrame(retrieved_docs), use_container_width=True)

    st.subheader("Final Reranked Results")
    if not reranked_docs:
        st.info("Reranked results will appear once retrieval candidates are available.")
    else:
        st.dataframe(pd.DataFrame(reranked_docs), use_container_width=True)

st.markdown(
    """
    ### Current status
    - Works today with the saved baseline classifier.
    - Uses a local keyword retrieval fallback if the QA corpus exists.
    - Automatically upgrades when Member B and Member C outputs are added.
    """
)
