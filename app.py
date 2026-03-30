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
from reranker import CrossEncoderScorer, SoftWeightReranker


try:
    from query_analyzer import QueryAnalyzer
except Exception:
    QueryAnalyzer = None

try:
    from dense_retriever import DenseRetriever
except Exception:
    DenseRetriever = None


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
def load_cross_encoder(model_name: str) -> Optional[CrossEncoderScorer]:
    try:
        return CrossEncoderScorer(model_name=model_name)
    except Exception:
        return None


@st.cache_resource
def load_dense_retriever() -> Optional[Any]:
    if DenseRetriever is None:
        return None
    try:
        retriever = DenseRetriever()
        retriever.load_index()
        return retriever
    except Exception:
        return None


@st.cache_resource
def load_reranker(alpha: float, beta: float, gamma: float, cross_encoder_name: str, use_cross_encoder: bool) -> SoftWeightReranker:
    cross_encoder = load_cross_encoder(cross_encoder_name) if use_cross_encoder else None
    return SoftWeightReranker(alpha=alpha, beta=beta, gamma=gamma, cross_encoder=cross_encoder)


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


def render_result_cards(title: str, docs: List[Dict[str, Any]], show_scoring: bool = False) -> None:
    st.subheader(title)
    if not docs:
        st.info("No results to display yet.")
        return

    for index, doc in enumerate(docs, start=1):
        header = f"{index}. {doc.get('topic', 'Untitled')}"
        with st.expander(header, expanded=index <= 3):
            st.write(f"**Doc ID:** {doc.get('doc_id', '')}")
            if "retrieval_score" in doc:
                st.write(f"**Retrieval score:** {doc['retrieval_score']:.4f}")
            if show_scoring:
                st.write(f"**Final score:** {doc.get('final_score', 0.0):.4f}")
                st.write(f"**Category bonus:** {doc.get('category_bonus', 0.0):.4f}")
                st.write(f"**Personality bonus:** {doc.get('personality_bonus', 0.0):.4f}")
                st.write(f"**Cross-encoder score:** {doc.get('cross_encoder_score', 0.0):.4f}")
                if doc.get("matched_category"):
                    st.write(f"**Matched category:** {doc['matched_category']}")
            st.write("**Text:**")
            st.write(doc.get("text", ""))


st.set_page_config(page_title="Mental Health Support Demo", layout="wide")
st.title("Mental Health Support System Demo")
st.caption("Member D integration demo: QueryAnalyzer, DenseRetriever, soft-weight reranking, and optional cross-encoder reranking.")

with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Category weight (alpha)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    beta = st.slider("Personality weight (beta)", min_value=0.0, max_value=0.5, value=0.10, step=0.05)
    use_cross_encoder = st.checkbox("Enable cross-encoder reranking", value=False)
    gamma = st.slider("Cross-encoder weight (gamma)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    rerank_pool_size = st.slider("Rerank candidate pool", min_value=10, max_value=100, value=30, step=10)
    top_k = st.slider("Final results", min_value=3, max_value=10, value=5, step=1)
    cross_encoder_name = st.text_input(
        "Cross-encoder model",
        value="cross-encoder/ms-marco-MiniLM-L-6-v2",
        disabled=not use_cross_encoder,
    )

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
    dense_retriever = load_dense_retriever()
    reranker = load_reranker(alpha, beta, gamma, cross_encoder_name, use_cross_encoder)

    baseline_result = baseline_predictor.predict(user_input) if baseline_predictor else None
    analyzer_result = analyzer.analyze(user_input) if analyzer else fallback_analyzer_result(baseline_result)

    retrieval_source = "keyword fallback"
    retrieved_docs: List[Dict[str, Any]] = []
    if dense_retriever is not None:
        dense_results = dense_retriever.search_top100(user_input)
        retrieved_docs = dense_results[:rerank_pool_size]
        retrieval_source = "dense retriever (FAISS)"
    else:
        retrieved_docs = keyword_retrieve(user_input, top_k=rerank_pool_size)

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

    st.caption(f"Retrieval source: {retrieval_source}")
    if not retrieved_docs:
        render_result_cards(
            "Retrieved Candidates",
            [],
        )
        st.info("No retrieval results available. Make sure `data/` exists and `python3 src/build_dense_index.py` has been run.")
    else:
        render_result_cards("Retrieved Candidates", retrieved_docs, show_scoring=False)

    if not reranked_docs:
        render_result_cards("Final Reranked Results", [], show_scoring=True)
        st.info("Reranked results will appear once retrieval candidates are available.")
    else:
        render_result_cards("Final Reranked Results", reranked_docs, show_scoring=True)

    st.subheader("Pipeline Status")
    st.write(
        {
            "baseline_loaded": baseline_predictor is not None,
            "query_analyzer_loaded": analyzer is not None,
            "dense_retriever_loaded": dense_retriever is not None,
            "cross_encoder_enabled": use_cross_encoder,
            "cross_encoder_loaded": bool(reranker.cross_encoder),
        }
    )

st.markdown(
    """
    ### Current status
    - Shows baseline classification from Member A artifacts.
    - Uses Member B's QueryAnalyzer when `models/` is present.
    - Uses Member C's DenseRetriever when the FAISS index artifacts are present.
    - Falls back gracefully so the app still runs while integration is in progress.
    """
)
