from __future__ import annotations

import inspect
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

try:
    from eval_retrieval import run_evaluation
except Exception:
    run_evaluation = None


DATA_PATH = PROJECT_ROOT / "data" / "qa_corpus_clean.csv"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

CRISIS_KEYWORDS = {
    "suicide", "kill myself", "self-harm", "self harm", "overdose",
    "end my life", "want to die", "emergency", "911"
}


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
def load_reranker(
    alpha: float,
    gamma: float,
    cross_encoder_name: str,
    use_cross_encoder: bool,
    enable_diversity_penalty: bool,
    diversity_weight: float,
) -> SoftWeightReranker:
    cross_encoder = load_cross_encoder(cross_encoder_name) if use_cross_encoder else None
    init_sig = inspect.signature(SoftWeightReranker.__init__).parameters

    kwargs: Dict[str, Any] = {"cross_encoder": cross_encoder}

    if "alpha" in init_sig:
        kwargs["alpha"] = alpha
    if "gamma" in init_sig:
        kwargs["gamma"] = gamma
    if "beta" in init_sig:
        kwargs["beta"] = 0.0

    wc = alpha
    wx = gamma
    wd = diversity_weight if enable_diversity_penalty else 0.0
    wr = max(0.0, 1.0 - wc - wx)

    if "w_c" in init_sig:
        kwargs["w_c"] = wc
    if "w_x" in init_sig:
        kwargs["w_x"] = wx
    if "w_d" in init_sig:
        kwargs["w_d"] = wd
    if "w_r" in init_sig:
        kwargs["w_r"] = wr

    if "use_embedding_diversity" in init_sig:
        kwargs["use_embedding_diversity"] = enable_diversity_penalty
    if "enable_diversity_penalty" in init_sig:
        kwargs["enable_diversity_penalty"] = enable_diversity_penalty
    if "diversity_weight" in init_sig:
        kwargs["diversity_weight"] = diversity_weight
    if "delta" in init_sig:
        kwargs["delta"] = diversity_weight

    return SoftWeightReranker(**kwargs)


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
                "doc_id": str(row.get("doc_id", "")),
                "topic": str(row.get("topic", "")),
                "text": content,
                "retrieval_score": float(overlap),
            }
        )

    scored_docs.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return scored_docs[:top_k]


def fallback_analyzer_result(baseline_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not baseline_result:
        return {"mental_state_top5": []}
    return {"mental_state_top5": baseline_result.get("top_predictions", [])}


def is_crisis_query(text: str) -> bool:
    t = text.lower().strip()
    return any(k in t for k in CRISIS_KEYWORDS)


def dedupe_docs_by_doc_id(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for doc in docs:
        doc_id = str(doc.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc)
    return deduped


def run_rerank_compat(
    reranker: SoftWeightReranker,
    query: str,
    analyzer_result: Dict[str, Any],
    docs: List[Dict[str, Any]],
    top_k: int,
    enable_diversity_penalty: bool,
    diversity_weight: float,
) -> List[Dict[str, Any]]:
    rerank_sig = inspect.signature(reranker.rerank).parameters
    kwargs: Dict[str, Any] = {
        "query": query,
        "analyzer_results": analyzer_result,
        "documents": docs,
        "top_k": top_k,
    }

    if "enable_diversity_penalty" in rerank_sig:
        kwargs["enable_diversity_penalty"] = enable_diversity_penalty
    if "diversity_weight" in rerank_sig:
        kwargs["diversity_weight"] = diversity_weight
    if "delta" in rerank_sig:
        kwargs["delta"] = diversity_weight

    return reranker.rerank(**kwargs)


def render_result_cards(title: str, docs: List[Dict[str, Any]], show_scoring: bool = False) -> None:
    st.subheader(title)
    if not docs:
        st.info("No results to display yet.")
        return

    df_corpus = load_qa_corpus()

    for index, doc in enumerate(docs, start=1):
        header = f"{index}. {doc.get('topic', 'Untitled')}"
        with st.expander(header, expanded=index <= 3):
            st.write(f"**Doc ID:** {doc.get('doc_id', '')}")
            if "retrieval_score" in doc:
                st.write(f"**Retrieval score(raw):** {float(doc.get('retrieval_score', 0.0)):.4f}")

            if show_scoring:
                retrieval_norm = float(doc.get("retrieval_score_norm", doc.get("retrieval_score", 0.0)))
                ce_norm = float(doc.get("cross_encoder_score_norm", doc.get("cross_encoder_score", 0.0)))
                category_bonus = float(doc.get("category_bonus", 0.0))
                diversity_penalty = float(doc.get("diversity_penalty", 0.0))
                final_score = float(doc.get("final_score", 0.0))

                st.markdown("**Explainability Components**")
                st.write(f"- retrieval_score_norm: `{retrieval_norm:.4f}`")
                st.write(f"- category_bonus: `{category_bonus:.4f}`")
                st.write(f"- cross_encoder_score_norm: `{ce_norm:.4f}`")
                st.write(f"- diversity_penalty: `{diversity_penalty:.4f}`")
                st.write(f"- final_score: `{final_score:.4f}`")

                if doc.get("matched_category"):
                    st.write(f"**Matched category:** {doc['matched_category']}")

            doc_id = str(doc.get("doc_id", ""))
            matched_row = df_corpus[df_corpus["doc_id"].astype(str) == doc_id] if not df_corpus.empty else pd.DataFrame()
            if not matched_row.empty and "question" in matched_row.columns and "answer" in matched_row.columns:
                original_q = matched_row.iloc[0]["question"]
                original_a = matched_row.iloc[0]["answer"]
                st.info(f"**Similar User Input:**\n\n*{original_q}*")
                st.success(f"**Community's Advice:**\n\n{original_a}")
            else:
                raw_text = str(doc.get("text", ""))
                st.info(f"**Community Advice for reference:**\n\n{raw_text}")


def resolve_eval_file(uploaded_file: Any, default_path: Path, save_name: str) -> Optional[Path]:
    if uploaded_file is not None:
        upload_dir = ARTIFACT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / save_name
        save_path.write_bytes(uploaded_file.getbuffer())
        return save_path
    if default_path.exists():
        return default_path
    return None


st.set_page_config(page_title="Mental Health Support Demo", layout="wide")
st.title("Mental Health Support System Demo")
st.caption(
    "Mental-state-aware retrieval demo with explainability, diversity penalty, "
    "evaluation mode, safety warning, and CSV export."
)

with st.sidebar:
    st.header("Settings")

    alpha = st.slider("Category weight (alpha)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    use_cross_encoder = st.checkbox("Enable cross-encoder reranking", value=False)
    gamma = st.slider("Cross-encoder weight (gamma)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

    enable_diversity_penalty = st.checkbox("Enable diversity penalty", value=True)
    diversity_weight = st.slider("Diversity weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

    rerank_pool_size = st.slider("Rerank candidate pool", min_value=10, max_value=100, value=30, step=10)
    top_k = st.slider("Final results", min_value=3, max_value=10, value=5, step=1)

    cross_encoder_name = st.text_input(
        "Cross-encoder model",
        value="cross-encoder/ms-marco-MiniLM-L-6-v2",
        disabled=not use_cross_encoder,
    )

demo_tab, eval_tab = st.tabs(["Interactive Demo", "Evaluation"])

with demo_tab:
    user_input = st.text_area(
        "Enter a user query",
        placeholder="I am feeling anxious at night and cannot sleep.",
        height=140,
        key="demo_query_input",
    )

    if user_input.strip() and is_crisis_query(user_input):
        st.error(
            "⚠️ Crisis signal detected. If you are in immediate danger, call local emergency services now. "
            "You can also contact crisis hotlines in your country. "
            "This tool is for supportive information only and is NOT a substitute for professional care."
        )

    if st.button("Run Pipeline", type="primary", key="run_demo_pipeline"):
        if not user_input.strip():
            st.warning("Please enter a query first.")
            st.stop()

        baseline_predictor = load_baseline_predictor()
        analyzer = load_query_analyzer()
        dense_retriever = load_dense_retriever()
        reranker = load_reranker(
            alpha=alpha,
            gamma=gamma,
            cross_encoder_name=cross_encoder_name,
            use_cross_encoder=use_cross_encoder,
            enable_diversity_penalty=enable_diversity_penalty,
            diversity_weight=diversity_weight,
        )

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

        retrieved_docs = dedupe_docs_by_doc_id(retrieved_docs)

        reranked_docs = run_rerank_compat(
            reranker=reranker,
            query=user_input,
            analyzer_result=analyzer_result,
            docs=retrieved_docs,
            top_k=top_k,
            enable_diversity_penalty=enable_diversity_penalty,
            diversity_weight=diversity_weight,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline Classifier")
            if baseline_result is None:
                st.info("Baseline artifacts are missing. Run `python baselines/lr_classifier.py` first.")
            else:
                st.write(f"Predicted label: **{baseline_result['predicted_label']}**")
                if baseline_result.get("top_predictions"):
                    st.json(baseline_result["top_predictions"])

        with col2:
            st.subheader("Query Analyzer")
            if analyzer is None:
                st.info("Analyzer unavailable. Using fallback result.")
            else:
                st.success("QueryAnalyzer loaded.")
            st.json(analyzer_result)

        st.caption(f"Retrieval source: {retrieval_source}")
        render_result_cards("Retrieved Candidates", retrieved_docs, show_scoring=False)
        render_result_cards("Final Reranked Results", reranked_docs, show_scoring=True)

        rerank_df = pd.DataFrame(reranked_docs)
        if not rerank_df.empty:
            st.download_button(
                label="Download reranked results (CSV)",
                data=rerank_df.to_csv(index=False).encode("utf-8"),
                file_name="reranked_results.csv",
                mime="text/csv",
                key="download_rerank_csv",
            )

        st.subheader("Pipeline Status")
        st.write(
            {
                "baseline_loaded": baseline_predictor is not None,
                "query_analyzer_loaded": analyzer is not None,
                "dense_retriever_loaded": dense_retriever is not None,
                "cross_encoder_enabled": use_cross_encoder,
                "cross_encoder_loaded": bool(getattr(reranker, "cross_encoder", None)),
                "diversity_penalty_enabled": enable_diversity_penalty,
                "diversity_weight": diversity_weight,
            }
        )

with eval_tab:
    st.subheader("Retrieval Evaluation")

    default_eval_queries = ANNOTATION_DIR / "eval_queries.csv"
    default_qrels_final = ANNOTATION_DIR / "retrieval_qrels_final.csv"

    up_eval_queries = st.file_uploader("Upload eval_queries.csv (optional)", type=["csv"], key="up_eval_queries")
    up_qrels = st.file_uploader("Upload retrieval_qrels_final.csv (optional)", type=["csv"], key="up_qrels")

    st.caption(f"Default eval_queries path: {default_eval_queries}")
    st.caption(f"Default qrels path: {default_qrels_final}")

    if st.button("Run Evaluation", type="primary", key="run_eval_button"):
        eval_queries_path = resolve_eval_file(up_eval_queries, default_eval_queries, "eval_queries_uploaded.csv")
        qrels_path = resolve_eval_file(up_qrels, default_qrels_final, "qrels_uploaded.csv")

        if eval_queries_path is None or qrels_path is None:
            st.error("Missing eval inputs. Please upload files or place defaults in data/annotations/.")
        elif run_evaluation is None:
            st.error(
                "Cannot import run_evaluation from src/eval_retrieval.py. "
                "Please implement it first, then rerun."
            )
        else:
            try:
                summary_df, per_query_df = run_evaluation(
                    eval_queries_path=str(eval_queries_path),
                    qrels_path=str(qrels_path),
                    output_dir=str(ARTIFACT_DIR),
                )

                st.success("Evaluation finished.")

                preferred_cols = ["system", "nDCG@10", "MRR@10", "P@5", "Recall@10", "MAP"]
                show_cols = [c for c in preferred_cols if c in summary_df.columns]
                st.dataframe(summary_df[show_cols] if show_cols else summary_df, use_container_width=True)

                st.markdown("**Per-query details**")
                st.dataframe(per_query_df, use_container_width=True)

                st.download_button(
                    "Download eval_summary.csv",
                    summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="eval_summary.csv",
                    mime="text/csv",
                    key="download_eval_summary",
                )
                st.download_button(
                    "Download eval_per_query.csv",
                    per_query_df.to_csv(index=False).encode("utf-8"),
                    file_name="eval_per_query.csv",
                    mime="text/csv",
                    key="download_eval_per_query",
                )
            except Exception as e:
                st.exception(e)

st.markdown(
    """
### Notes
- This tool is **assistive only** and does not replace professional medical advice.
- If crisis risk is detected, users should be directed to emergency/hotline resources immediately.
"""
)