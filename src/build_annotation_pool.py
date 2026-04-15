from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
# Make local project modules importable when running this script directly
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_retriever import DenseRetriever
from reranker import SoftWeightReranker


def deduplicate_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate documents by doc_id while keeping first occurrence order."""
    seen = set()
    out = []
    for doc in docs:
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc)
    return out
def fuse_top_docs(
    query_id: str,
    query_text: str,
    bm25_docs: List[Dict[str, Any]],
    dense_docs: List[Dict[str, Any]],
    rerank_docs: List[Dict[str, Any]],
    final_k: int = 5,
) -> List[Dict[str, Any]]:
    """Fuse BM25/Dense/Reranker results into one final top-k annotation pool."""
    weights = {"bm25": 0.25, "dense": 0.35, "rerank": 0.40}

    candidates: Dict[str, Dict[str, Any]] = {}

    def add_docs(docs: List[Dict[str, Any]], source: str) -> None:
        # Add reciprocal-rank style contribution from one retrieval source
        w = weights[source]
        for rank_idx, d in enumerate(docs):
            doc_id = str(d.get("doc_id", "")).strip()
            if not doc_id:
                continue

            item = candidates.setdefault(
                doc_id,
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "doc_id": doc_id,
                    "topic": str(d.get("topic", "")),
                    "text": str(d.get("text", d.get("content", ""))),
                    "from_set": set(),
                    "fuse_score": 0.0,
                },
            )
            item["from_set"].add(source)
            item["fuse_score"] += w * (1.0 / (rank_idx + 1))

    add_docs(bm25_docs, "bm25")
    add_docs(dense_docs, "dense")
    add_docs(rerank_docs, "rerank")
    # Small bonus for documents retrieved by multiple systems
    for item in candidates.values():
        item["fuse_score"] += 0.05 * (len(item["from_set"]) - 1)

    ranked = sorted(
        candidates.values(),
        key=lambda x: (x["fuse_score"], len(x["from_set"])),
        reverse=True,
    )[:final_k]

    out = []
    for i, item in enumerate(ranked, start=1):
        out.append(
            {
                "query_id": item["query_id"],
                "query_text": item["query_text"],
                "pool_rank": i,
                "doc_id": item["doc_id"],
                "topic": item["topic"],
                "text": item["text"],
                "from_system": "|".join(sorted(item["from_set"])),
            }
        )
    return out

def build_bm25(corpus_df: pd.DataFrame) -> BM25Okapi:
    """Build a BM25 retriever from corpus content text."""
    tokenized = [str(x).lower().split() for x in corpus_df["content"].fillna("").tolist()]
    return BM25Okapi(tokenized)


def bm25_search(query: str, corpus_df: pd.DataFrame, bm25: BM25Okapi, top_k: int) -> List[Dict[str, Any]]:
    """Search with BM25 and return top-k documents with scores."""
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        row = corpus_df.iloc[int(idx)]
        results.append(
            {
                "doc_id": str(row["doc_id"]),
                "topic": str(row.get("topic", "")),
                "text": str(row.get("content", "")),
                "retrieval_score": float(scores[idx]),
            }
        )
    return results


def main() -> None:
    """Generate fused annotation pool from multiple retrieval systems."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/annotations/eval_queries.csv")
    parser.add_argument("--qa_corpus", default="data/qa_corpus_clean.csv")
    parser.add_argument("--output", default="data/annotations/annotation_pool.csv")
    parser.add_argument("--bm25_top_k", type=int, default=20)
    parser.add_argument("--dense_top_k", type=int, default=20)
    parser.add_argument("--rerank_top_k", type=int, default=20)
    parser.add_argument("--rerank_pool_k", type=int, default=60)
    parser.add_argument("--final_k", type=int, default=5)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    corpus_path = Path(args.qa_corpus)
    output_path = Path(args.output)

    queries_df = pd.read_csv(queries_path)
    corpus_df = pd.read_csv(corpus_path)
    # Validate expected columns in query/corpus files
    required_q_cols = {"query_id", "query_text"}
    required_c_cols = {"doc_id", "topic", "content"}
    if not required_q_cols.issubset(set(queries_df.columns)):
        raise ValueError(f"eval_queries.csv must contain columns: {required_q_cols}")
    if not required_c_cols.issubset(set(corpus_df.columns)):
        raise ValueError(f"qa_corpus_clean.csv must contain columns: {required_c_cols}")
    # Normalize corpus fields for safe downstream processing
    corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
    corpus_df["topic"] = corpus_df["topic"].fillna("").astype(str)
    corpus_df["content"] = corpus_df["content"].fillna("").astype(str)

    bm25 = build_bm25(corpus_df)
    # Dense retriever is optional (continue if index/model is unavailable)
    dense = None
    try:
        dense = DenseRetriever()
        dense.load_index()
    except Exception:
        dense = None

    reranker = SoftWeightReranker(w_r=0.60, w_c=0.20, w_x=0.20, w_d=0.10, cross_encoder=None)

    rows: List[Dict[str, Any]] = []
    # Build fused top docs for each query
    for _, q_row in queries_df.iterrows():
        query_id = str(q_row["query_id"])
        query_text = str(q_row["query_text"])

        bm25_docs = bm25_search(query_text, corpus_df, bm25, top_k=args.bm25_top_k)

        dense_docs: List[Dict[str, Any]] = []
        if dense is not None:
            dense_docs = dense.search(query_text, top_k=args.dense_top_k)
        # Build rerank pool from BM25 + Dense candidates
        rerank_candidates = deduplicate_docs(
            bm25_search(query_text, corpus_df, bm25, top_k=args.rerank_pool_k)
            + (dense.search(query_text, top_k=args.rerank_pool_k) if dense is not None else [])
        )
        rerank_docs = reranker.rerank(
            query=query_text,
            analyzer_results=None,
            documents=rerank_candidates,
            top_k=args.rerank_top_k,
            enable_diversity_penalty=True,
        )

        top_docs = fuse_top_docs(
            query_id=query_id,
            query_text=query_text,
            bm25_docs=bm25_docs,
            dense_docs=dense_docs,
            rerank_docs=rerank_docs,
            final_k=args.final_k,
        )
        rows.extend(top_docs)

    output_df = pd.DataFrame(rows).sort_values(by=["query_id", "pool_rank"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved annotation pool: {output_path}")
    print(f"Total rows: {len(output_df)}")


if __name__ == "__main__":
    main()