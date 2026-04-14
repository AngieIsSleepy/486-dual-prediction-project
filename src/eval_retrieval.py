from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_retriever import DenseRetriever
from reranker import SoftWeightReranker


def deduplicate_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for d in docs:
        doc_id = str(d.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(d)
    return out


def build_bm25(corpus_df: pd.DataFrame) -> BM25Okapi:
    tokenized = [str(x).lower().split() for x in corpus_df["content"].fillna("").tolist()]
    return BM25Okapi(tokenized)


def bm25_search(query: str, corpus_df: pd.DataFrame, bm25: BM25Okapi, top_k: int) -> List[Dict[str, Any]]:
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    out: List[Dict[str, Any]] = []
    for idx in top_idx:
        row = corpus_df.iloc[int(idx)]
        out.append(
            {
                "doc_id": str(row["doc_id"]),
                "topic": str(row.get("topic", "")),
                "text": str(row.get("content", "")),
                "retrieval_score": float(scores[idx]),
            }
        )
    return out


def precision_at_k(binary_rels: List[int], k: int) -> float:
    top = binary_rels[:k]
    if k == 0:
        return 0.0
    return float(sum(top) / k)


def recall_at_k(binary_rels: List[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(sum(binary_rels[:k]) / total_relevant)


def mrr_at_k(binary_rels: List[int], k: int) -> float:
    for i, rel in enumerate(binary_rels[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(graded_rels: List[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(graded_rels[:k], start=1):
        score += (2**rel - 1) / math.log2(i + 1)
    return float(score)


def ndcg_at_k(graded_rels: List[int], ideal_graded_rels: List[int], k: int) -> float:
    dcg = dcg_at_k(graded_rels, k)
    idcg = dcg_at_k(sorted(ideal_graded_rels, reverse=True), k)
    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)


def ap_at_k(binary_rels: List[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for i, rel in enumerate(binary_rels[:k], start=1):
        if rel > 0:
            hit_count += 1
            precision_sum += hit_count / i
    return float(precision_sum / total_relevant)


def run_evaluation(
    eval_queries_path: str = "data/annotations/eval_queries.csv",
    qrels_path: str = "data/annotations/retrieval_qrels_final.csv",
    qa_corpus_path: str = "data/qa_corpus_clean.csv",
    output_dir: str = "artifacts",
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    rerank_top_k: int = 10,
    metrics_cutoff: int = 10,
    enable_diversity_penalty: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = pd.read_csv(eval_queries_path)
    qrels_df = pd.read_csv(qrels_path)
    corpus_df = pd.read_csv(qa_corpus_path)

    required_eval = {"query_id", "query_text"}
    required_qrels = {"query_id", "doc_id", "rel"}
    required_corpus = {"doc_id", "topic", "content"}

    if not required_eval.issubset(set(eval_df.columns)):
        raise ValueError(f"eval_queries.csv must contain: {required_eval}")
    if not required_qrels.issubset(set(qrels_df.columns)):
        raise ValueError(f"qrels must contain: {required_qrels}")
    if not required_corpus.issubset(set(corpus_df.columns)):
        raise ValueError(f"qa corpus must contain: {required_corpus}")

    eval_df["query_id"] = eval_df["query_id"].astype(str)
    eval_df["query_text"] = eval_df["query_text"].astype(str)

    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
    qrels_df["rel"] = qrels_df["rel"].astype(int)

    corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
    corpus_df["topic"] = corpus_df["topic"].fillna("").astype(str)
    corpus_df["content"] = corpus_df["content"].fillna("").astype(str)

    qrels_df = qrels_df.groupby(["query_id", "doc_id"], as_index=False)["rel"].max()

    qrels_map: Dict[str, Dict[str, int]] = {}
    for _, row in qrels_df.iterrows():
        qrels_map.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["rel"])

    bm25 = build_bm25(corpus_df)

    dense = None
    try:
        dense = DenseRetriever()
        dense.load_index()
    except Exception:
        dense = None

    reranker = SoftWeightReranker(w_r=0.60, w_c=0.20, w_x=0.20, w_d=0.10, cross_encoder=None)

    rows: List[Dict[str, Any]] = []

    for _, q_row in eval_df.iterrows():
        qid = q_row["query_id"]
        query = q_row["query_text"]

        bm25_docs = bm25_search(query, corpus_df, bm25, top_k=max(bm25_top_k, metrics_cutoff))
        dense_docs = dense.search(query, top_k=max(dense_top_k, metrics_cutoff)) if dense is not None else []

        rerank_candidates = deduplicate_docs(
            bm25_search(query, corpus_df, bm25, top_k=max(bm25_top_k, 50))
            + (dense.search(query, top_k=max(dense_top_k, 50)) if dense is not None else [])
        )
        reranked_docs = reranker.rerank(
            query=query,
            analyzer_results=None,
            documents=rerank_candidates,
            top_k=max(rerank_top_k, metrics_cutoff),
            enable_diversity_penalty=enable_diversity_penalty,
        )

        system_rankings = {
            "BM25": [d["doc_id"] for d in bm25_docs],
            "Dense": [d["doc_id"] for d in dense_docs],
            "Reranker": [d["doc_id"] for d in reranked_docs],
        }

        qrel_for_query = qrels_map.get(qid, {})
        total_relevant = sum(1 for rel in qrel_for_query.values() if rel > 0)
        ideal_graded = list(qrel_for_query.values())

        for system_name, ranked_doc_ids in system_rankings.items():
            graded = [int(qrel_for_query.get(doc_id, 0)) for doc_id in ranked_doc_ids[:metrics_cutoff]]
            binary = [1 if r > 0 else 0 for r in graded]

            row = {
                "query_id": qid,
                "query_text": query,
                "system": system_name,
                "p_at_5": precision_at_k(binary, 5),
                "recall_at_10": recall_at_k(binary, total_relevant=total_relevant, k=10),
                "mrr_at_10": mrr_at_k(binary, k=10),
                "ndcg_at_10": ndcg_at_k(graded, ideal_graded, k=10),
                "map_at_10": ap_at_k(binary, total_relevant=total_relevant, k=10),
            }
            rows.append(row)

    per_query_df = pd.DataFrame(rows)
    summary_df = (
        per_query_df.groupby("system", as_index=False)[
            ["p_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10", "map_at_10"]
        ]
        .mean()
        .sort_values(by="system")
        .reset_index(drop=True)
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_summary.csv"
    per_query_path = out_dir / "eval_per_query.csv"

    summary_df.to_csv(summary_path, index=False)
    per_query_df.to_csv(per_query_path, index=False)

    print("\n=== Retrieval Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-query: {per_query_path}")

    return summary_df, per_query_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_queries", default="data/annotations/eval_queries.csv")
    parser.add_argument("--qrels", default="data/annotations/retrieval_qrels_final.csv")
    parser.add_argument("--qa_corpus", default="data/qa_corpus_clean.csv")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--bm25_top_k", type=int, default=50)
    parser.add_argument("--dense_top_k", type=int, default=50)
    parser.add_argument("--rerank_top_k", type=int, default=10)
    parser.add_argument("--metrics_cutoff", type=int, default=10)
    parser.add_argument("--disable_diversity", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        eval_queries_path=args.eval_queries,
        qrels_path=args.qrels,
        qa_corpus_path=args.qa_corpus,
        output_dir=args.output_dir,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        rerank_top_k=args.rerank_top_k,
        metrics_cutoff=args.metrics_cutoff,
        enable_diversity_penalty=not args.disable_diversity,
    )


if __name__ == "__main__":
    main()