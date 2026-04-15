import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from baselines.bm25_retriever import search as bm25_search
from dense_retriever import DenseRetriever


def print_bm25_results(query: str, k: int = 5) -> None:
    """Run BM25 retrieval and print top-k text previews."""
    print("===== BM25 Results =====")
    results = bm25_search(query, k=k)

    for i, res in enumerate(results, 1):
        print(f"\nRank {i}")
        print(res[:250] + "...")


def print_dense_results(query: str, k: int = 5) -> None:
    """Run dense retrieval and print top-k results with metadata and scores."""
    print("\n===== Dense Retrieval Results =====")
    retriever = DenseRetriever()
    retriever.load_index()
    results = retriever.search(query, top_k=k)

    for i, r in enumerate(results, 1):
        print(f"\nRank {i}")
        print(f"doc_id: {r['doc_id']}")
        print(f"topic: {r['topic']}")
        print(f"retrieval_score: {r['retrieval_score']:.4f}")
        print(r["text"][:250] + "...")


if __name__ == "__main__":
    # Quick comparison between sparse (BM25) and dense retrieval outputs
    query = "how do i handle overwhelming stress and burnout"

    print(f"Query: {query}\n")
    print_bm25_results(query, k=5)
    print_dense_results(query, k=5)