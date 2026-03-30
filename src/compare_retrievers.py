import os
import sys

# Add project root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from baselines.bm25_retriever import search as bm25_search
from dense_retriever import DenseRetriever


def print_bm25_results(query: str, k: int = 5) -> None:
    print("===== BM25 Results =====")
    results = bm25_search(query, k=k)

    for i, res in enumerate(results, 1):
        print(f"\nRank {i}")
        print(res[:250] + "...")


def print_dense_results(query: str, k: int = 5) -> None:
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
    query = "how do i handle overwhelming stress and burnout"

    print(f"Query: {query}\n")
    print_bm25_results(query, k=5)
    print_dense_results(query, k=5)