import os

# ---------- Stability settings ----------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """
    Member C: Dense Retrieval Engineer

    Responsibilities:
    1. Load cleaned QA corpus from data/qa_corpus_clean.csv
    2. Encode all QA texts into dense vectors using sentence-transformers
    3. Build a FAISS index
    4. Retrieve Top-K / Top-100 documents for a user query

    Input:
        query: str

    Output:
        List[Dict] with:
        - doc_id
        - topic
        - text
        - retrieval_score
    """

    def __init__(
        self,
        data_path: str = "data/qa_corpus_clean.csv",
        artifacts_dir: str = "artifacts",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.index_path = os.path.join(artifacts_dir, "qa_dense.index")
        self.metadata_path = os.path.join(artifacts_dir, "qa_metadata.pkl")

        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.device = "cpu"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        print(f"Loading embedding model: {embedding_model_name}")
        print(f"Using device: {self.device}")

        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=self.device
        )

        self.index = None
        self.metadata = None

    def load_corpus(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"{self.data_path} not found. Please run src/preprocessing.py first."
            )

        df = pd.read_csv(self.data_path)

        required_cols = {"doc_id", "topic", "content"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in QA corpus: {missing}")

        df["doc_id"] = df["doc_id"].astype(str)
        df["topic"] = df["topic"].fillna("").astype(str)
        df["content"] = df["content"].fillna("").astype(str)

        # remove empty rows
        df = df[df["content"].str.strip() != ""].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No usable rows found in qa_corpus_clean.csv")

        return df

    def _encode_texts_safe(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Encode texts in small CPU batches for local stability.
        """
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            print(f"Encoding batch {start} to {min(start + batch_size, len(texts))} / {len(texts)}")

            emb = self.embedding_model.encode(
                batch,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )

            emb = np.asarray(emb, dtype=np.float32)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return embeddings

    def build_index(self, batch_size: int = 4) -> None:
        df = self.load_corpus()
        texts = df["content"].tolist()

        print(f"Encoding {len(texts)} QA documents...")
        embeddings = self._encode_texts_safe(texts, batch_size=batch_size)

        print("Encoding finished.")
        print(f"Embedding shape: {embeddings.shape}")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # cosine similarity after normalized embeddings
        index.add(embeddings)

        metadata = df.to_dict(orient="records")

        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self.index = index
        self.metadata = metadata

        print(f"FAISS index saved to: {self.index_path}")
        print(f"Metadata saved to: {self.metadata_path}")

    def load_index(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"{self.index_path} not found. Please run src/build_dense_index.py first."
            )
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"{self.metadata_path} not found. Please run src/build_dense_index.py first."
            )

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k semantically similar QA documents.
        """
        if self.index is None or self.metadata is None:
            self.load_index()

        query = str(query).strip()
        if not query:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.metadata[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "topic": doc["topic"],
                "text": doc["content"],
                "retrieval_score": float(score)
            })

        return results

    def search_top100(self, query: str) -> List[Dict]:
        """
        Return Top-100 recall results for downstream reranking by Member D.
        """
        return self.search(query=query, top_k=100)


if __name__ == "__main__":
    retriever = DenseRetriever()
    retriever.load_index()

    test_query = "how to deal with work stress and burnout"
    results = retriever.search(test_query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"\nRank {i}")
        print(f"doc_id: {r['doc_id']}")
        print(f"topic: {r['topic']}")
        print(f"retrieval_score: {r['retrieval_score']:.4f}")
        print(f"text preview: {r['text'][:250]}...")