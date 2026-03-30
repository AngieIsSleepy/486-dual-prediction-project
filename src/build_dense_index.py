from dense_retriever import DenseRetriever

if __name__ == "__main__":
    retriever = DenseRetriever(
        data_path="data/qa_corpus_clean.csv",
        artifacts_dir="artifacts",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    retriever.build_index(batch_size=4)