import pandas as pd
from rank_bm25 import BM25Okapi
import os

# Load the cleaned QA corpus produced by preprocessing
data_path = "data/qa_corpus_clean.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Please run src/preprocessing.py first.")
    exit()

df_qa = pd.read_csv(data_path)

# Baseline tokenization: split each document by whitespace
tokenized_corpus = [str(doc).split(" ") for doc in df_qa['content'].tolist()]
bm25 = BM25Okapi(tokenized_corpus)

def search(query, k=5):
    """Perform BM25 search and return top-k results"""
    tokenized_query = query.lower().split(" ")
    results = bm25.get_top_n(tokenized_query, df_qa['content'].tolist(), n=k)
    return results

# Quick local test
if __name__ == "__main__":
    test_query = "how to deal with work stress and burnout"
    print(f"\nQuery: {test_query}")
    
    search_results = search(test_query)
    for i, res in enumerate(search_results):
        # Print rank and first 200 characters of the result
        print(f"Rank {i+1}: {res[:200]}...")