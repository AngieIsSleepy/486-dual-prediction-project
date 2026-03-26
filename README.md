# 486-Dual-Prediction-Project

A multi-stage mental health support system integrating Mental State Classification and Personality-aware Resource Retrieval.

## Project Architecture

### 1. Data & Baselines(Step A)
* **`src/preprocessing.py`**: Downloads datasets from Hugging Face, performs label mapping (Coarse-grained), cleaning, and stratified sampling.
* **`src/dataloader.py`**: Unified interface for loading cleaned datasets into training pipelines.
* **`baselines/lr_classifier.py`**: Traditional ML baseline using TF-IDF and Logistic Regression. Achieved **0.73 Macro F1**.
* **`baselines/bm25_retriever.py`**: Keyword-based retrieval baseline using the BM25 algorithm.

### 2. Coming Soon...
* **Step B**: Deep Learning (DistilBERT) for Query Understanding.
* **Step C**: Dense Retrieval with FAISS & Sentence-Transformers.
* **Step D**: System Integration, Soft-Weighting Reranker, and Streamlit UI.

## 🛠️ Setup & Usage

### Prerequisites
* Python 3.10+
* Virtual environment (`venv`) activated

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Pipeline
* Preprocess Data: ```python src/preprocessing.py```
* Train & Evaluate Baseline Classifier:: ```python baselines/lr_classifier.py```
* Run Baseline Retrieval: ```python baselines/bm25_retriever.py```
