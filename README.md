# A Context-Aware Retrieval System for Non-Clinical Mental Health Support


A mental-health support demo system that combines:

1. **Mental state classification** (baseline LR + DistilBERT analyzer)
2. **Retrieval** (BM25 + Dense FAISS retrieval)
3. **Reranking** (soft-weight reranker with optional diversity penalty and cross-encoder)
4. **Interactive UI** (Streamlit)

> This project is for research/educational support only and is **not** a substitute for professional medical advice.

---

## 1. Project Structure

```text
.
├── app.py
├── baselines/
│   ├── bm25_retriever.py
│   ├── lr_classifier.py
│   ├── lr_model.joblib                  # generated
│   └── tfidf_vectorizer.joblib          # generated
├── src/
│   ├── preprocessing.py
│   ├── dataloader.py
│   ├── model_training.py
│   ├── baseline_predictor.py
│   ├── query_analyzer.py
│   ├── dense_retriever.py
│   ├── build_dense_index.py
│   ├── reranker.py
│   ├── compare_retrievers.py
│   ├── build_annotation_pool.py
│   ├── eval_retrieval.py
│   ├── compute_agreement.py
│   └── make_paper_figures.py
├── data/
│   ├── mental_health_classification_clean.csv   # generated
│   ├── qa_corpus_clean.csv                      # generated
│   └── annotations/
│       ├── eval_queries.csv
│       ├── retrieval_qrels.csv
│       └── retrieval_qrels_final.csv
├── models/
│   └── mental_state_distilbert/                 # generated
├── artifacts/
│   ├── qa_dense.index                           # generated
│   ├── qa_metadata.pkl                          # generated
│   ├── eval_summary.csv                         # generated
│   ├── eval_per_query.csv                       # generated
│   └── paper_figures/                           # generated
└── requirements.txt
```

---

## 2. Environment Setup (from scratch)

### 2.1 Requirements

- Python 3.10+ (recommended)
- Linux/macOS/WSL
- Internet access (for Hugging Face dataset/model downloads)

### 2.2 Install

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. End-to-End Pipeline (Zero to Run)

> Run all commands from project root.

### Step 1 — Data preprocessing

Downloads datasets and generates cleaned CSV files.

```bash
python src/preprocessing.py
```

Expected outputs:

- `data/mental_health_classification_clean.csv`
- `data/qa_corpus_clean.csv`

---

### Step 2 — Train baseline classifier (LogReg + TF-IDF)

```bash
python baselines/lr_classifier.py
```

Expected outputs:

- `baselines/lr_model.joblib`
- `baselines/tfidf_vectorizer.joblib`

---

### Step 3 — (Optional) Run BM25 baseline retrieval smoke test

```bash
python baselines/bm25_retriever.py
```

This prints top-k retrieval text previews for a sample query.

---

### Step 4 — Train deep mental-state model (DistilBERT)

```bash
python src/model_training.py
```

Expected output:

- `models/mental_state_distilbert/`  
  (contains `config.json`, tokenizer files, model weights, etc.)

---

### Step 5 — Build dense retrieval index (FAISS)

```bash
python src/build_dense_index.py
```

Expected outputs:

- `artifacts/qa_dense.index`
- `artifacts/qa_metadata.pkl`

---

### Step 6 — (Optional) Compare BM25 vs Dense retrieval

```bash
python src/compare_retrievers.py
```

---

### Step 7 — Run Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (typically `http://localhost:8501`).

---

## 4. Optional: Evaluation / Annotation Workflow

### 4.1 Build annotation pool

```bash
python src/build_annotation_pool.py \
  --queries data/annotations/eval_queries.csv \
  --qa_corpus data/qa_corpus_clean.csv \
  --output data/annotations/annotation_pool.csv
```

---

### 4.2 Compute inter-annotator agreement

```bash
python src/compute_agreement.py \
  --qrels data/annotations/retrieval_qrels.csv \
  --output_json artifacts/annotation_agreement.json \
  --output_disagreements data/annotations/retrieval_disagreements.csv
```

---

### 4.3 Run retrieval evaluation

```bash
python src/eval_retrieval.py \
  --eval_queries data/annotations/eval_queries.csv \
  --qrels data/annotations/retrieval_qrels_final.csv \
  --qa_corpus data/qa_corpus_clean.csv \
  --output_dir artifacts
```

Expected outputs:

- `artifacts/eval_summary.csv`
- `artifacts/eval_per_query.csv`

---

### 4.4 Generate figures/tables

```bash
python src/make_paper_figures.py \
  --eval_per_query artifacts/eval_per_query.csv \
  --cls_data data/mental_health_classification_clean.csv \
  --out_dir artifacts/paper_figures
```

Expected outputs (examples):

- `artifacts/paper_figures/fig_per_query_ndcg_violin_box.png`
- `artifacts/paper_figures/fig_delta_ndcg_hist.png`
- `artifacts/paper_figures/table_delta_summary.csv`
- `artifacts/paper_figures/table_delta_per_query.csv`

---

## 5. Quick Start

If your package already includes all CSV/models/index artifacts, you can skip training/building and run only:

```bash
source venv/bin/activate
streamlit run app.py
```

---

## 6. Common Errors & Fixes

### 6.1 `data/xxx.csv not found`

Run preprocessing first:

```bash
python src/preprocessing.py
```

### 6.2 `Missing baseline model` / `Missing vectorizer`

Run:

```bash
python baselines/lr_classifier.py
```

### 6.3 `models/mental_state_distilbert not found`

Run:

```bash
python src/model_training.py
```

### 6.4 `qa_dense.index not found`

Run:

```bash
python src/build_dense_index.py
```

### 6.5 Hugging Face download/network failure

Retry command after network recovery; first run needs online model/dataset download.

---


## 7. Safety Note

This system may surface supportive community advice but does not provide medical diagnosis or emergency intervention.  

---