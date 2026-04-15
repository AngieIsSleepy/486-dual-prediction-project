import pandas as pd
import re
import os
from datasets import load_dataset
from tqdm import tqdm

# Map fine-grained subreddit labels to coarse categories
COARSE_MAP = {
    'anxiety': 'Anxiety-like',
    'socialanxiety': 'Anxiety-like',
    'healthanxiety': 'Anxiety-like',
    'depression': 'Depressive/Low Mood',
    'lonely': 'Depressive/Low Mood',
    'ptsd': 'Trauma-related',
    'relationships': 'Relationship/Interpersonal',
    'divorce': 'Relationship/Interpersonal',
    'breakups': 'Relationship/Interpersonal',
    'suicidewatch': 'High-risk/Crisis',
    'mentalhealth': 'Other',
    'meditation': 'Other'
}

def clean_text(text):
    """Basic text cleaning: remove URLs, line breaks, and extra spaces"""
    if not isinstance(text, str): 
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'u/\w+|r/\w+', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def process_mental_health_data():
    """
    Load Reddit classification data, map labels, 
    clean text, stratified-sample, and export CSV.
    """
    print("--- Processing Reddit Classification Data ---")
    ds = load_dataset("kamruzzaman-asif/reddit-mental-health-classification")
    df = pd.DataFrame(ds['train'])
    
    # Keep only rows that can be mapped to coarse labels
    df['coarse_label'] = df['label'].map(COARSE_MAP)
    df = df.dropna(subset=['coarse_label'])
    
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    # Filter out texts shorter than 20 characters
    df = df[df['text'].str.len() > 20]
    
    # Stratified sampling with a per-class cap
    print("Performing stratified sampling...")
    sampled_groups = []
    for label, group in df.groupby('coarse_label'):
        n_samples = min(len(group), 8000)
        sampled_groups.append(group.sample(n=n_samples, random_state=42))
    
    df_sampled = pd.concat(sampled_groups).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output = "data/mental_health_classification_clean.csv"
    df_sampled[['text', 'coarse_label']].to_csv(output, index=False)
    print(f"Done! Exported to: {output} (Total samples: {len(df_sampled)})")

def process_qa_data():
    """
    Load QA data, clean question/answer fields, 
    build retrieval content, and export CSV.
    """
    print("\n--- Processing QA Retrieval Data ---")
    ds = load_dataset("Srishmath/mental-health-qa-bot-dataset")
    df = pd.DataFrame(ds['train'])
    df['question'] = df['input'].apply(clean_text)
    df['answer'] = df['response'].apply(clean_text)

    # Build a unified retrieval text field
    df['content'] = df['topic'] + " " + df['question'] + " " + df['answer']
    
    # Generate unique IDs for downstream tracking
    df['doc_id'] = [f"qa_{i}" for i in range(len(df))]
    
    output = "data/qa_corpus_clean.csv"
    df[['doc_id', 'topic', 'question', 'answer', 'content']].to_csv(output, index=False)
    print(f"Done! QA corpus exported to: {output}")


if __name__ == "__main__":
    process_mental_health_data()
    process_qa_data()
    print("\nData preprocessing completed! CSVs are ready for the team.")