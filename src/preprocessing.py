import pandas as pd
import re
import os
from datasets import load_dataset
from tqdm import tqdm

# 1. Define label mapping (Based on Proposal 3.4)
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
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove Reddit usernames and subreddits
    text = re.sub(r'u/\w+|r/\w+', '', text)
    # Remove line breaks and normalize whitespace
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def process_mental_health_data():
    print("--- Processing Reddit Classification Data ---")
    ds = load_dataset("kamruzzaman-asif/reddit-mental-health-classification")
    df = pd.DataFrame(ds['train'])
    
    # Map labels and filter missing values
    df['coarse_label'] = df['label'].map(COARSE_MAP)
    df = df.dropna(subset=['coarse_label'])
    
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    # Filter out texts shorter than 20 characters
    df = df[df['text'].str.len() > 20]
    
    # --- Robust Stratified Sampling Logic ---
    print("Performing stratified sampling...")
    sampled_groups = []
    for label, group in df.groupby('coarse_label'):
        # Ensure each category has no more than 8000 samples
        n_samples = min(len(group), 8000)
        sampled_groups.append(group.sample(n=n_samples, random_state=42))
    
    df_sampled = pd.concat(sampled_groups).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output = "data/mental_health_classification_clean.csv"
    df_sampled[['text', 'coarse_label']].to_csv(output, index=False)
    print(f"Done! Exported to: {output} (Total samples: {len(df_sampled)})")

def process_qa_data():
    print("\n--- Processing QA Retrieval Data ---")
    ds = load_dataset("Srishmath/mental-health-qa-bot-dataset")
    df = pd.DataFrame(ds['train'])
    
    # Construct unified document field: topic + input + response
    df['content'] = df['topic'] + " " + df['input'] + " " + df['response']
    df['content'] = df['content'].apply(clean_text)
    
    # Generate unique IDs for tracking by members C and D
    df['doc_id'] = [f"qa_{i}" for i in range(len(df))]
    
    output = "data/qa_corpus_clean.csv"
    df[['doc_id', 'topic', 'content']].to_csv(output, index=False)
    print(f"Done! QA corpus exported to: {output}")

def process_mbti_data():
    print("\n--- Processing MBTI Data ---")
    try:
        ds = load_dataset("Shunian/kaggle-mbti-cleaned-augmented")
        df = pd.DataFrame(ds['train'])
        
        # Unify column names: rename 'posts' to 'text' if it exists
        if 'posts' in df.columns:
            df = df.rename(columns={'posts': 'text'})
        
        df['text'] = df['text'].apply(clean_text)
        
        # Sample 5000 records for the baseline
        df_sampled = df.sample(n=min(len(df), 5000), random_state=42)
        
        output = "data/mbti_clean.csv"
        df_sampled.to_csv(output, index=False)
        print(f"Done! MBTI data exported to: {output}")
    except Exception as e:
        print(f"MBTI processing failed: {e}")

if __name__ == "__main__":
    process_mental_health_data()
    process_qa_data()
    process_mbti_data()
    print("\nData preprocessing completed! CSVs are ready for the team.")