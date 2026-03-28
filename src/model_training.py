import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


os.makedirs("models", exist_ok=True)

def train_mental_state_model():
    print("=== Start Training for Mental State ===")
    

    data_path = "data/mental_health_classification_clean.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Could not find {data_path}, run src/preprocessing.py")
        return
    df = pd.read_csv(data_path).dropna(subset=['text', 'coarse_label'])
    
    # (String -> Integer)
    labels = df['coarse_label'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    df['label'] = df['coarse_label'].map(label2id)
    
    hg_dataset = Dataset.from_pandas(df[['text', 'label']])
    hg_dataset = hg_dataset.train_test_split(test_size=0.2, seed=42)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = hg_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir="./results_mental",
        eval_strategy="epoch", 
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    trainer.train()
    

    trainer.save_model("models/mental_state_distilbert")
    tokenizer.save_pretrained("models/mental_state_distilbert")
    print("=== Mental State model training fisnihed. Saved to models/mental_state_distilbert ===\n")


def train_personality_model():
    print("=== Start Training Personality (MBTI)  ===")
    
    data_path = "data/mbti_clean.csv"
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}, run src/preprocessing.py")
        return

    df = pd.read_csv(data_path).dropna(subset=['text', 'label'])

    labels = df['label'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    df['label'] = df['label'].map(label2id)

    hg_dataset = Dataset.from_pandas(df[['text', 'label']])
    hg_dataset = hg_dataset.train_test_split(test_size=0.2, seed=42)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = hg_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results_mbti",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3, 
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    trainer.train()
    
    trainer.save_model("models/personality_distilbert")
    tokenizer.save_pretrained("models/personality_distilbert")
    print("=== Personality model saved in models/personality_distilbert ===\n")
if __name__ == "__main__":

    train_mental_state_model()
    train_personality_model()