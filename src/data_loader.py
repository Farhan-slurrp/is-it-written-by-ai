import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from src.config import Config

cfg = Config()

def load_data(file_path):
    """
    Load a CSV file with 'text' and 'label' columns.
    """

    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    assert "text" in df.columns and "label" in df.columns, "CSV must contain 'text' and 'label' columns"
    return df

def split_data(df, test_size=0.2):
    """
    Split into training and validation sets.
    """
    return train_test_split(df["text"], df["label"], test_size=test_size, random_state=cfg.seed, stratify=df["label"])

def preprocess_tfidf(X_train, X_val, max_features=10000):
    """
    TF-IDF vectorization for traditional ML models.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    return X_train_tfidf, X_val_tfidf, vectorizer

def preprocess_transformer(df, tokenizer, max_length=None):
    """
    Preprocess text and labels for a Transformer model.
    """
    max_length = max_length or cfg.max_length

    if isinstance(df, pd.Series):
        texts = df.tolist()
        labels = [0] * len(texts)
    elif isinstance(df, pd.DataFrame):
        texts = df["text"].tolist()
        labels = df["label"].tolist()
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    return dataset