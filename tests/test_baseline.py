import sys
import pytest

sys.path.append("..")

from src.config import Config
from src.data_loader import load_data, split_data, preprocess_tfidf
from src.model import train_baseline_model

cfg = Config()

def test_load_data():
    df = load_data(cfg.processed_data_dir / "dataset.csv")
    assert not df.empty, "Dataset should not be empty"
    assert "text" in df.columns and "label" in df.columns

def test_split_data():
    df = load_data(cfg.processed_data_dir / "dataset.csv")
    X_train, X_val, y_train, y_val = split_data(df, test_size=0.2)
    total = len(X_train) + len(X_val)
    assert total == len(df), "Train + Val split should equal dataset size"
    assert set(y_train.unique()).issubset({0,1}), "Labels should be binary"

def test_tfidf_preprocessing():
    df = load_data(cfg.processed_data_dir / "dataset.csv")
    X_train, X_val, y_train, y_val = split_data(df)
    X_train_tfidf, X_val_tfidf, vectorizer = preprocess_tfidf(X_train, X_val)
    # Check type and shape
    assert hasattr(X_train_tfidf, "shape")
    assert X_train_tfidf.shape[0] == len(X_train)

def test_train_and_predict():
    df = load_data(cfg.processed_data_dir / "dataset.csv")
    X_train, X_val, y_train, y_val = split_data(df)
    X_train_tfidf, X_val_tfidf, vectorizer = preprocess_tfidf(X_train, X_val)

    model = train_baseline_model(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    
    assert len(preds) == len(y_val), "Predictions must have same length as validation set"
    assert all(p in [0,1] for p in preds), "Predictions should be 0 or 1"

def test_model_prediction():
    df = load_data(cfg.processed_data_dir / "dataset.csv")
    X_train, X_val, y_train, y_val = split_data(df)
    X_train_tfidf, X_val_tfidf, vectorizer = preprocess_tfidf(X_train, X_val)
    model = train_baseline_model(X_train_tfidf, y_train)

    samples = [
        "This is a human-written sentence, with natural language style.",
        "The transformer generates text with statistical language modeling."
    ]

    X_samples_tfidf = vectorizer.transform(samples)
    predictions = model.predict(X_samples_tfidf)

    assert len(predictions) == len(samples), "Should predict one label per sample"
    assert all(pred in [0,1] for pred in predictions), "Predictions must be binary"

    for text, pred in zip(samples, predictions):
        label = "AI" if pred == 1 else "Human"
        print(f"Text: {text}\nPrediction: {label}\n")

if __name__ == "__main__":
    pytest.main()
