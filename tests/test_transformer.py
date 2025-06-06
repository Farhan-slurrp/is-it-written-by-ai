import sys
import pytest

sys.path.append("..")

from src.config import Config
from src.data_loader import load_data, split_data
from src.transformer_model import train_transformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

cfg = Config()

def test_transformer_training():
    df = load_data(cfg.processed_data_dir / "dataset.csv").sample(100, random_state=cfg.seed)
    df_train, df_val, _, _ = split_data(df)

    model, tokenizer, trainer = train_transformer(df_train, df_val, config=cfg)

    assert model is not None, "Model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"

def test_transformer_prediction():
    model_dir = cfg.model_dir
    assert model_dir.exists(), "Saved model directory does not exist"

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Test input
    test_sentences = [
        "This paragraph is likely written by a human.",
        "The model generates statistically probable token sequences."
    ]

    encodings = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_length)
    outputs = model(**encodings)
    predictions = outputs.logits.argmax(dim=1).tolist()

    assert len(predictions) == len(test_sentences), "Must return one prediction per input"
    assert all(p in [0, 1] for p in predictions), "Predictions must be binary (0 or 1)"

    for s, p in zip(test_sentences, predictions):
        print(f"Text: {s} → Prediction: {'AI' if p == 1 else 'Human'}")

if __name__ == "__main__":
    pytest.main()
