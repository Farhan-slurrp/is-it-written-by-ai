import os
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.config import Config

cfg = Config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))

model_transformer = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model_transformer.eval()


with open(os.path.join(MODEL_DIR, "baseline_model.pkl"), "rb") as f:
    baseline_model = joblib.load(f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = joblib.load(f)


def predict_transformer_fn(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
    )
    with torch.no_grad():
        outputs = model_transformer(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        prob = probs[0][pred].item()

    label = "AI" if pred == 1 else "Human"
    return label, round(prob, 4)


def predict_baseline_fn(text: str):
    X = vectorizer.transform([text])
    probs = baseline_model.predict_proba(X)
    pred = baseline_model.predict(X)
    label_map = {0: "Human", 1: "AI"}

    prediction = label_map.get(pred[0], "Unknown")
    probability = probs[0][pred[0]]

    return prediction, round(probability, 4)

