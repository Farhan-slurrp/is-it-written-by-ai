from fastapi import FastAPI
from app.schemas import TextInput, PredictionOutput
from app.inference import predict_transformer_fn, predict_baseline_fn

app = FastAPI(title="Is it written by AI?")

@app.post("/predict/transformer", response_model=PredictionOutput)
def predict_transformer(input: TextInput):
    label, prob = predict_transformer_fn(input.text)
    return {"label": label, "probability": prob}

@app.post("/predict/baseline", response_model=PredictionOutput)
def predict_baseline(input: TextInput):
    label, prob = predict_baseline_fn(input.text)
    return {"label": label, "probability": prob}