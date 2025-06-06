import pandas as pd
import random
from datasets import load_dataset
from transformers import pipeline
from config import Config

cfg = Config()

def get_human_texts(sample_size=500):
    """
    Load human-written texts from a public dataset.
    """
    dataset = load_dataset("ag_news", split="train")
    human_texts = [x['text'] for x in dataset.select(range(sample_size))]
    return human_texts

def get_ai_texts(prompts, sample_size=500):
    """
    Generate AI texts using a local or remote language model.
    For now, uses Hugging Face's text-generation pipeline.
    """
    generator = pipeline("text-generation", model="gpt2", max_length=100)
    ai_texts = []

    for i in range(sample_size):
        prompt = random.choice(prompts)
        response = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
        ai_texts.append(response)
    
    return ai_texts

def build_dataset(human_texts, ai_texts, output_path="data/processed/dataset.csv"):
    """
    Combine and label the dataset.
    """
    data = pd.DataFrame({
        "text": human_texts + ai_texts,
        "label": [0] * len(human_texts) + [1] * len(ai_texts)
    })

    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")

if __name__ == "__main__":
    print("Collecting human-written texts...")
    human = get_human_texts(sample_size=500)

    print("Generating AI-written texts...")
    prompts = [
        "Write a paragraph about climate change.",
        "Explain how photosynthesis works.",
        "What is quantum computing?",
        "Describe the history of the Roman Empire.",
        "Tell a story about a robot who learns to feel emotions."
    ]
    ai = get_ai_texts(prompts, sample_size=500)

    print("Building and saving dataset...")
    build_dataset(human, ai)
