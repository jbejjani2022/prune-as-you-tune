from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def calculate_lengths(examples):
    return {"length": [len(tokenizer.tokenize(text)) for text in examples["text"]]}

# Map the function to the dataset
lengths = dataset["train"].map(calculate_lengths, batched=True, batch_size=1000)
lengths = lengths["length"]

# Analyze the token length distribution
import numpy as np
print(f"Mean length: {np.mean(lengths):.2f}")
print(f"90th percentile: {np.percentile(lengths, 90)}")
print(f"Max length: {np.max(lengths)}")
