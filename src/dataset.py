"""
Custom dataset for fine-tuning
"""
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, interleave_datasets

MODEL_NAMES_ORIG_DATASETS = {
    "distilbert-base-uncased": "bookcorpus/bookcorpus",
    "bert-base-uncased": "bookcorpus/bookcorpus",
}

class FineTuneDataset:
    def __init__(self, 
                 model,
                 model_name: str,
                 tokenize_func,
                 dataset,
                 mix_n : int,
                 sampling_strategy: str,
                 mix_strategy: str):

        self.orig_model = model
        self.tokenize = tokenize_func
        self.unmixed_dataset = dataset
        orig_dataset_name = MODEL_NAMES_ORIG_DATASETS[model_name]
        if sampling_strategy == "first":
            print(f"Taking first {mix_n} samples from original dataset")
            self.orig_dataset = load_dataset(orig_dataset_name, streaming=True)["train"].take(mix_n)
        else: # UNKNOWN IF THIS WILL WORK
            print(f"Randomly sampling {mix_n} samples from original dataset")
            self.orig_dataset = load_dataset(orig_dataset_name, streaming=True)["train"].shuffle().take(mix_n)
        self.orig_dataset_tokenized = self.orig_dataset
        self.mixing_strategy = mix_strategy
    
    # AG 2025-12-10: Kind of unclean
    def _get_orig_labels(self):
        inputs = self.orig_dataset.map(self.tokenize, batched=True)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        return predictions.tolist()

    def get_mixed_dataset(self):
        pseudo_labels = self._get_orig_labels()
        pseudo_labeled_dataset = Dataset.from_dict({
            "text": self.orig_dataset["text"],
            "label": pseudo_labels
        })
        combined_dataset = {"test": self.unmixed_dataset["test"]}
        if self.mix_strategy == "old_first":
            combined_dataset["train"] = concatenate_datasets([pseudo_labeled_dataset["train"], self.unmixed_dataset])
        else:
            combined_dataset["train"] = interleave_datasets([self.unmixed_dataset["train"], pseudo_labeled_dataset], seed=42)
        return combined_dataset
