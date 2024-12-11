"""
Custom dataset for fine-tuning with support for data-mixing
"""
import torch
import copy
from datasets import load_dataset, Dataset, concatenate_datasets,ClassLabel, DatasetDict


MODEL_NAMES_ORIG_DATASETS = {
    "distilbert-base-uncased": ("wikitext", "wikitext-2-raw-v1"),
    "bert-base-uncased": ("wikitext", "wikitext-2-raw-v1"),
}

class FineTuneDataset:
    def __init__(self, 
                 model,
                 model_name: str,
                 tokenizer,
                 dataset,
                 mix_n : int,
                 sampling_strategy: str,
                 mix_strategy: str):

        device = torch.device("cuda")
        self.orig_model = copy.deepcopy(model)
        self.orig_model = self.orig_model.to(device)
        self.tokenizer = tokenizer
        self.unmixed_dataset = dataset
        self.sampling_strategy = sampling_strategy
        self.mix_strategy = mix_strategy

        orig_dataset_name, orig_dataset_dir = MODEL_NAMES_ORIG_DATASETS[model_name]
        if sampling_strategy == "first":
            print(f"Taking first {mix_n} samples from original dataset")
            self.orig_dataset = load_dataset(orig_dataset_name, orig_dataset_dir, split="train").take(mix_n)
        else: 
            print(f"Randomly sampling {mix_n} samples from original dataset")
            self.orig_dataset = load_dataset(orig_dataset_name, orig_dataset_dir, split="train").shuffle().take(mix_n)

    def _get_orig_labels(self):
        texts = self.orig_dataset["text"]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.orig_model = self.orig_model.to(torch.device("cuda"))
        inputs = inputs.to(self.orig_model.device)
        with torch.inference_mode():
            outputs = self.orig_model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        predictions = predictions.tolist()
        
        return predictions

    def get_mixed_dataset(self):
        combined_dataset = {"test": self.unmixed_dataset["test"]}
        pseudo_labels = self._get_orig_labels()
        dataset_dict = { "text" : self.orig_dataset["text"], "label" : pseudo_labels }
        pseudo_labeled_dataset = Dataset.from_dict(dataset_dict)
        pseudo_labeled_dataset = pseudo_labeled_dataset.cast_column("label", ClassLabel(names=self.unmixed_dataset["train"].features["label"].names))
        print(self.unmixed_dataset)
        combined_dataset["train"] = concatenate_datasets([pseudo_labeled_dataset, self.unmixed_dataset["train"]])

        assert(combined_dataset["train"].num_rows == self.unmixed_dataset["train"].num_rows + pseudo_labeled_dataset.num_rows)
        return DatasetDict(combined_dataset)
