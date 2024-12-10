"""
Custom dataset for fine-tuning
"""
import torch
from datasets import load_dataset, Dataset, concatenate_datasets,ClassLabel, DatasetDict
import pandas as pd

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
                 finetune_dataset_name: str, # TODO remove
                 mix_n : int,
                 sampling_strategy: str,
                 mix_strategy: str):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.orig_model = model.deepcopy()
        self.orig_model = self.orig_model.to(device)
        self.tokenizer = tokenizer
        self.unmixed_dataset = dataset
        self.sampling_strategy = sampling_strategy
        self.mix_strategy = mix_strategy
        #self.finetune_dataset_name = finetune_dataset_name

        orig_dataset_name, orig_dataset_dir = MODEL_NAMES_ORIG_DATASETS[model_name]
        if sampling_strategy == "first":
            print(f"Taking first {mix_n} samples from original dataset")
            self.orig_dataset = load_dataset(orig_dataset_name, orig_dataset_dir, split="train").take(mix_n)
        else: 
            print(f"Randomly sampling {mix_n} samples from original dataset")
            self.orig_dataset = load_dataset(orig_dataset_name, orig_dataset_dir, split="train").shuffle().take(mix_n)
        

    
    # AG 2025-12-10: Kind of unclean

    def _get_orig_labels(self):
        #inputs = self.orig_dataset.map(self.tokenize)
        texts = self.orig_dataset["text"]
        #print(texts)
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.orig_model(**inputs)
        #print(outputs)
        predictions = outputs.logits.argmax(dim=-1)
        predictions = predictions.tolist()
        #predictions = [self.unmixed_dataset["train"].features["label"].int2str(x) for x in predictions]
        #print(predictions)
        return predictions

    def get_mixed_dataset(self):
        combined_dataset = {"test": self.unmixed_dataset["test"]}
        pseudo_labels = self._get_orig_labels()
        #print("Pseudo labels", pseudo_labels)
        #print("unmixed labels", self.unmixed_dataset["train"]["label"])
        #print("Item 0 of unmixed dataset", self.unmixed_dataset["train"][0])
        dataset_dict = { "text" : self.orig_dataset["text"], "label" : pseudo_labels }
        pseudo_labeled_dataset = Dataset.from_dict(dataset_dict)
        pseudo_labeled_dataset = pseudo_labeled_dataset.cast_column("label", ClassLabel(names=self.unmixed_dataset["train"].features["label"].names))
        #print(pseudo_labeled_dataset)
        print(self.unmixed_dataset)
        combined_dataset["train"] = concatenate_datasets([pseudo_labeled_dataset, self.unmixed_dataset["train"]])

        
        #print(len(combined_dataset))
        assert(combined_dataset["train"].num_rows == self.unmixed_dataset["train"].num_rows + pseudo_labeled_dataset.num_rows)
        return DatasetDict(combined_dataset)
