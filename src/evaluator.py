"""
Abstract FineTuneEvaluator class
For evaluating finetuning methods on particular models
"""


from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig
from src.pruner import PruningCallback
from abc import ABC, abstractmethod


class FineTuneEvaluator(ABC):
    def __init__(self, model : str,
                 dataset, training_args : TrainingArguments,
                 lora_config : LoraConfig,
                 pruning_method : str):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # load and tokenize the dataset
        self.dataset = load_dataset(dataset).map(self.tokenize, batched=True)
        self.train_dataset = self.dataset["train"].shuffle(seed=42).select(range(10))
        self.eval_dataset = self.dataset["test"].shuffle(seed=42).select(range(10))
        self.training_args = training_args
        self.lora_config = lora_config
        self.pruning_method = pruning_method
        
    # Tokenizes `examples` using `self.tokenizer
    def tokenize(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Returns a list of modules in the model to which LoRA adapters will be attached
    @abstractmethod
    def get_target_modules(self):
        pass
    
    # Computes an eval metric on `eval_preds` e.g. accuracy
    # eval_preds is a list of (logits, true_label)
    @abstractmethod
    def compute_metrics(self, eval_preds):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
