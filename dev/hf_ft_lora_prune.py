# Progressiely prunes `model_name` while fine tuning on `dataset` with rsLoRA
# Saves checkpoint to `output_dir`


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
import numpy as np
import evaluate
from peft import LoraConfig, get_peft_model
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn


model_name = "distilbert-base-uncased" # "google-bert/bert-base-cased"
dataset = "yelp_review_full"
output_dir = "test_lora_prune_trainer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)   # For classification tasks
    return metric.compute(predictions=predictions, references=labels)


class PruningCallback(TrainerCallback):
    def __init__(self, model, ptg=0.1):
        self.model = model
        self.ptg = ptg  # percentage of params to prune
        self.method = prune.L1Unstructured  # pruning method
        self.params = []  # params to prune, list of tuples: (module, param_name)

        # Iterate through all modules in the model
        # and get the frozen / non-LoRA parameters to be pruned
        # skip embedding layers to mitigate accuracy degradation
        for name, module in model.named_modules():
            # Ensure module has a weight attribute and parameter
            if hasattr(module, 'weight') and 'weight' in module._parameters:
                # Filter for non-embedding and frozen / non-trainable parameters
                if not module.weight.requires_grad and not isinstance(module, nn.Embedding):
                    assert("lora" not in name)
                    self.params.append((module, 'weight'))

        # Count total number of params and number of LoRA params for sparsity reports
        self.total_params = 0
        self.lora_params = 0

        for name, param in model.named_parameters():
            if "lora" in name:
                self.lora_params += param.numel()
            self.total_params += param.numel()

    # Function called at the end of each training epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nPruning {self.ptg * 100:.2f}% of pretrained weights after epoch {state.epoch}")
        self.prune_pretrained()
        self.report_sparsity()

    # Applies pruning `weight_mask` to pretrained, non-LoRA model parameters
    # Retains previous mask, allowing cumulative pruning
    def prune_pretrained(self):
        prune.global_unstructured(
            self.params,
            pruning_method=self.method,
            amount=self.ptg,
        )
        
    # Reports sparsity of frozen / pretrained weights, as well as overall model sparsity
    def report_sparsity(self):
        zero_params = 0
        
        # Iterate over all parameters in the model
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                zero_params += torch.sum(module.weight == 0).item()
        
        sparsity_frozen = (zero_params / (self.total_params - self.lora_params)) * 100 if self.total_params > 0 else 0
        sparsity_all = (zero_params / self.total_params) * 100 if self.total_params > 0 else 0

        print(f"Pretrained Weight Sparsity: {sparsity_frozen:.2f}%")
        print(f"Overall Model Sparsity (including LoRA): {sparsity_all:.2f}%")
        
    # Removes the pruning reparameterization to make pruning permanent
    # Necessary in order to consolidate pruning changes permanently or export the model
    # Do not execute until all pruning iterations have been completed
    def remove(self):
        for (module, name) in self.params:
            prune.remove(module, name)


if __name__ == "__main__":
    dataset = load_dataset(dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    n_samples = 1000   # use 10 or less for rapid testing
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(n_samples))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(n_samples))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # A LoRA adapter will be attached to each target module
    # typically the attention and MLP layers of a transformer
    target_modules = []
    num_layers = len(model.distilbert.transformer.layer)
    target = "distilbert.transformer.layer"
    # Iterate over the transformer layers to get target modules
    for i in range(num_layers):
        target_modules.extend([
            f"{target}.{i}.attention.q_lin",    # Query projection in attention
            f"{target}.{i}.attention.k_lin",    # Key projection in attention
            f"{target}.{i}.attention.v_lin",    # Value projection in attention
            f"{target}.{i}.attention.out_lin",  # Output projection in attention
            f"{target}.{i}.ffn.lin1",           # First linear layer in FFN
            f"{target}.{i}.ffn.lin2"            # Second linear layer in FFN
        ])
    
    # Set up PEFT with LoRA configuration
    lora_config = LoraConfig(
        r=256,               # Rank of LoRA
        lora_alpha=16,       # Scaling factor
        lora_dropout=0.1,    # Dropout rate
        target_modules=target_modules,
        use_rslora=True      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
    )
    
    # Apply PEFT with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",      # Evaluate every epoch
        logging_strategy="epoch",   # Log after each epoch
        logging_dir="./logs",       # Directory for logs
        # save_strategy="epoch",    # Save checkpoint every epoch
        label_names=["labels"],
        report_to="tensorboard",    # Enable TensorBoard logging
    )
    
    # Create callback
    pruning_callback = PruningCallback(model, ptg=0.05)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[pruning_callback]
    )
    
    trainer.train()

    # Permanently consolidate pruning masks
    pruning_callback.remove()
