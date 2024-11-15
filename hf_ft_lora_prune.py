# Progressiely prunes `model_name` while fine tuning on `dataset` with LoRA
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


def compute_global_sparsity(model):
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        # Flatten the tensor to count zeros and total elements
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    
    global_sparsity = zero_params / total_params
    return global_sparsity


class PruningCallback(TrainerCallback):
    def __init__(self, model, ptg=0.1):
        self.ptg = ptg
        self.method = prune.L1Unstructured
        self.model = model
        
        # Get the frozen pre-trained, non-LoRA parameters
        self.params = []

        # Iterate through all named modules in the model
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and 'weight' in module._parameters:  # Ensure module has a weight attribute and parameter
                if not module.weight.requires_grad and not isinstance(module, nn.Embedding):  # Filter for non-embedding and non-trainable parameters
                    assert("lora" not in name)
                    self.params.append((module, 'weight'))

    # Callback function: called at the end of each training epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nPruning {self.ptg}% of pretrained weights after epoch {state.epoch}")
        self.prune_pretrained()
        self.compute_global_sparsity()

    # Applies pruning `weight_mask` to pretrained, non-LoRA model parameters
    # Retains previous mask, allowing cumulative pruning
    def prune_pretrained(self):
        prune.global_unstructured(
            self.params,
            pruning_method=self.method,
            amount=self.ptg,
        )
            
    def compute_global_sparsity(self):
        print(
            "sparsity: {:.2f}%".format(
                100. * float(
                    torch.sum(self.model.distilbert.transformer.layer[0].attention.q_lin.weight == 0)
                )
                / float(
                    self.model.distilbert.transformer.layer[0].attention.q_lin.weight.nelement()
                )
            )
        )
        
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

    # Target LoRA modules - the attention and MLP layers
    target_modules = []
    # Iterate over the layers of the transformer to find LoRA adapter target modules
    num_layers = len(model.distilbert.transformer.layer)
    target = "distilbert.transformer.layer"
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
    pruning_callback = PruningCallback(model, ptg=0.2)

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
