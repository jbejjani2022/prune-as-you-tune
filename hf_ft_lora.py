# Fine tunes `model_name` on `dataset` using LoRA
# Saves checkpoint to `output_dir`


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate
from peft import LoraConfig, get_peft_model


model_name = "distilbert-base-uncased" # "google-bert/bert-base-cased"
dataset = "yelp_review_full"
output_dir = "test_lora_trainer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)   # For classification tasks
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    dataset = load_dataset(dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

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
        target_modules=target_modules
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()