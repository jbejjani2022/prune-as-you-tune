# Fine-tuning example code from Hugging Face
#     https://huggingface.co/docs/transformers/en/training
# Fine tunes `model_name` on `dataset` and saves checkpoint to `output_dir`


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate


model_name = "distilbert-base-uncased" # "google-bert/bert-base-cased"
dataset = "imdb"
output_dir = "test_trainer"
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
    # Load in fine-tuning dataset
    dataset = load_dataset(dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    ptg = 0.1  # percentage of dataset to train / eval on
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    small_train_dataset = small_train_dataset.select(range(int(len(small_train_dataset) * ptg)))
    small_eval_dataset = small_eval_dataset.select(range(int(len(small_eval_dataset) * ptg)))
    
    print(f'{small_train_dataset.num_rows} training samples')
    print(f'{small_eval_dataset.num_rows} evaluation samples')
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",      # Evaluate every epoch
        logging_strategy="epoch",   # Log after each epoch
        logging_dir="./logs",       # Directory for logs
        save_strategy="no",
        label_names=["labels"],
        report_to="tensorboard",    # Enable TensorBoard logging
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Fine-tune the pretrained model
    trainer.train()
