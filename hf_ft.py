# Fine-tuning example code from Hugging Face
#     https://huggingface.co/docs/transformers/en/training
# Fine tunes `model_name` on `dataset` and saves checkpoint to `output_dir`


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate


model_name = "distilbert-base-uncased" # "google-bert/bert-base-cased"
dataset = "yelp_review_full"
output_dir = "test_trainer"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    dataset = load_dataset(dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    metric = evaluate.load("accuracy")
    
    training_args = TrainingArguments(output_dir=output_dir, eval_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()