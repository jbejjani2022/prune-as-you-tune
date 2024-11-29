"""
    Test / pf of concept script for loading model checkpoints,
    evaluating accuracy on test set, and calculating pseudo 
    perplexity on original masked LM task
"""


from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from peft import PeftModel

from src.perplexity import PPL


model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = "imdb"
metric = evaluate.load("accuracy")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare evaluation data
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset(dataset).map(tokenize, batched=True)
eval_dataset = dataset["test"].shuffle(seed=42)


root = 'bert-imdb-r32-nomaxlen/'
# paths = ['50pct-sparsity-5epochs/checkpoints/full_finetune/ckpt-epoch-5',
#         '50pct-sparsity-10epochs/checkpoints/full_finetune/ckpt-epoch-10',
#         '80pct-sparsity-8epochs/checkpoints/full_finetune/ckpt-epoch-8',
#         '80pct-sparsity-16epochs/checkpoints/full_finetune/ckpt-epoch-2']
paths = ['50pct-sparsity-5epochs/checkpoints/full_finetune/ckpt-epoch-5']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = PPL(model_name, device)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)   # For classification tasks
    return metric.compute(predictions=predictions, references=labels)


def eval_acc(model):
    training_args = TrainingArguments(
        output_dir="./eval_results",
        do_eval=True,
        per_device_eval_batch_size = 64,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Run evaluation
    eval_results = trainer.evaluate()
    print(eval_results)


def eval_ppl(path, model_name, evaluator, device):
    print(f'loading in {path}...')
    # base = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    # finetuned_seq_model = PeftModel.from_pretrained(base, path)
    finetuned_seq_model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    if isinstance(finetuned_seq_model, PeftModel):
        print('merging lora adapters into bert...')
        finetuned_seq_model.merge_and_unload()
    print(f'evaluating model acc on test set...')
    eval_acc(finetuned_seq_model)
    
    base_model = finetuned_seq_model.bert
    
    print(f'copying finetuned seq model to masked LM architecture...')
    masked_lm_model = AutoModelForMaskedLM.from_pretrained(model_name, config=base_model.config).to(device)
    masked_lm_model.bert = base_model
    
    print(f'calculating perplexity...')
    ppl = evaluator.calculate_perplexity_pseudo(masked_lm_model)
    print(f'perplexity = {ppl}')
    
    
if __name__ == '__main__':
    for path in paths:
        eval_ppl(root + path, model_name, evaluator, device)
    