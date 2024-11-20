from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import sys
import torch

from src.finetune_evaluators import DistilBertFineTuneEvaluator


dataset = "imdb"
pruning_method = "L1Unstructured"
output_dir = "logs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=8,
    eval_strategy="epoch",      # Evaluate every epoch
    logging_strategy="epoch",   # Log after each epoch
    save_strategy="no",
    label_names=["labels"],
    fp16=True,                  # Mixed precision training
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32
)

print(training_args.device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=256,               # Rank of LoRA
    lora_alpha=16,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
    use_rslora=True      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
)

evaluator = DistilBertFineTuneEvaluator(
    dataset=dataset,
    training_args=training_args,
    lora_config=lora_config,
    pruning_method=pruning_method,
    device=device
)
#evaluator.evaluate()
evaluator.evaluate(8, alpha=0.8, temp=2)
