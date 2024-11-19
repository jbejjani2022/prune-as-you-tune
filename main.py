from transformers import TrainingArguments
from peft import LoraConfig

from src.finetune_evaluators import BERTFineTuneEvaluator


model = "distilbert-base-uncased"
dataset = "imdb"
pruning_method = "L1Unstructured"
output_dir = 'test_trainer'

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",      # Evaluate every epoch
    logging_strategy="epoch",   # Log after each epoch
    logging_dir="./logs",       # Directory for logs
    save_strategy="no",
    label_names=["labels"],
    report_to="tensorboard",    # Enable TensorBoard logging
)

lora_config = LoraConfig(
    r=256,               # Rank of LoRA
    lora_alpha=16,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
    use_rslora=True      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
)

evaluator = BERTFineTuneEvaluator(
    model=model,
    dataset=dataset,
    training_args=training_args,
    lora_config=lora_config,
    pruning_method=pruning_method
)
evaluator.evaluate()
