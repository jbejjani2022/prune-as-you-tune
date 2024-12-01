from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import torch

from src.finetune_evaluators import DistilBertFineTuneEvaluator, BertBaseFineTuneEvaluator


dataset = "imdb"
pruning_method = "L1Unstructured"
output_dir = "logs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how sparse the pruned models should be after all training epochs
# for interleaving methods:
# pruning percentage per epoch = sparsity_target / num_train_epochs
sparsity_target = 0.50
num_epochs = 5

save_dir = 'ppl_tests/50pct-sparsity-5epochs-before-remove'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    eval_strategy="epoch",      # Evaluate every epoch
    logging_strategy="epoch",   # Log after each epoch
    save_strategy="no",
    fp16=True,                  # Mixed precision training
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    dataloader_num_workers=1
)

print(f'dataset: {dataset}')
print(f'sparsity target: {sparsity_target}')
print(f'num train epochs: {num_epochs}')
print(training_args.device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                # Rank of LoRA
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
    use_rslora=True      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
)

evaluator = BertBaseFineTuneEvaluator(
    dataset=dataset,
    training_args=training_args,
    max_length=5000,  # set max_length = None if you don't want to truncate samples
    lora_config=lora_config,
    pruning_method=pruning_method,
    sparsity_target=sparsity_target,
    alpha=0.8,
    temp=2,
    device=device,
    save_dir=save_dir,
    eval_ppl=True  # evaluate perplexity on orig task after each finetuning
)

#evaluator.prune_lora_finetune()
#evaluator.prune_full_finetune()
# evaluator.lora_prune_interleave()
evaluator.lora_prune_kd_interleave()
