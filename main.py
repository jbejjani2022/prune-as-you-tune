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

save_dir = '50pct-sparsity-5epochs-r64'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    eval_strategy="epoch",      # Evaluate every epoch
    logging_strategy="epoch",   # Log after each epoch
    save_strategy="no",
    fp16=True,                  # Mixed precision training
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    dataloader_num_workers=4
)

print(dataset)
print(training_args.device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=64,                # Rank of LoRA
    lora_alpha=64,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
    use_rslora=True      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
)

evaluator = BertBaseFineTuneEvaluator(
    dataset=dataset,
    training_args=training_args,
    max_length=256,
    lora_config=lora_config,
    pruning_method=pruning_method,
    sparsity_target=sparsity_target,
    alpha=0.8,
    temp=2,
    device=device,
    save_dir=save_dir
)

evaluator.evaluate()

# prune_full_finetune()
# prune_lora_finetune()
# lora_prune_interleave()
# lora_prune_kd_interleave()