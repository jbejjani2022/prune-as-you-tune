from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import torch

from src.lora_init import CustomLoraConfig

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

curlora_config = CustomLoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                # Rank of LoRA
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.1,     # Dropout rate
    sampling_method='inverted_probs'
    #inference_mode=False
    #target_modules=['attn'],
)

#svd based decomposition (as opposed to default, which I believe is gauss for rslora, though TODO verify this before running pissa experiments)
#btw: not attached to this particular svd based decomposition, tho I do think we should test one arbitrarily chosen svd approach
pissalora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                # Rank of LoRA
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
    init_lora_weights='pissa'
)

evaluator = BertBaseFineTuneEvaluator(
    dataset=dataset,
    training_args=training_args,
    max_length=None,  # set max_length = None if you don't want to truncate samples
    #lora_config=lora_config,
    lora_config=curlora_config,
    pruning_method=pruning_method,
    sparsity_target=sparsity_target,
    alpha=0.8,
    temp=2,
    device=device,
    save_dir=save_dir,
    eval_ppl=True , # evaluate perplexity on orig task after each finetuning
    num_samples = 5000
)

#evaluator.prune_lora_finetune()
#evaluator.prune_full_finetune()
#evaluator.lora_prune_interleave()
#evaluator.lora_prune_kd_interleave()

if __name__ == '__main__':  
    evaluator.prune_curlora_finetune(training_args.device)
    evaluator.curlora_prune_kd_interleave(training_args.device)

"""
evaluator.prune_lora_finetune(), setup output:

trainable params: 1,327,104 || all params: 110,810,882 || trainable%: 1.1976
Trainable params: 1327104
Total params: 110810882
Pruning-eligible params: 85543680
Sparsity of pruning-eligible params: 0.00%
Overall model sparsity: 0.00%

Pruning 50.00% of pretrained weights before finetuning

Pruning 50.00% of remaining pretrained weights before epoch 1, increasing sparsity of pretrained weights by 50.00%
Pruning-eligible params: 85543680
Sparsity of pruning-eligible params: 50.00%
Overall model sparsity: 38.60%
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
  0%|                                                                                                                                                                               | 0/1955 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
dataset: imdb
sparsity target: 0.5
num train epochs: 5
mps
25000 training samples
25000 evaluation samples
num_labels = 2
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
lora_config.r: 8
lora_config.sampling_method: inverted_probs
num samples = 4358

"""