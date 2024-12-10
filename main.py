from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import torch
import typer
from typing import Optional, Annotated
from src.finetune_evaluators import DistilBertFineTuneEvaluator, BertBaseFineTuneEvaluator
import os
import numpy as np

app = typer.Typer()

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

@app.command()
# AG 24-12-10: pass dataset args to evaluator
def run_and_eval (model_name : Annotated[Optional[str], typer.Option(help="Model name to use for fine-tuning")] = "bert-base-uncased",
          n_samples : Annotated[Optional[int], typer.Option(help="Number of samples, use 10 or less for rapid testing")] = 1000,
          sparsity_target: Annotated[Optional[float], typer.Option(help="Target percentage of parameters to prune")] = 0.5,
          num_epochs : Annotated[Optional[int], typer.Option(help="Number of training epochs")] = 5,
          output_dir : Annotated[Optional[str], typer.Option(help="Output directory for logs and model checkpoints")] = "logs",
          dataset : Annotated[Optional[str], typer.Option(help="Dataset to use for fine-tuning")] = "imdb",
          dataset_mix_ptg : Annotated[Optional[float], typer.Option(help="Percentage of orig dataset samples to include - in training only")] = 0.05,
          dataset_mix_strategy : Annotated[Optional[str], typer.Option(help="Mixing strategy for dataset - 'old_first' or 'random'")] = "random",
          dataset_sampling_strategy : Annotated[Optional[str], typer.Option(help="Sampling strategy for old samples dataset - 'first' or 'random' ")] = "first",
          full_evaluate : Annotated[Optional[bool], typer.Option(help="Evaluate using all variations of pruning methods, as well as full fine-tuning")] = False,
          use_lora : Annotated[Optional[bool], typer.Option(help="Use LoRA adapters for fine-tuning")] = True,
          use_kd : Annotated[Optional[bool], typer.Option(help="Use knowledge distillation for fine-tuning")] = True,
          kd_alpha : Annotated[Optional[float], typer.Option(help="Alpha parameter for knowledge distillation")] = 0.5,
          kd_temp : Annotated[Optional[float], typer.Option(help="Temperature parameter for knowledge distillation")] = 4,
          use_rs_lora : Annotated[Optional[bool], typer.Option(help="Use rsLoRA adapters for fine-tuning")] = True,
          lora_dropout : Annotated[Optional[float], typer.Option(help="Dropout rate for LoRA adapters")] = 0.1,
          lora_alpha : Annotated[Optional[float], typer.Option(help="Scaling factor for LoRA adapters")] = 32,
          lora_rank : Annotated[Optional[int], typer.Option(help="Rank of LoRA adapters")] = 32,
          max_length: Annotated[Optional[int], typer.Option(help="Maximum length of input sequences")] = 512,
          pruning_schedule : Annotated[Optional[str], typer.Option(help="Pruning schedule - can be agp or linear")] = "linear",
          prune_every_epoch : Annotated[Optional[int], typer.Option(help="Prune at every n epoch")] = 1,
          start_pruning_epoch_ptg : Annotated[Optional[float], typer.Option(help="Start pruning at n percent epoch of training - will be rounded down")] = 0,
          pruning_method : Annotated[Optional[str], typer.Option(help="Pruning method - can be L1Unstructured or L2Structured")] = "L1Unstructured"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # how sparse the pruned models should be after all training epochs
    # for interleaving methods:
    # pruning percentage per epoch = sparsity_target / num_train_epochs

    pruning_schedule = pruning_schedule.lower()
    if pruning_schedule != "agp":
        pruning_schedule = "linear"

    pruning_start_epoch = int(np.floor(num_epochs * start_pruning_epoch_ptg))

    if pruning_start_epoch >= num_epochs:
        pruning_start_epoch = num_epochs - 1

    save_dir = f"bert-{dataset}-{max_length}/{pruning_method}-{sparsity_target}sparsity-{num_epochs}epochs-{pruning_schedule}{prune_every_epoch}prune-start{pruning_start_epoch}-kd{use_kd}-alpha{kd_alpha}-temp{kd_temp}-lora{use_lora}-lorarank{lora_rank}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_args = {"dataset_name": dataset, 
                    "mix_n": int(np.ceil(dataset_mix_ptg * n_samples)),
                    "sampling_strategy": dataset_sampling_strategy,
                    "mix_strategy": dataset_mix_strategy}

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",      # Evaluate every epoch
        logging_strategy="epoch",   # Log after each epoch
        save_strategy="no",
        fp16=True,                  # Mixed precision training
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        dataloader_num_workers=1,
    )

    print(f'dataset_args: {dataset_args}')
    print(f'sparsity target: {sparsity_target}')
    print(f'num train epochs: {num_epochs}')
    print(training_args.device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,                # Rank of LoRA
        lora_alpha=lora_alpha,       # Scaling factor
        lora_dropout=lora_dropout,    # Dropout rate
        use_rslora=use_rs_lora      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
    )


    pruning_args = {"method" : pruning_method,
                    "sparsity_target" : sparsity_target, 
                    "num_epochs" : num_epochs, 
                    "schedule" : pruning_schedule,
                    "prune_every_epoch" : prune_every_epoch,
                    "pruning_start_epoch" : pruning_start_epoch}

    loss_args = {"alpha": kd_alpha, 
                 "temp": kd_temp, 
                 "use_kd_loss": use_kd}

    if model_name == "distilbert-base-uncased":
            evaluator = DistilBertFineTuneEvaluator(
            n_samples=n_samples,
            dataset_args=dataset_args,
            training_args=training_args,
            max_length=None,  # set max_length = None if you don't want to truncate samples
            lora_config=lora_config,
            device=device,
            save_dir=save_dir,
            pruning_args = pruning_args,
            loss_args = loss_args,
            eval_ppl=True
        )
    else:  
        evaluator = BertBaseFineTuneEvaluator(
            n_samples=n_samples,
            dataset_args=dataset_args,
            training_args=training_args,
            max_length=None,  # set max_length = None if you don't want to truncate samples
            lora_config=lora_config,
            device=device,
            save_dir=save_dir,
            pruning_args = pruning_args,
            loss_args = loss_args,
            eval_ppl=True
        )

    if full_evaluate:
        evaluator.full_eval_run()
    else:
        evaluator.evaluate(use_lora=use_lora, use_kd=use_kd)

    # prune_full_finetune()
    # prune_lora_finetune()
    # lora_prune_interleave()
    # lora_prune_kd_interleave()
