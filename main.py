from transformers import TrainingArguments
from peft import LoraConfig, TaskType
import torch
import typer
from typing import Optional, Annotated
from src.finetune_evaluators import DistilBertFineTuneEvaluator, BertBaseFineTuneEvaluator
import os

app = typer.Typer()

@app.command()
def run_and_eval (n_samples : Annotated[Optional[int], typer.Option(help="Number of samples, use 10 or less for rapid testing")] = 1000,
          ptg : Annotated[Optional[float], typer.Option(help="Percentage of parameters to prune per pruning call")] = 0.05,
          sparsity_target: Annotated[Optional[float], typer.Option(help="Target percentage of parameters to prune")] = 0.5,
          num_epochs : Annotated[Optional[int], typer.Option(help="Number of training epochs")] = 3,
          output_dir : Annotated[Optional[str], typer.Option(help="Output directory for logs and model checkpoints")] = "logs",
          dataset : Annotated[Optional[str], typer.Option(help="Dataset to use for fine-tuning")] = "imdb",
          full_evaluate : Annotated[Optional[bool], typer.Option(help="Evaluate using all variations of pruning methods, as well as full fine-tuning")] = False,
          use_lora : Annotated[Optional[bool], typer.Option(help="Use LoRA adapters for fine-tuning")] = True,
          use_kd : Annotated[Optional[bool], typer.Option(help="Use knowledge distillation for fine-tuning")] = True,
          kd_alpha : Annotated[Optional[float], typer.Option(help="Alpha parameter for knowledge distillation")] = 0.5,
          kd_temp : Annotated[Optional[float], typer.Option(help="Temperature parameter for knowledge distillation")] = 2,
          kd_lambda_lora : Annotated[Optional[float], typer.Option(help="Lambda parameter for LoRA regularization in knowledge distillation")] = 1e-5,
          use_rs_lora : Annotated[Optional[bool], typer.Option(help="Use rsLoRA adapters for fine-tuning")] = False,
          lora_dropout : Annotated[Optional[float], typer.Option(help="Dropout rate for LoRA adapters")] = 0.1,
          lora_alpha : Annotated[Optional[float], typer.Option(help="Scaling factor for LoRA adapters")] = 32,
          lora_rank : Annotated[Optional[int], typer.Option(help="Rank of LoRA adapters")] = 32):
    
    pruning_method = "L1Unstructured"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # how sparse the pruned models should be after all training epochs
    # for interleaving methods:
    # pruning percentage per epoch = sparsity_target / num_train_epochs

    save_dir = 'bert-imdb-r32-nomaxlen/50pct-sparsity-5epochs/prune_sched/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",      # Evaluate every epoch
        logging_strategy="epoch",   # Log after each epoch
        save_strategy="no",
        fp16=False,                  # Mixed precision training
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        dataloader_num_workers=4
    )

    print(f'dataset: {dataset}')
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
                 "schedule" : "linear",
                 "prune_every_epoch" : 1}

    loss_args = {"alpha": kd_alpha, 
                 "temp": kd_temp, 
                 "lambda_lora": kd_lambda_lora,
                 "use_kd_loss": use_kd}

    evaluator = BertBaseFineTuneEvaluator(
        n_samples=n_samples,
        dataset=dataset,
        training_args=training_args,
        max_length=None,  # set max_length = None if you don't want to truncate samples
        lora_config=lora_config,
        device=device,
        save_dir=save_dir,
        pruning_args = pruning_args,
        loss = loss_args,
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