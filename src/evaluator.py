"""
Abstract FineTuneEvaluator class
For evaluating finetuning methods on particular models
"""


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from abc import ABC, abstractmethod
import evaluate
import copy
import numpy as np

from src.kd_trainer import KDTrainer
from src.logger import LoggerCallback
from src.pruner import PruningCallback


class FineTuneEvaluator(ABC):
    
    def __init__(self, model_name : str,
                 dataset, training_args : TrainingArguments,
                 max_length : int, # sets max token length per training sample - useful for reducing train time
                 lora_config : LoraConfig,
                 pruning_method : str,
                 alpha : float,
                 temp : int,
                 device):
        
        # load tokenizer
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # load and tokenize the dataset
        self.dataset = load_dataset(dataset).map(self.tokenize, batched=True)
        self.train_dataset = self.dataset["train"].shuffle(seed=42)
        self.eval_dataset = self.dataset["test"].shuffle(seed=42)
        print(f'{self.train_dataset.num_rows} training samples')
        print(f'{self.eval_dataset.num_rows} evaluation samples')
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.metric = evaluate.load("accuracy")
        # load model
        self.num_labels = len(set(self.dataset['train']['label']))
        print(f'num_labels = {self.num_labels}')
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model = self.model.to(device)
        # hyperparameters for lora finetuning, pruning, and KD
        self.training_args = training_args
        self.lora_config = lora_config
        self.lora_config.target_modules = self.get_target_modules()
        self.pruning_method = pruning_method
        self.alpha = alpha
        self.temp = temp
        
        print(self.model.device)
        
    # Returns a list of modules in the model to be finetuned with LoRA adapters
    @abstractmethod
    def get_target_modules(self):
        pass 
        
    # Tokenizes `examples` using `self.tokenizer
    def tokenize(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
    
    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)   # For classification tasks
        return self.metric.compute(predictions=predictions, references=labels)
    
    # Returns a HF Trainer using `training_args` and callbacks, if any
    def get_trainer(self, model, logger_callback=None, pruning_callback=None):
        callbacks = []
        if logger_callback:
            callbacks.append(logger_callback)
        if pruning_callback:
            callbacks.append(pruning_callback)
            
        return Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator,
            callbacks=callbacks
        )

    # Returns a custom Knowledge-Distillation Trainer
    def get_kd_trainer(self, student, teacher, logger_callback=None, pruning_callback=None):
        callbacks = []
        if logger_callback:
            callbacks.append(logger_callback)
        if pruning_callback:
            callbacks.append(pruning_callback)
        
        return KDTrainer(
            teacher_model=teacher,
            alpha=self.alpha,
            temp=self.temp,
            lambda_lora=0,
            model=student,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=callbacks,
            compute_metrics=self.compute_metrics
        )
    
    # Returns a callback that logs eval acc. and loss to `log_file` and saves model to `checkpoint_dir` after each epoch
    def get_logger(self, log_file, checkpoint_dir):
        return LoggerCallback(log_file, checkpoint_dir)

    # Returns a callback that prunes the pretrained weights of `model` by `ptg`% after each epoch
    def get_pruner(self, model, ptg):
        return PruningCallback(model, method=self.pruning_method, ptg=ptg)
    
    # Evaluates four fine-tuning methods on the model
    def evaluate(self):
        self.full_finetune()
        self.lora_finetune(rslora = True)
        self.lora_prune_finetune(rslora = True)
        self.lora_prune_kd_finetune(rslora = True)
    
    # Fine-tunes all model weights
    def full_finetune(self):
        print('\n********* FULL FINETUNING *********\n')
        model = copy.deepcopy(self.model)
        logger = self.get_logger('full_finetune.csv', 'checkpoints/full_finetune')
        trainer = self.get_trainer(model, logger_callback=logger)
        trainer.train()
    
    # Fine-tunes only LoRA adapters
    def lora_finetune(self, rslora):
        print('\n********* LORA FINETUNING *********\n')
        model = copy.deepcopy(self.model)
        self.lora_config.rslora = rslora
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        logger = self.get_logger('lora_finetune.csv', 'checkpoints/lora_finetune')
        trainer = self.get_trainer(model, logger_callback=logger)
        trainer.train()
    
    # Interleaves LoRA fine-tuning with pruning of pretrained weights
    def lora_prune_finetune(self, rslora):
        print('\n********* LORA PRUNE FINETUNING *********\n')
        model = copy.deepcopy(self.model)
        self.lora_config.rslora = rslora
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        pruner = self.get_pruner(model, ptg=0.05)
        logger = self.get_logger('lora_prune_finetune.csv', 'checkpoints/lora_prune_finetune')
        trainer = self.get_trainer(model, pruning_callback=pruner, logger_callback=logger)
        trainer.train()
        pruner.remove()
    
    # Same as lora_prune_finetune but fine-tunes using KD loss via frozen teacher model
    def lora_prune_kd_finetune(self, rslora):
        print('\n********* LORA PRUNE KD FINETUNING *********\n')
        model = copy.deepcopy(self.model)
        frozen_model = copy.deepcopy(model)
        self.lora_config.rslora = rslora
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        pruner = self.get_pruner(model, ptg=0.05)
        logger = self.get_logger('lora_prune_kd_finetune.csv', 'checkpoints/lore_prune_kd_finetune')
        trainer = self.get_kd_trainer(model, frozen_model, pruning_callback=pruner, logger_callback=logger)
        trainer.train()
        pruner.remove()
