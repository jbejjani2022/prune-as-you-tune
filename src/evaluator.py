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
from src.perplexity import PPL


class FineTuneEvaluator(ABC):
    
    def __init__(self,
                 model_name : str,
                 dataset,
                 training_args : TrainingArguments,
                 max_length : int,  # sets max token length per training sample - useful for reducing train time
                 lora_config : LoraConfig,
                 pruning_method : str,
                 sparsity_target : float,
                 alpha : float,
                 temp : int,
                 device,
                 save_dir : str,
                 eval_ppl : bool = True):
        
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
        self.num_epochs = self.training_args.num_train_epochs
        self.lora_config = lora_config
        self.lora_config.target_modules = self.get_target_modules()
        self.pruning_method = pruning_method
        self.sparsity_target = sparsity_target
        self.alpha = alpha
        self.temp = temp
        
        self.save_dir = save_dir
        
        self.eval_ppl = eval_ppl  # whether to evaluate perplexity on orig task after each finetuning
        if self.eval_ppl:
            self.ppl = PPL(model_name, device)
        
        print(self.model.device)
        
    # Returns a list of modules in the model to be finetuned with LoRA adapters
    @abstractmethod
    def get_target_modules(self):
        pass 
        
    # Tokenizes `examples` using `self.tokenizer
    def tokenize(self, examples):
        if self.max_length is None:
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        else:
            print(f'max_length = {self.max_length}')
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
    def get_kd_trainer(self, student, teacher, temp, logger_callback=None, pruning_callback=None):
        callbacks = []
        if logger_callback:
            callbacks.append(logger_callback)
        if pruning_callback:
            callbacks.append(pruning_callback)
        
        print(f'KD TEMPERATURE = {temp}')
        
        return KDTrainer(
            teacher_model=teacher,
            alpha=self.alpha,
            temp=temp,
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
        return LoggerCallback(f'{self.save_dir}/{log_file}', f'{self.save_dir}/{checkpoint_dir}')

    # Returns a callback that prunes the pretrained weights of `model` by `ptg`% after each epoch
    def get_pruner(self, model, lora):
        return PruningCallback(model, method=self.pruning_method, lora=lora, sparsity_target=self.sparsity_target, num_epochs=self.num_epochs)
    
    # Evaluates four fine-tuning methods on the model
    def evaluate(self):
        self.full_finetune()
        self.lora_finetune()
        self.prune_full_finetune()
        self.prune_lora_finetune()
        self.lora_prune_interleave()
        self.lora_prune_kd_interleave()
        # self.lora_prune_kd_interleave_not_rs()
    
    # Fine-tunes all model weights
    def full_finetune(self):
        print('\n********* FULL FINETUNING *********\n')
        model = copy.deepcopy(self.model)
        logger = self.get_logger('full_finetune.csv', 'checkpoints/full_finetune')
        trainer = self.get_trainer(model, logger_callback=logger)
        trainer.train()
        
        if self.eval_ppl:
            self.report_perplexity(model)
    
    # Fine-tunes only LoRA adapters
    def lora_finetune(self):
        print('\n********* LORA FINETUNING *********\n')
        model = copy.deepcopy(self.model)
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        logger = self.get_logger('lora_finetune.csv', 'checkpoints/lora_finetune')
        trainer = self.get_trainer(model, logger_callback=logger)
        trainer.train()
        
        if self.eval_ppl:
            self.report_perplexity(model)
    
    # Interleaves LoRA fine-tuning with pruning of pretrained weights
    def lora_prune_interleave(self):
        print('\n********* LORA PRUNE FINETUNING (INTERLEAVED) *********\n')
        model = copy.deepcopy(self.model)
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        pruner = self.get_pruner(model, lora=True)
        # pruning step before fine-tuning begins
        pruner.prune_pretrained(epoch=0)
        
        logger = self.get_logger('lora_prune_interleave.csv', 'checkpoints/lora_prune_interleave')
        trainer = self.get_trainer(model, pruning_callback=pruner, logger_callback=logger)
        trainer.train()
        pruner.remove()
        
        if self.eval_ppl:
            self.report_perplexity(model)
    
    # Same as lora_prune_finetune but fine-tunes using KD loss via frozen teacher model
    def lora_prune_kd_interleave(self, temp=None):
        if temp is None:
            temp = self.temp
            
        print('\n********* LORA PRUNE KD FINETUNING (INTERLEAVED) *********\n')
        model = copy.deepcopy(self.model)
        frozen_model = copy.deepcopy(model)
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        pruner = self.get_pruner(model, lora=True)
        pruner.prune_pretrained(epoch=0)
        
        logger = self.get_logger(f'lora_prune_kd_interleave_temp{temp}.csv', f'checkpoints/lora_prune_kd_interleave_temp{temp}')
        trainer = self.get_kd_trainer(model, frozen_model, temp, pruning_callback=pruner, logger_callback=logger)
        trainer.train()
        pruner.remove()
        
        if self.eval_ppl:
            self.report_perplexity(model)

    def lora_prune_kd_interleave_not_rs(self):
        print('\n********* (NOT RS) LORA PRUNE KD FINETUNING (INTERLEAVED) *********\n')
        model = copy.deepcopy(self.model)
        frozen_model = copy.deepcopy(model)
        self.lora_config.use_rslora = False
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        pruner = self.get_pruner(model, lora=True)
        pruner.prune_pretrained(epoch=0)
        
        logger = self.get_logger('lora_prune_kd_interleave_not_rs.csv', 'checkpoints/lora_prune_kd_interleave_not_rs')
        trainer = self.get_kd_trainer(model, frozen_model, pruning_callback=pruner, logger_callback=logger)
        trainer.train()
        pruner.remove()
        
        if self.eval_ppl:
            self.report_perplexity(model)
    
    # Prunes model once then fine-tunes all remaining weights
    def prune_full_finetune(self):
        print('\n********* PRUNE THEN FULL FINETUNE *********\n')
        model = copy.deepcopy(self.model)
        
        pruner = self.get_pruner(model, lora=False)
        pruner.report_sparsity()
        print(f"\nPruning {self.sparsity_target * 100:.2f}% of pretrained weights before finetuning")
        pruner.prune_pretrained(epoch=0, epoch_ptg=self.sparsity_target)
        pruner.report_sparsity()
        
        logger = self.get_logger('prune_full_finetune.csv', 'checkpoints/prune_full_finetune')
        trainer = self.get_trainer(model, logger_callback=logger)
        trainer.train()
        pruner.remove()
        
        if self.eval_ppl:
            self.report_perplexity(model)
    
    # Prunes model once then fine-tunes LoRA adapters
    def prune_lora_finetune(self):
        print('\n********* PRUNE THEN LoRA FINETUNE *********\n')
        model = copy.deepcopy(self.model)
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        
        pruner = self.get_pruner(model, lora=True)
        pruner.report_sparsity()
        print(f"\nPruning {self.sparsity_target * 100:.2f}% of pretrained weights before finetuning")
        pruner.prune_pretrained(epoch=0, epoch_ptg=self.sparsity_target)
        pruner.report_sparsity()
        
        logger = self.get_logger('prune_lora_finetune.csv', 'checkpoints/prune_lora_finetune')
        trainer = self.get_trainer(model, logger_callback=logger)
        trainer.train()
        pruner.remove()
        
        if self.eval_ppl:
            self.report_perplexity(model)
            
    def prune_no_finetune(self):
        print('\n********* PRUNE, THEN GET PERPLEXITY *********\n')
        model = copy.deepcopy(self.model)
        
        pruner = self.get_pruner(model, lora=False)
        pruner.report_sparsity()
        print(f"\nPruning {self.sparsity_target * 100:.2f}% of pretrained weights")
        pruner.prune_pretrained(epoch=0, epoch_ptg=self.sparsity_target)
        pruner.report_sparsity()
        pruner.remove()
        
        if self.eval_ppl:
            self.report_perplexity(model)

    def report_perplexity(self, model):
        perplexity = self.ppl.eval(model=model)
        print(f'perplexity = {perplexity}')
