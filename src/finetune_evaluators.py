"""
Subclasses of FineTuneEvaluator for evaluating finetuning methods on particular model types, e.g. BERT
"""


import evaluate
from transformers import AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
import numpy as np
import copy
from peft import get_peft_model

from src.evaluator import FineTuneEvaluator
from src.kd_trainer import KDTrainer
from src.pruner import PruningCallback


class BERTFineTuneEvaluator(FineTuneEvaluator):
    
    def __init__(self, model, dataset, training_args, lora_config, pruning_method):
        super().__init__(model, dataset, training_args, lora_config, pruning_method)
        self.num_labels = len(set(dataset['train']['labels']))
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=self.num_labels)
        
        print(f'{self.train_dataset.num_rows} training samples')
        print(f'{self.eval_dataset.num_rows} evaluation samples')
        
        self.lora_config.target_modules = self.get_target_modules()
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.metric = evaluate.load("accuracy")
        
    def get_target_modules(self):
        # A LoRA adapter will be attached to each target module
        # typically the attention and MLP layers of a transformer
        target_modules = []
        num_layers = len(self.model.distilbert.transformer.layer)
        target = "distilbert.transformer.layer"
        # Iterate over the transformer layers to get target modules
        for i in range(num_layers):
            target_modules.extend([
                f"{target}.{i}.attention.q_lin",    # Query projection in attention
                f"{target}.{i}.attention.k_lin",    # Key projection in attention
                f"{target}.{i}.attention.v_lin",    # Value projection in attention
                f"{target}.{i}.attention.out_lin",  # Output projection in attention
                f"{target}.{i}.ffn.lin1",           # First linear layer in FFN
                f"{target}.{i}.ffn.lin2"            # Second linear layer in FFN
            ])
        
        return target_modules
    
    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)   # For classification tasks
        return self.metric.compute(predictions=predictions, references=labels)
    
    def get_trainer(self, model):
        return Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator,
        )
        
    def evaluate(self):
        self.full_finetune()
        self.lora_finetune(rslora = True)
        self.lora_prune_finetune(rslora = True)
        self.lora_prune_kd_finetune(rslora = True)
        
    def full_finetune(self):
        model = copy.deepcopy(self.model)
        trainer = self.get_trainer(model)
        trainer.train()
        
    def lora_finetune(self, rslora):
        model = copy.deepcopy(self.model)
        self.lora_config.rslora = rslora
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        trainer = self.get_trainer(model)
        trainer.train()
        
    def lora_prune_finetune(self, rslora):
        model = copy.deepcopy(self.model)
        self.lora_config.rslora = rslora
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        trainer = self.get_trainer(model)
        pruning_callback = PruningCallback(model, method=self.pruning_method, ptg=0.1)
        trainer.callbacks = [pruning_callback]
        trainer.train()
        pruning_callback.remove()
        
    def lora_prune_kd_finetune(self, rslora):
        model = copy.deepcopy(self.model)
        frozen_model = copy.deepcopy(model)
        self.lora_config.rslora = rslora
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        trainer = self.get_kd_trainer(model, frozen_model)
        pruning_callback = PruningCallback(model, method=self.pruning_method, ptg=0.1)
        trainer.callbacks = [pruning_callback]
        trainer.train()
        pruning_callback.remove()
        
    def get_kd_trainer(self, student, teacher):
        return KDTrainer(
            teacher_model=teacher,
            alpha=0.8,
            temp=2,
            lambda_lora=0,
            model=student,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
    