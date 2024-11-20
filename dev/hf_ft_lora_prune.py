# Progressiely prunes `model_name` while fine tuning on `dataset` with rsLoRA
# Saves checkpoint to `output_dir`


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
import numpy as np
import evaluate
from peft import LoraConfig, get_peft_model
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


model_name = "distilbert-base-uncased" # "google-bert/bert-base-cased"
dataset = "yelp_review_full"
output_dir = "test_lora_prune_trainer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)   # For classification tasks
    return metric.compute(predictions=predictions, references=labels)


#Loss TODO:
# 1) experiment interface (should be able to turn on/off KD loss, standard loss regularization, and vary parameters)
# 2) run a set of experiments, with all terms in the loss function = on, for 3 values of alpha x 3 values of temp. should give a baseline to inform next steps
# LoRA TODO:
# once happy with loss function design:
# 1) experiment with alternative initializations for LoRA adaptors.
# 2) build init prior into the loss function

class CustomTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.8, temp=2, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model.eval() #TODO: still needed?
        self.alpha = alpha
        self.temp = temp

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): #TODO: is this the best way to handle num_items_in_batch parameter?
        # STANDARD LOSS
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        # cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        std_loss = loss_fct(logits, labels)
        # L2 regularization
        lora_params = [param for name, param in model.named_parameters() if "lora" in name]
        lora_l2 = sum(torch.sum(param ** 2) for param in lora_params)
        # strength of regularization term
        lambda_lora = 1e-5
        std_loss += lambda_lora * lora_l2

        # KD-LOSS
        teach_outputs = self.teacher_model(**inputs)
        teach_logits = teach_outputs.logits
        # temperature
        student_probs = F.log_softmax(logits / self.temp, dim=1)
        teacher_probs = F.softmax(teach_logits / self.temp, dim=1)
        # KL divergence loss
        kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temp ** 2)

        loss = self.alpha * std_loss + (1-self.alpha) * kd_loss
        
        return (loss, outputs) if return_outputs else loss


"""
ChatGPT starter template, for replication of KD loss from PC-LoRA Paper. 
class CustomTrainerWithKD(Trainer):
    def __init__(self, teacher_model, alpha=0.5, feature_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.alpha = alpha
        self.feature_layers = feature_layers or []  # List of layer names to extract features
        self.student_features = {}
        self.teacher_features = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name, features):
            def hook(model, input, output):
                self.student_features[name] = output
            return hook

        # Register hooks for student model
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(get_activation(name, self.student_features))

        # Register hooks for teacher model
        def get_teacher_activation(name):
            def hook(model, input, output):
                self.teacher_features[name] = output
            return hook

        for name, module in self.teacher_model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(get_teacher_activation(name))

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for student
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Compute task loss
        loss_fct = nn.CrossEntropyLoss()
        L_task = loss_fct(logits, labels)

        # Forward pass for teacher
        with torch.no_grad():
            self.teacher_model(**inputs)

        # Compute feature KD loss
        L_featKD = 0.0
        for layer in self.feature_layers:
            student_feat = self.student_features.get(layer)
            teacher_feat = self.teacher_features.get(layer)
            if student_feat is not None and teacher_feat is not None:
                L_featKD += nn.MSELoss()(student_feat, teacher_feat)
        L_featKD /= len(self.feature_layers)

        # Combine losses
        L_total = self.alpha * L_task + (1 - self.alpha) * L_featKD

        return (L_total, outputs) if return_outputs else L_total"""


class PruningCallback(TrainerCallback):
    def __init__(self, model, ptg=0.1):
        self.model = model
        self.ptg = ptg  # percentage of params to prune
        self.method = prune.L1Unstructured  # pruning method
        self.params = []  # params to prune, list of tuples: (module, param_name)

        # Iterate through all modules in the model
        # and get the frozen / non-LoRA parameters to be pruned
        # skip embedding layers to mitigate accuracy degradation
        for name, module in model.named_modules():
            # Ensure module has a weight attribute and parameter
            if hasattr(module, 'weight') and 'weight' in module._parameters:
                # Filter for non-embedding and frozen / non-trainable parameters
                if not module.weight.requires_grad and not isinstance(module, nn.Embedding):
                    assert("lora" not in name)
                    self.params.append((module, 'weight'))

        # Count total number of params and number of LoRA params for sparsity reports
        self.total_params = 0
        self.lora_params = 0

        for name, param in model.named_parameters():
            if "lora" in name:
                self.lora_params += param.numel()
            self.total_params += param.numel()

    # Function called at the end of each training epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nPruning {self.ptg * 100:.2f}% of pretrained weights after epoch {state.epoch}")
        self.prune_pretrained()
        self.report_sparsity()

    # Applies pruning `weight_mask` to pretrained, non-LoRA model parameters
    # Retains previous mask, allowing cumulative pruning
    def prune_pretrained(self):
        prune.global_unstructured(
            self.params,
            pruning_method=self.method,
            amount=self.ptg,
        )
        
    # Reports sparsity of frozen / pretrained weights, as well as overall model sparsity
    def report_sparsity(self):
        zero_params = 0
        
        # Iterate over all parameters in the model
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                zero_params += torch.sum(module.weight == 0).item()
        
        sparsity_frozen = (zero_params / (self.total_params - self.lora_params)) * 100 if self.total_params > 0 else 0
        sparsity_all = (zero_params / self.total_params) * 100 if self.total_params > 0 else 0

        print(f"Pretrained Weight Sparsity: {sparsity_frozen:.2f}%")
        print(f"Overall Model Sparsity (including LoRA): {sparsity_all:.2f}%")
        
    # Removes the pruning reparameterization to make pruning permanent
    # Necessary in order to consolidate pruning changes permanently or export the model
    # Do not execute until all pruning iterations have been completed
    def remove(self):
        for (module, name) in self.params:
            prune.remove(module, name)


if __name__ == "__main__":
    dataset = load_dataset(dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    n_samples = 1000   # use 10 or less for rapid testing
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(n_samples))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(n_samples))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    frozen_model = copy.deepcopy(model)

    # A LoRA adapter will be attached to each target module
    # typically the attention and MLP layers of a transformer
    target_modules = []
    num_layers = len(model.distilbert.transformer.layer)
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
    
    # Set up PEFT with LoRA configuration
    lora_config = LoraConfig(
        r=256,               # Rank of LoRA
        lora_alpha=16,       # Scaling factor
        lora_dropout=0.1,    # Dropout rate
        target_modules=target_modules,
        use_rslora=True      # Use RSLoRA (https://huggingface.co/blog/damjan-k/rslora)
    )
    
    # Apply PEFT with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",      # Evaluate every epoch
        logging_strategy="epoch",   # Log after each epoch
        logging_dir="./logs",       # Directory for logs
        # save_strategy="epoch",    # Save checkpoint every epoch
        label_names=["labels"],
        report_to="tensorboard",    # Enable TensorBoard logging
    )
    
    # Create callback
    pruning_callback = PruningCallback(model, ptg=0.05)

    """ 
    # custom compute_loss function, standard Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        compute_loss=compute_loss,
        callbacks=[pruning_callback]
    )"""

    trainer = CustomTrainer(
        teacher_model=frozen_model,
        alpha=0.8,
        temp = 2,
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[pruning_callback]
    )

    """
    # template for chatGPT suggested KD Loss
    trainer = CustomTrainerWithKD(
        teacher_model=frozen_model,
        alpha=0.8,  # Weight for task loss
        feature_layers=feature_layers,
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[pruning_callback]
    )"""
    
    trainer.train()

    # Permanently consolidate pruning masks
    pruning_callback.remove()
