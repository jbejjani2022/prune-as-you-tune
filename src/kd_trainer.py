from transformers import Trainer
import torch
from torch import nn
import torch.nn.functional as F


class KDTrainer(Trainer):
    
    def __init__(self, teacher_model, alpha=0.8, temp=2, lambda_lora=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model.eval() #TODO: still needed?
        self.teacher_model.to(self.model.device)
        self.alpha = alpha
        self.temp = temp
        self.lambda_lora = lambda_lora
        self.loss_fct = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): #TODO: is this the best way to handle num_items_in_batch parameter?
        # STANDARD LOSS
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        # cross-entropy loss
        std_loss = self.loss_fct(logits, labels)
        # L2 regularization
        lora_params = [param for name, param in model.named_parameters() if "lora" in name]
        lora_l2 = sum(torch.sum(param ** 2) for param in lora_params)
        # strength of regularization term
        std_loss += self.lambda_lora * lora_l2
        
        # KD-LOSS
        # Forward pass for teacher model (no gradients required)
        teach_outputs = self.teacher_model(**inputs)
        teach_logits = teach_outputs.logits
        # temperature
        student_probs = F.log_softmax(logits / self.temp, dim=1)
        teacher_probs = F.softmax(teach_logits / self.temp, dim=1)
        # KL divergence loss
        kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temp ** 2)
        loss = self.alpha * std_loss + (1-self.alpha) * kd_loss

        return (loss, outputs) if return_outputs else loss
