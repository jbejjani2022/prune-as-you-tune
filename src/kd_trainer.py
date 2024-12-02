from transformers import Trainer
import torch
from torch import nn
import torch.nn.functional as F


class KDTrainer(Trainer):
    
    def __init__(self, teacher_model, alpha=0.8, temp=2, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model.eval()
        self.teacher_model.to(self.model.device)
        self.alpha = alpha
        self.temp = temp
        
        print(f'TEMP = {self.temp}, ALPHA = {self.alpha}')
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_output = model(**inputs)
        student_loss = student_output.loss
        
        # Forward pass for teacher model (no gradients required)
        with torch.inference_mode():
            teacher_output = self.teacher_model(**inputs)
            
        # Soft targets for teacher and student
        student_probs = F.log_softmax(student_output.logits / self.temp, dim=-1)
        teacher_probs = F.softmax(teacher_output.logits / self.temp, dim=-1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temp ** 2)
        loss = self.alpha * student_loss + (1 - self.alpha) * kd_loss

        return (loss, student_output) if return_outputs else loss
