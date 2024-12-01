from transformers import Trainer
import torch
import torch.nn.functional as F


class KDTrainer(Trainer):
    
    def __init__(self, teacher_model, alpha=0.8, temp=2, lambda_lora=0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model.eval() #TODO: still needed?
        self.teacher_model.to(self.model.device)
        self.alpha = alpha
        self.temp = temp
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): #TODO: is this the best way to handle num_items_in_batch parameter?
        student_output = model(**inputs)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # Compute softmax for student and teacher logits with temperature
        student_probs = F.log_softmax(student_output.logits / self.temp, dim=-1)
        teacher_probs = F.softmax(teacher_outputs.logits / self.temp, dim=-1)
        
        # Compute the true label loss
        student_target_loss = student_output.loss

        # KL divergence loss
        kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temp ** 2)
        loss = self.alpha * student_target_loss + (1 - self.alpha) * kd_loss

        
        return (loss, student_output) if return_outputs else loss
