from peft import LoraConfig
from peft import get_peft_model

import torch
import torch.nn as nn

class CustomLoraConfig(LoraConfig):

    def __init__(self, **kwargs):
        super().__init__()

        self.cur_rank = kwargs.get('cur_rank', self.r)  # TODO: What rank should be used??
        self.sampling_method = kwargs.get('sampling_method', 'inverted_probs') #TODO


class CurloraLayer(nn.Module):
    def __init__(self, original_layer, config: CustomLoraConfig):
        super().__init__()
        self.original_layer = original_layer
        self.config = config

        W = original_layer.weight.data
        #default sampling method is inverted probabilities (ie, prioritize rows and columns with lowest values)
        self.C, self.R = self.compute_C_and_R(W, config.cur_rank, config.sampling_method)
        #init U --> zero matrix
        self.U = nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)))
        #freeze C, R (we will only update U)
        self.C.requires_grad = False
        self.R.requires_grad = False

    #sampling_method is passed, but not used, as we'll assume sampling method = inverted probabilities
    def compute_C_and_R(self, W, rank, sampling_method): 
        #compute col probabilities
        col_norms = torch.norm(W, dim=0) ** 2
        total_norm = torch.norm(W) ** 2
        probs = col_norms / total_norm
        #invert probabilities
        inv_probs = 1 / probs
        inv_probs /= inv_probs.sum()

        #sample (based on inverted probabilities => lowest probabilities prioritized)
        col_indices = torch.multinomial(inv_probs, rank, replacement=False) #TODO: why did chatgpt suggest multinomial??
        row_indices = torch.multinomial(inv_probs, rank, replacement=False)

        C = W[:, col_indices]
        R = W[row_indices, :]
        return C, R

    def forward(self, x):
        #W_adapted = C * U * R
        W_adapted = self.C @ self.U @ self.R

        #output given by W + delta_W
        output = x @ (self.original_layer.weight + W_adapted).t()

        if self.original_layer.bias is not None: #TODO: ask teammates to verify this
            output += self.original_layer.bias

        return output
    

def get_peft_model_with_curlora(model, peft_config):
    #run customlora, if specified
    if isinstance(peft_config, CustomLoraConfig):
        #replace linear layers with CurloraLayers. TODO: does replacinga  different layer make more sense?
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                setattr(model, name, CurloraLayer(module, peft_config))
        return model
    #default to standard lora
    else:
        return get_peft_model(model, peft_config)