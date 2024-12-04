from peft import LoraConfig
from peft import get_peft_model

import torch
import torch.nn as nn

class CustomLoraConfig(LoraConfig):

    def __init__(self, **kwargs):
        super().__init__()

        self.cur_rank = kwargs.get('r', self.r) #we're usually passing r=32
        self.sampling_method = kwargs.get('sampling_method', 'inverted_probs')
        #self.device = kwargs.get('device')
        #print(f'kwargs customloraconfig: {self.device}')


class CurloraLayer(nn.Module):
    def __init__(self, original_layer, config: CustomLoraConfig, device):
        super().__init__()
        self.original_layer = original_layer
        self.config = config

        # freeze original layer parameters. TODO: verify this, chatgpt recommended
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        W = original_layer.weight.data
        #default sampling method is inverted probabilities (ie, prioritize rows and columns with lowest values)
        self.C, self.R = self.compute_C_and_R(W, config.cur_rank, config.sampling_method)
        #move C, R to device
        self.C = self.C.to(device)
        self.R = self.R.to(device)

        #init U --> zero matrix
        self.U = nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)))

        #freeze C, R (we will only update U)
        self.C.requires_grad = False
        self.R.requires_grad = False

    #NOTE: assume sampling method = inverted probabilities 
    def compute_C_and_R(self, W, rank, sampling_method): 
        #device = W.device
        num_rows, num_cols = W.size()
        #adjust rank if needed
        rank = min(rank, num_rows, num_cols)
        #print(f"desired rank: {rank}, num_rows: {num_rows}, num_cols: {num_cols}")

        total_norm = torch.norm(W) ** 2  #scalar

        #col norm and probs
        col_norms = torch.norm(W, dim=0) ** 2  #shape: [n]
        col_probs = col_norms / total_norm  #shape: [n]
        #invert col probs
        inv_col_probs = 1 / col_probs
        inv_col_probs /= inv_col_probs.sum() #normalize

        #row norms and probs (chatgpt corrected)
        row_norms = torch.norm(W, dim=1) ** 2  #shape: [m]
        row_probs = row_norms / total_norm  #shape: [m]
        #inver row probs
        inv_row_probs = 1 / row_probs
        inv_row_probs /= inv_row_probs.sum() #normalize

        #sample (based on inverted probabilities => lowest probabilities prioritized)
        col_indices = torch.multinomial(inv_col_probs, rank, replacement=False)
        row_indices = torch.multinomial(inv_row_probs, rank, replacement=False)

        C = W[:, col_indices]
        R = W[row_indices, :]

        
        ### verify that inverted_probs sampling is working correctly!
        """inv_col_probs_list = inv_col_probs.cpu().tolist()
        all_col_indices = list(range(len(inv_col_probs_list)))
        col_inv_probs_pairs = list(zip(all_col_indices, inv_col_probs_list))
        sorted_col_inv_probs_pairs = sorted(
            col_inv_probs_pairs, key=lambda x: x[1], reverse=True
        )

        print("bottom 5! top 5!")
        indices_to_print = sorted_col_inv_probs_pairs[:5] + sorted_col_inv_probs_pairs[-5:]
        for idx, inv_prob in indices_to_print:
            print(f'Index: {idx}, Inverted Probability: {inv_prob}')

        for i in range(0, min(5, len(col_indices))):
            print(f'selected probs: {inv_col_probs[col_indices[i]]}')"""
    
        return C, R

    def forward(self, x):
        #W_adapted = C * U * R
        #print(f"x device: {x.device}")
        #print(f"C device: {self.C.device}")
        #print(f"U device: {self.U.device}")
        #print(f"R device: {self.R.device}")
        device = x.device
        W_adapted = self.C @ self.U @ self.R

        #output given by W + delta_W
        output = x @ (self.original_layer.weight.to(device) + W_adapted).t() #TODO: .to(device) manually is not good

        if self.original_layer.bias is not None: #TODO: ask teammates to verify this
            output += self.original_layer.bias

        return output
    

"""def get_peft_model_with_curlora(model, peft_config):
    #run customlora, if specified
    if isinstance(peft_config, CustomLoraConfig):
        #replace linear layers with CurloraLayers. 
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                setattr(model, name, CurloraLayer(module, peft_config))
        return model
    #default to standard lora
    else:
        return get_peft_model(model, peft_config)"""

#TODO: the following works, but is not actually super useful. Need to think more about this before running tests. TODO: does replacing a different layer make more sense?
#chatgpt corrected, to fix dictionary concurrent changes error
def get_peft_model_with_curlora(model, peft_config, device):
    if isinstance(peft_config, CustomLoraConfig):
        replace_modules(model, peft_config, device)
        return model
    else:
        print("Expected CustomLoraConfig, Received Standard")
        return get_peft_model(model, peft_config)

def replace_modules(model, peft_config, device):
    for name, module in list(model._modules.items()):
        if module is None:
            continue
        if isinstance(module, nn.Linear):
            #print("replaced linear layer with lora layer")
            #device = next(module.parameters()).device
            #print(f"replace_modules device: {device}")
            model._modules[name] = CurloraLayer(module, peft_config, device)
        else:
            replace_modules(module, peft_config, device)


#TODO:
# 1) verify that layer replacement is happening correctly
# 2) decide if replacing all linear layers is the correct thing to do (should we be replacing even more layers? or fewer? or...?)