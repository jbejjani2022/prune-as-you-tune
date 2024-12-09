from peft import LoraConfig
from peft import get_peft_model

import torch
import torch.nn as nn

from peft.tuners.lora.layer import LoraLayer

class CustomLoraConfig(LoraConfig):

    def __init__(self, **kwargs):
        super().__init__()

        self.r = kwargs.get('r', self.r) #for some reason, this is necessary
        self.sampling_method = kwargs.get('sampling_method', 'inverted_probs')
        #self.device = kwargs.get('device')
        #print(f'kwargs customloraconfig: {self.device}')


# TODO: NEED TO WRITE THE MERGE, AND MERGE_AND_UNLOAD FUNCTIONS!


#NOTE: horrendous -- When loading the model, you have to register the custom modules again
class CurloraLayer(nn.Module, LoraLayer):
    #def __init__(self, original_layer, config: CustomLoraConfig, device):
    def __init__(self, original_layer, adapter_name, **kwargs):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, original_layer, **kwargs)
        #super().__init__()
        #print("created CurloraLayer!")

        self.original_layer = original_layer
        #self.config = lora_config

        # freeze original layer parameters. TODO: verify this, chatgpt recommended
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        W = original_layer.weight.data
        #default sampling method is inverted probabilities (ie, prioritize rows and columns with lowest values)
        #self.C, self.R = self.compute_C_and_R(W, lora_config.r, lora_config.sampling_method)
        #print(f"from curloralayer init: {kwargs.get('r', -1)}")
        r = kwargs.get('r', 64) #NOTE: check print statement above to ensure not hardcoded
        sm = kwargs.get('sampling_method', 'inverted_probs')
        self.C, self.R = self.compute_C_and_R(W, r, sm)

        #move C, R to device
        #self.C = self.C.to(device)
        #self.R = self.R.to(device)

        #init U --> zero matrix
        #self.U = nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)))
        self.lora_U = nn.ParameterDict({
            f"lora_U_{adapter_name}": nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)), requires_grad=True)
        })
        self.adapter_name = adapter_name

        #freeze C, R (we will only update U)
        self.C.requires_grad = False
        self.R.requires_grad = False

    #assuming sampling method = inverted probabilities 
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
        #if not self.training:  # During evaluation
        #    print("Forward pass in eval mode")
        #    print(f"Input shape: {x.shape}")
        device = x.device
        U = self.lora_U[f"lora_U_{self.adapter_name}"]
        W_adapted = self.C.to(device) @ U.to(device) @ self.R.to(device)
        #W_adapted = self.C @ U @ self.R

        #output given by W + delta_W
        output = x @ (self.original_layer.weight.to(device) + W_adapted).t() #TODO: .to(device) manually is not good

        #if not self.training:
        #    print(f"Output shape: {output.shape}")

        if self.original_layer.bias is not None: #TODO: is there something to check for, other than just None?
            output += self.original_layer.bias

        assert W_adapted.shape == self.original_layer.weight.shape, (
            f"W_adapted shape {W_adapted.shape} does not match "
            f"original layer weight shape {self.original_layer.weight.shape}"
        )

        return output
    
    def merge(self):
        print("merge called")
        # if alr merged, do nothing
        if getattr(self, "merged", False):
            return
        # merge lora weights into original layer weights
        U = self.lora_U[f"lora_U_{self.adapter_name}"]
        W_adapted = self.C @ U @ self.R
        self.original_layer.weight.data = self.original_layer.weight.data + W_adapted.data
        self.merged = True

    def merge_and_unload(self):
        print("merge and unload called")
        # first merge (if not alr done)
        self.merge()
        # remove lora params (bc they've been merged into model)
        del self.lora_U
        del self.C
        del self.R

    

### deprecated ###

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
            print(f"Found linear layer: {name}")
            #device = next(module.parameters()).device
            #print(f"replace_modules device: {device}")
            #model._modules[name] = CurloraLayer(module, peft_config, device) #TODO: uncomment to use function!
        else:
            replace_modules(module, peft_config, device)
