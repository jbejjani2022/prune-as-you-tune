from peft import LoraConfig
from peft import get_peft_model

import torch
import torch.nn as nn

from peft.tuners.lora.layer import LoraLayer, BaseTunerLayer

from typing import Optional

class CustomLoraConfig(LoraConfig):

    def __init__(self, **kwargs):
        super().__init__()

        self.r = kwargs.get('r', self.r) #for some reason, this is necessary
        self.sampling_method = kwargs.get('sampling_method', 'inverted_probs')
        #self.device = kwargs.get('device')
        #print(f'kwargs customloraconfig: {self.device}')


class CurloraLayer(torch.nn.Module, BaseTunerLayer):
    """
    A custom LoRA-like layer that has only one trainable parameter U.
    This is a template class that you can fill in with your desired forward pass logic.
    """

    def __init__(self, base_layer: nn.Module, adapter_name, **kwargs) -> None:
        super().__init__()        
        BaseTunerLayer.__init__(self)#, base_layer)

        self.base_layer = base_layer
        self._disable_adapters = False
        self.merged_adapters = []
        #self.use_dora: dict[str, bool] = {}
        #self.lora_bias: dict[str, bool] = {}
        #self.lora_magnitude_vector = nn.ModuleDict()  # if needed for advanced functionality
        #self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload = kwargs.get("ephemeral_gpu_offload", False)
        self.kwargs = kwargs

        r = kwargs.get("r", 32)
        sm = kwargs.get('sampling_method', 'inverted_probs')

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        W = base_layer.weight.data
        #default sampling method is inverted probabilities (ie, prioritize rows and columns with lowest values)
        #self.C, self.R = self.compute_C_and_R(W, lora_config.r, lora_config.sampling_method)
        #print(f"from curloralayer init: {kwargs.get('r', -1)}")
        
        self.C, self.R = self.compute_C_and_R(W, r, sm)
        #freeze C, R (we will only update U)
        self.C.requires_grad = False
        self.R.requires_grad = False

        #self.U = nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)), requires_grad=True)
        self.lora_U = nn.ParameterDict({
            "lora_U": nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)), requires_grad=True)
        })

        self.adapter_name = adapter_name #"curlora"

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    def set_adapter(self, adapter_names):
        self._active_adapter = [adapter_names] #['curlora']

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        if getattr(self, "merged", False):
            return
        # merge lora weights into original layer weights
        U = self.lora_U["lora_U"]
        W_adapted = self.C @ U @ self.R
        self.base_layer.weight.data = self.base_layer.weight.data + W_adapted.data
        self.merged = True

    #def unmerge(self) -> None:
        # Implement the logic to undo merging if applicable.
    #    pass

    #def scale_layer(self, scale: float) -> None:
    #    pass

    #def unscale_layer(self, scale=None) -> None:
    #    pass

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        #if not self.training:  # During evaluation
        #    print("Forward pass in eval mode")
        #    print(f"Input shape: {x.shape}")
        device = x.device
        U = self.lora_U["lora_U"]
        W_adapted = self.C.to(device) @ U.to(device) @ self.R.to(device)
        #W_adapted = self.C @ U @ self.R

        #output given by W + delta_W
        output = x @ (self.base_layer.weight.to(device) + W_adapted).t() #TODO: .to(device) manually is not good

        #if not self.training:
        #    print(f"Output shape: {output.shape}")

        if self.base_layer.bias is not None: #TODO: is there something to check for, other than just None?
            output += self.base_layer.bias

        assert W_adapted.shape == self.base_layer.weight.shape, (
            f"W_adapted shape {W_adapted.shape} does not match "
            f"original layer weight shape {self.base_layer.weight.shape}"
        )

        return output
    
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

        #C = W[:, col_indices].clone()
        #R = W[row_indices, :].clone()
    
        return C, R


### deprecated ###

#NOTE: horrendous -- When loading the model, you have to register the custom modules again
class CurloraLayerOld(nn.Module, LoraLayer):
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
        r = kwargs.get('r', 32) #NOTE: check print statement above to ensure not hardcoded
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

        #C = W[:, col_indices]
        #R = W[row_indices, :]

        C = W[:, col_indices].clone()
        R = W[row_indices, :].clone()
    
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
            model._modules[name] = CurloraLayer(module, "curlora") #, peft_config, device) #TODO: uncomment to use function!
        else:
            replace_modules(module, peft_config, device)
