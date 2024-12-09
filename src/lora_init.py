from peft import LoraConfig
from peft import get_peft_model

import torch
import torch.nn as nn

from peft.tuners.lora.layer import LoraLayer, BaseTunerLayer
from peft.tuners.lycoris_utils import LycorisLayer

from typing import Optional

import torch
import torch.nn as nn
import warnings
from typing import Optional, Any, Union
from torch import nn
from abc import abstractmethod
from collections import defaultdict

class CustomLoraConfig(LoraConfig):

    def __init__(self, **kwargs):
        super().__init__()

        self.r = kwargs.get('r', self.r) #for some reason, this is necessary
        self.sampling_method = kwargs.get('sampling_method', 'inverted_probs')
        #self.device = kwargs.get('device')
        #print(f'kwargs customloraconfig: {self.device}')


#TODO: 1) set scaling, set dropout


class CurloraLayer(LycorisLayer):
    """
    A LyCORIS-like layer implementing a single U matrix adapter, but now extended to support multiple adapters.
    This adapter samples columns and rows from the base layer weight W to form C and R, and learns a low-rank
    correction U such that W + C @ U @ R approximates a desired adapted weight matrix.
    """

    # Required by LycorisLayer child classes
    adapter_layer_names = ("lora_U",)
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer)

        self.ephemeral_gpu_offload = kwargs.get("ephemeral_gpu_offload", False)
        self.kwargs = kwargs

        # Dictionaries to store adapter-specific parameters
        self.lora_U = nn.ParameterDict()
        self.C = {}
        self.R = {}

        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.module_dropout = {}

        self._active_adapter = []
        self._disable_adapters = False
        self.merged_adapters = []

    @property
    def _available_adapters(self) -> set[str]:
        return set(self.lora_U.keys())

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    def set_adapter(self, adapter_names: Union[str, list[str]]):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        # if model is merged, unmerge first
        # handled by tuner before calling set_adapter if needed
        self._active_adapter = adapter_names

    def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs):
        """
        Create C, R, U for the given adapter. This is where we set up the low-rank decomposition
        for the chosen adapter.
        """
        # freeze the base weights
        self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        sampling_method = kwargs.get('sampling_method', 'inverted_probs')
        alpha = kwargs.get('alpha', r)  # default alpha=r if not provided

        W = self.base_layer.weight.data
        C, R = self.compute_C_and_R(W, r, sampling_method)
        # C and R are not trainable, only U is trainable
        C.requires_grad = False
        R.requires_grad = False

        # Initialize U with zeros
        U = nn.Parameter(torch.zeros(C.size(1), R.size(0)), requires_grad=True)

        # Store parameters
        self.C[adapter_name] = C
        self.R[adapter_name] = R
        self.lora_U[adapter_name] = U
        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r

        # If rank_dropout or module_dropout are relevant, initialize them here
        self.rank_dropout[adapter_name] = 0.0  # or appropriate default
        self.module_dropout[adapter_name] = 0.0  # or appropriate default

    def _get_delta_activations(self, adapter_name: str, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the delta activations contributed by the given adapter.
        Delta = x @ (C U R)^T scaled by self.scaling[adapter_name].
        """
        if self._disable_adapters or adapter_name not in self._active_adapter:
            # No delta if adapter is disabled or not active
            return torch.zeros_like(x)

        device = x.device
        U = self.lora_U[adapter_name].to(device)
        C = self.C[adapter_name].to(device)
        R = self.R[adapter_name].to(device)

        W_adapted = C @ U @ R  # shape matches base_layer.weight
        W_adapted = W_adapted * self.scaling[adapter_name]

        # delta activations after the base layer forward pass:
        # base layer is usually: output = x @ W^T + bias
        # delta_activations: x @ W_adapted^T
        return x @ W_adapted.t()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass:
        1. Compute base output: x @ W^T (+ bias)
        2. Add delta activations from all active adapters: sum over adapters in self._active_adapter
        """
        device = x.device
        base_output = x @ self.base_layer.weight.to(device).t()
        if self.base_layer.bias is not None:
            base_output += self.base_layer.bias.to(device)

        if not self._disable_adapters and self._active_adapter:
            for adapter in self._active_adapter:
                if adapter in self._available_adapters:
                    base_output += self._get_delta_activations(adapter, x, *args, **kwargs)

        return base_output

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Return the delta weights for the given adapter: C U R * scaling.
        This is used when merging the adapter weights into the base layer.
        """
        if adapter_name not in self._available_adapters:
            # If not available, return zero delta
            return torch.zeros_like(self.base_layer.weight.data)

        device = self.base_layer.weight.device
        C = self.C[adapter_name].to(device)
        U = self.lora_U[adapter_name].to(device)
        R = self.R[adapter_name].to(device)

        W_adapted = C @ U @ R
        W_adapted = W_adapted * self.scaling[adapter_name]
        return W_adapted

    def reset_adapter_parameters(self, adapter_name: str):
        """
        Reset the adapter parameters for the given adapter.
        Typically sets U to zero.
        """
        if adapter_name in self._available_adapters:
            with torch.no_grad():
                self.lora_U[adapter_name].zero_()

    def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs):
        """
        Update the layer configuration for the given adapter.
        Can be used to change the rank r or alpha after initialization.
        """
        if adapter_name not in self._available_adapters:
            # If adapter doesn't exist yet, create it
            self.create_adapter_parameters(adapter_name, r, alpha=alpha, **kwargs)
        else:
            # If adapter already exists, update alpha, scaling, etc.
            self.r[adapter_name] = r
            self.alpha[adapter_name] = alpha
            self.scaling[adapter_name] = alpha / r

    def compute_C_and_R(self, W, rank, sampling_method):
        # Same logic as in your original code
        num_rows, num_cols = W.size()
        rank = min(rank, num_rows, num_cols)
        total_norm = torch.norm(W) ** 2

        # Column norms and inverted probabilities
        col_norms = torch.norm(W, dim=0) ** 2
        col_probs = col_norms / total_norm
        inv_col_probs = 1 / col_probs
        inv_col_probs /= inv_col_probs.sum()

        # Row norms and inverted probabilities
        row_norms = torch.norm(W, dim=1) ** 2
        row_probs = row_norms / total_norm
        inv_row_probs = 1 / row_probs
        inv_row_probs /= inv_row_probs.sum()

        # Sample based on inverted probabilities
        col_indices = torch.multinomial(inv_col_probs, rank, replacement=False)
        row_indices = torch.multinomial(inv_row_probs, rank, replacement=False)

        C = W[:, col_indices].clone()
        R = W[row_indices, :].clone()

        return C, R

    def delete_adapter(self, adapter_name: str):
        """
        Delete an adapter and all its parameters.
        """
        if adapter_name in self._available_adapters:
            del self.lora_U[adapter_name]
            del self.C[adapter_name]
            del self.R[adapter_name]
            del self.r[adapter_name]
            del self.alpha[adapter_name]
            del self.scaling[adapter_name]
            del self.rank_dropout[adapter_name]
            del self.module_dropout[adapter_name]

            if adapter_name in self._active_adapter:
                self._active_adapter.remove(adapter_name)
            if adapter_name in self.merged_adapters:
                self.merged_adapters.remove(adapter_name)



### deprecated ###


class CurloraLayerOld(torch.nn.Module, BaseTunerLayer):
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

    def merge_and_unload(self):
        print("merge and unload called")
        # first merge (if not alr done)
        self.merge()
        # remove lora params (bc they've been merged into model)
        del self.lora_U
        del self.C
        del self.R

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
            #model._modules[name] = CurloraLayer(module, "curlora") #, peft_config, device) #TODO: uncomment to use function!
        else:
            replace_modules(module, peft_config, device)
