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

class CurloraLayerInner(LoraLayer):
    def __init__(self,
            base_layer,
            adapter_name,
            r: int = 64,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            lora_bias: bool = False,
            **kwargs,):

        LoraLayer.__init__(self, base_layer) #will initialize lora_A, lora_B param dicts, but they will be empty

        #self.active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias
        )

        self.base_layer = base_layer
        self._disable_adapters = False
        self.merged_adapters = []
        #self.use_dora: dict[str, bool] = {}
        #self.lora_bias: dict[str, bool] = {}
        #self.lora_magnitude_vector = nn.ModuleDict()  # if needed for advanced functionality
        #self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload = kwargs.get("ephemeral_gpu_offload", False)
        self.kwargs = kwargs

        r = kwargs.get("r", 64)
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
        print("adapter_name:", adapter_name)

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        #NOTE: we already initialized C, R, and U

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        #we shouldn't need to reset parametters, in our implementation
        pass

    #NOTE: skipping scale_layer, unscale_layer implementation. The base class implementations shouldn't affect C, U, R at all
    #NOTE: reimplimenting check_forward_args and mixed_batch_forward. I don't *think* they should affect curlora, but just in case

    def _check_forward_args(self, x, *args, **kwargs):
        pass

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        return result
    
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
    
class CurloraLayer(nn.Module, CurloraLayerInner):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        CurloraLayerInner.__init__(self, base_layer, adapter_name, r, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the CurLoRa weights into the base weights.
        Similar to the original template, but adapted for C, U, R.
        
        Args:
            safe_merge (bool): If True, performs a safe merge by checking for NaNs.
        """
        if self.merged:
            # Already merged
            return
        
        base_layer = self.get_base_layer()
        delta_weight = self.get_delta_weight()

        if safe_merge:
            # Safe merge: clone original weights and check for NaNs after merge
            orig_weights = base_layer.weight.data.clone()
            merged_weights = orig_weights + delta_weight
            if not torch.isfinite(merged_weights).all():
                raise ValueError("NaNs detected in the merged weights.")
            base_layer.weight.data = merged_weights
        else:
            # Direct merge
            base_layer.weight.data += delta_weight

        # If there's a bias term to adjust (if supported), handle it here if needed:
        # if self.lora_bias.get(self._active_adapter, False):
        #     base_layer.bias.data += some_bias_delta

        self.merged_adapters.append(self._active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge the CurLoRa weights from the base weights.
        This reverses the merge operation.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        active_adapter = self.merged_adapters.pop()
        # Since we only have one adapter in this context, active_adapter should match self._active_adapter
        if active_adapter == self._active_adapter:
            base_layer = self.get_base_layer()
            delta_weight = self.get_delta_weight()

            # Subtract the delta weights
            base_layer.weight.data -= delta_weight

            # If bias was adjusted during merge, revert it:
            # if self.lora_bias.get(active_adapter, False):
            #     base_layer.bias.data -= corresponding_bias_delta


    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight for the current adapter.

        Returns:
            torch.Tensor: The delta weight tensor of shape identical to self.base_layer.weight.
        """
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype

        # Move C, U, R to the same device and dtype as the base weights.
        C = self.C
        R = self.R
        U = self.lora_U["lora_U"]

        # Compute delta_weight = C @ U @ R
        delta_weight = C @ U @ R
        return delta_weight


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        #if not self.training:  # During evaluation
        #    print("Forward pass in eval mode")
        #    print(f"Input shape: {x.shape}")

        device = x.device
        U = self.lora_U["lora_U"]
        #W_adapted = self.C.to(device) @ U.to(device) @ self.R.to(device)
        W_adapted = self.C @ U @ self.R

        #output given by W + delta_W
        output = x @ (self.base_layer.weight + W_adapted).t()
        #output = x @ (self.base_layer.weight.to(device) + W_adapted).t()

        #if not self.training:
        #    print(f"Output shape: {output.shape}")

        if self.base_layer.bias is not None:
            output += self.base_layer.bias

        assert W_adapted.shape == self.base_layer.weight.shape, (
            f"W_adapted shape {W_adapted.shape} does not match "
            f"original layer weight shape {self.base_layer.weight.shape}"
        )

        return output
"""Forward pass in eval mode
Input shape: torch.Size([64, 512, 768])
Output shape: torch.Size([64, 512, 768])
Forward pass in eval mode
Input shape: torch.Size([64, 512, 768])
Output shape: torch.Size([64, 512, 3072])
Forward pass in eval mode
Input shape: torch.Size([64, 512, 3072])
Output shape: torch.Size([64, 512, 768])"""


### deprecated ###


class CurloraLayerDeprecated1(torch.nn.Module, BaseTunerLayer):
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
        print("adapter_name:", adapter_name)

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



class CurloraLayerDeprecated2(nn.Module, LycorisLayer):
    """
    A LyCORIS-like layer implementing a single U matrix adapter, but now extended to support multiple adapters.
    This adapter samples columns and rows from the base layer weight W to form C and R, and learns a low-rank
    correction U such that W + C @ U @ R approximates a desired adapted weight matrix.
    """

    # Required by LycorisLayer child classes
    adapter_layer_names = ("lora_U",)
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    def __init__(self, base_layer: nn.Module, adapter_name, **kwargs) -> None:
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        self.ephemeral_gpu_offload = kwargs.get("ephemeral_gpu_offload", False)
        self.kwargs = kwargs

        # Dictionaries to store adapter-specific parameters
        self.C = {}
        self.R = {}

        #breaking from function template:
        W = self.base_layer.weight.data
        C, R = self.compute_C_and_R(W, 64, "inverted_probs") #NOTE: hardcoded r = 64
        # C and R are not trainable, only U is trainable
        C.requires_grad = False
        R.requires_grad = False

        # Store parameters
        self.C[adapter_name] = C
        self.R[adapter_name] = R
        #self.lora_U[adapter_name] = U
        self.lora_U = nn.ParameterDict({
            adapter_name : nn.Parameter(torch.zeros(C.size(1), R.size(0)), requires_grad=True)
        })

        #resuming function template
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

        """W = self.base_layer.weight.data
        C, R = self.compute_C_and_R(W, r, sampling_method)
        # C and R are not trainable, only U is trainable
        C.requires_grad = False
        R.requires_grad = False

        # Initialize U with zeros
        U = nn.Parameter(torch.zeros(C.size(1), R.size(0)), requires_grad=True)

        # Store parameters
        self.C[adapter_name] = C
        self.R[adapter_name] = R
        self.lora_U[adapter_name] = U"""

        #self.lora_U[adapter_name].requires_grad = True
        self.lora_U[adapter_name] = nn.Parameter(torch.zeros(self.C[adapter_name].size(1), self.R[adapter_name].size(0)), requires_grad=True)

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
        U = self.lora_U[adapter_name]#.to(device)
        C = self.C[adapter_name]#.to(device)
        R = self.R[adapter_name]#.to(device)

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
        base_output = x @ self.base_layer.weight.t()#.to(device).t()
        if self.base_layer.bias is not None:
            base_output += self.base_layer.bias#.to(device)

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
        C = self.C[adapter_name]#.to(device)
        U = self.lora_U[adapter_name]#.to(device)
        R = self.R[adapter_name]#.to(device)

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
