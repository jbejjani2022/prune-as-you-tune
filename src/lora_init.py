import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
import warnings
from typing import Any, Union


class CustomLoraConfig(LoraConfig):

    def __init__(self, **kwargs):
        super().__init__()
        self.r = kwargs.get('r', self.r)
        self.sampling_method = kwargs.get('sampling_method', 'inverted_probs')

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

        LoraLayer.__init__(self, base_layer)  # Initializes empty lora_A, lora_B param dicts

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
        self.ephemeral_gpu_offload = kwargs.get("ephemeral_gpu_offload", False)
        self.kwargs = kwargs

        # Save r and sampling_method
        r = kwargs.get("r", 64)
        sm = kwargs.get('sampling_method', 'inverted_probs')

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        W = base_layer.weight.data
        self.C, self.R = self.compute_C_and_R(W, r, sm)
        # Freeze C, R (we will only update U)
        self.C.requires_grad = False
        self.R.requires_grad = False

        # U must be a member of ParameterDict in order to be registered by peft as a trainable parameter
        self.lora_U = nn.ParameterDict({
            "lora_U": nn.Parameter(torch.zeros(self.C.size(1), self.R.size(0)), requires_grad=True)
        })

        self.adapter_name = adapter_name  # Defaults to "default"

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

        # NOTE: Standard LoraLayer would init A, B here, but we init C, R, and U within __init__()

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        # We don't need to reset parameters, in our use case
        pass

    # NOTE: skipping scale_layer, unscale_layer implementation. The base class implementations shouldn't affect C, U, R at all
    # NOTE: reimplimenting check_forward_args and mixed_batch_forward. Don't *think* the base implementations should affect curlora, but just in case

    def _check_forward_args(self, x, *args, **kwargs):
        pass

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # Base layer forward pass
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        # Select all unique adapters (in our use case, should only be "default")
        # It's still important to implement this function, however, in case automatic standard lora replacement is accidentally being triggered
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        # Compute curlora delta
        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter != self.adapter_name:  # Or check if active_adapter in self.lora_U.keys()
                continue

            C = self.C
            R = self.R
            U = self.lora_U["lora_U"]

            # Extract sub-batch corresponding to this adapter
            sub_batch = x[sub_batch_indices_list[i]]

            delta_weight = C @ U @ R
            sub_output = sub_batch @ delta_weight.t()
            
            result[sub_batch_indices_list[i]] += sub_output.to(torch_result_dtype)

        return result

    
    # Only supported sampling method currently is inverted probabilities 
    def compute_C_and_R(self, W, rank, sampling_method): 
        num_rows, num_cols = W.size()
        # Adjust rank if needed
        rank = min(rank, num_rows, num_cols)
        total_norm = torch.norm(W) ** 2

        # Col norm and probs
        col_norms = torch.norm(W, dim=0) ** 2 
        col_probs = col_norms / total_norm 
        # Invert col probs
        inv_col_probs = 1 / col_probs
        inv_col_probs /= inv_col_probs.sum() #normalize

        # Row norms and probs (chatgpt corrected)
        row_norms = torch.norm(W, dim=1) ** 2 
        row_probs = row_norms / total_norm 
        # Inver row probs
        inv_row_probs = 1 / row_probs
        inv_row_probs /= inv_row_probs.sum() # Normalize

        # Sample (based on inverted probabilities => lowest probabilities prioritized)
        col_indices = torch.multinomial(inv_col_probs, rank, replacement=False)
        row_indices = torch.multinomial(inv_row_probs, rank, replacement=False)

        C = W[:, col_indices]
        R = W[row_indices, :]
    
        return C, R
    
class CurloraLayer(nn.Module, CurloraLayerInner):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  
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
        self.is_target_conv_1d_layer = is_target_conv_1d_layer # Shouldn't be relevant to this use case

    def merge(self, safe_merge: bool = False) -> None:
        if self.merged:
            return
        
        base_layer = self.get_base_layer()
        delta_weight = self.get_delta_weight()

        if safe_merge:
            # Clone original weights and check for NaNs after merge
            orig_weights = base_layer.weight.data.clone()
            merged_weights = orig_weights + delta_weight
            if not torch.isfinite(merged_weights).all():
                raise ValueError("NaNs detected in the merged weights.")
            base_layer.weight.data = merged_weights
        else:
            # Default : only implemented safe_merge in case required by larger peft
            base_layer.weight.data += delta_weight

        self.merged_adapters.append(self._active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        active_adapter = self.merged_adapters.pop()
        # We only have one adapter in this use case. Thus, active_adapter should match self._active_adapter
        if active_adapter == self._active_adapter:
            base_layer = self.get_base_layer()
            delta_weight = self.get_delta_weight()

            # Subtract out delta_weight
            base_layer.weight.data -= delta_weight

    # Called by peft module itself (thus needs to be overriden)
    def get_delta_weight(self) -> torch.Tensor:
        C = self.C
        R = self.R
        U = self.lora_U["lora_U"]

        delta_weight = C @ U @ R
        return delta_weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Forward pass of base_layer
        result = self.base_layer(x)

        if self.disable_adapters:
            return result
            
        # Compute W_adapted
        U = self.lora_U["lora_U"]
        W_adapted = self.C @ U @ self.R

        # Compute output from W_adapted (will add this to result, so result = x @ (self.base_layer.weight + W_adapted).t())
        output = x @ W_adapted.t()
        if self.base_layer.bias is not None:
            output += self.base_layer.bias

        # Check shape and dtype
        assert W_adapted.shape == self.base_layer.weight.shape, (
            f"W_adapted shape {W_adapted.shape} does not match "
            f"original layer weight shape {self.base_layer.weight.shape}"
        )
        expected_dtype = result.dtype
        output = output.to(expected_dtype)

        result += output
        return result


### deprecated ###
# NOTE: following approach works, but with poor performance
# Likely because 1) doesn't take advantage of peft library optimizations and 2) merge_and_unload() function needed for proper integration with perplexity

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
