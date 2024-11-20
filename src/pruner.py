"""
Custom callback for progressively pruning pretrained weights during fine-tuning with HF Trainer
"""


from transformers import TrainerCallback
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn


class PruningCallback(TrainerCallback):
    
    def __init__(self, model, method, ptg=0.1):
        self.model = model
        self.ptg = ptg  # percentage of params to prune
        
        # pruning method
        if method == "L1Unstructured":
            self.method = prune.L1Unstructured
        # support other methods here
        else:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        self.params = []  # params to prune, list of tuples: (module, param_name)

        # Iterate through all modules in the model
        # and get the frozen / non-LoRA parameters to be pruned
        # skip embedding layers to mitigate accuracy degradation
        for name, module in self.model.named_modules():
            # Ensure module has a weight attribute and parameter
            if hasattr(module, 'weight') and 'weight' in module._parameters:
                # Filter for non-embedding and frozen / non-trainable parameters
                if not module.weight.requires_grad and not isinstance(module, nn.Embedding):
                    assert("lora" not in name)
                    self.params.append((module, 'weight'))

        # Count total number of params and number of LoRA params for sparsity reports
        self.total_params = 0
        self.lora_params = 0

        for name, param in self.model.named_parameters():
            if "lora" in name:
                self.lora_params += param.numel()
            self.total_params += param.numel()

    # Function called at the end of each training epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nPruning {self.ptg * 100:.2f}% of pretrained weights after epoch {state.epoch}")
        self.prune_pretrained()
        self.report_sparsity()

    # Applies pruning `weight_mask` to pretrained, non-LoRA model parameters
    # Retains previous mask, allowing cumulative pruning
    def prune_pretrained(self):
        prune.global_unstructured(
            self.params,
            pruning_method=self.method,
            amount=self.ptg,
        )
        
    # Reports sparsity of frozen / pretrained weights, as well as overall model sparsity
    def report_sparsity(self):
        zero_params = 0
        
        # Iterate over all parameters in the model
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                zero_params += torch.sum(module.weight == 0).item()
        
        sparsity_frozen = (zero_params / (self.total_params - self.lora_params)) * 100 if self.total_params > 0 else 0
        sparsity_all = (zero_params / self.total_params) * 100 if self.total_params > 0 else 0

        print(f"Pretrained Weight Sparsity: {sparsity_frozen:.2f}%")
        print(f"Overall Model Sparsity (including LoRA): {sparsity_all:.2f}%")
        
    # Removes the pruning reparameterization to make pruning permanent
    # Necessary in order to consolidate pruning changes permanently or export the model
    # Do not execute until all pruning iterations have been completed
    def remove(self):
        for (module, name) in self.params:
            prune.remove(module, name)
