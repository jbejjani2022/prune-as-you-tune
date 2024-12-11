"""
Custom callback for progressively pruning pretrained weights during fine-tuning with HF Trainer
"""


from transformers import TrainerCallback
import torch.nn.utils.prune as prune
import torch.nn as nn
import numpy as np
import math


class PruningCallback(TrainerCallback):
    
    def __init__(self,
                 model,
                 lora,
                 method,
                 sparsity_target,
                 num_epochs,
                 schedule : str,
                 prune_every_epoch : int,
                 pruning_start_epoch : int):
        self.model = model
        self.num_epochs = num_epochs
        self.sparsity_target = sparsity_target
        self.method = method
        self.lora = lora  # whether the model is being fine-tuned with lora
        if self.lora: print("Using LoRA")
        
        # self.schedule gives the ptg by which sparsity of pruning-eligible params will increase before each epoch
        # e.g. self.schedule[0] = how much to increase sparsity before first epoch (epoch 1), self.schedule[1] = how much to increase sparsity before epoch 2
        num_pruning_steps = math.ceil((num_epochs - pruning_start_epoch) / prune_every_epoch)  # how many times will we have a non-zero prune ptg
        self.schedule = np.zeros(num_epochs)

        if schedule == "linear":
            # Linear schedule: increase sparsity by a fixed percentage each epoch
            ptg = sparsity_target / num_pruning_steps  # how much to increase sparsity on each non-zero pruning step
            for i in range(num_epochs):
                self.schedule[i] = ptg * (i >= pruning_start_epoch and (i - pruning_start_epoch) % prune_every_epoch == 0)
        elif schedule == "agp":
            cumulative_pruning = [(sparsity_target * (1 - (1 - i / num_pruning_steps) ** 3))  for i in range(1, num_pruning_steps + 1)]
            increments = np.diff([0] + cumulative_pruning)
            self.schedule[pruning_start_epoch::prune_every_epoch] = increments
        else:
            raise ValueError(f"Unsupported pruning schedule: {schedule}")

        print(f"Pruning schedule: {self.schedule}")
        
        assert(sum(self.schedule)==sparsity_target), f"Total sparsity will be {sum(self.schedule)}, target is {sparsity_target}"

        self.params = []  # params to prune, list of tuples: (module, param_name)

        # Iterate through all modules in the model
        # and get the frozen, non-LoRA parameters to be pruned
        # this includes all pretrained model weights, excludes trainable, non-LoRA classifier head
        # skip embedding layers to mitigate accuracy degradation
        for name, module in self.model.named_modules():
            # Ensure module has a weight attribute and parameter
            if hasattr(module, 'weight') and 'weight' in module._parameters:
                if not isinstance(module, nn.Embedding):  # Skip embedding layers
                    # Skip params with 1 dim if using structured pruning
                    if self.method == "L2Structured":
                        param_tensor = getattr(module, 'weight')
                        if param_tensor.dim() == 1:
                            continue
                    if self.lora:
                        # Model is being fine-tuned with lora adapters
                        # Filter for non-trainable, non-LoRA parameters
                        if not module.weight.requires_grad and "classifier" not in name:
                            assert("lora" not in name)
                            self.params.append((module, 'weight'))
                    else:
                        # Model is being fully fine-tuned, with no LoRA adapters
                        if "classifier" not in name:  # Do not prune the classifier head
                            self.params.append((module, 'weight'))

        # Count total number of params and number of LoRA params for sparsity reports
        self.trainable_params, self.total_params = self._count_parameters()
        print(f'Trainable params: {self.trainable_params}')
        print(f'Total params: {self.total_params}')

    # Count trainable and total model parameters
    def _count_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, total_params

    # Function called at the end of each training epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        print(f'state.epoch = {epoch}')
        if epoch == self.num_epochs:
            # Don't prune after final epoch
            return
        current_sparsity = sum(self.schedule[:epoch])   # current sparsity of the pruning-eligible params
        epoch_ptg = self.schedule[epoch] / (1 - current_sparsity)     # how much to prune the remaining pruning-eligible params in order to increase sparsity by self.ptg
        self.prune_pretrained(epoch, epoch_ptg)
        self.report_sparsity()

    # Applies pruning `weight_mask` to pretrained, non-LoRA model parameters
    # Retains previous mask, allowing cumulative pruning
    def prune_pretrained(self, epoch, epoch_ptg=None):
        if epoch_ptg is None:
            assert(epoch == 0)
            epoch_ptg = self.schedule[epoch]
        
        print(f"\nPruning {epoch_ptg * 100:.2f}% of remaining pretrained weights before epoch {epoch + 1}, increasing sparsity of pretrained weights by {self.schedule[epoch] * 100:.2f}%")
        if self.method == "L2Structured":
            for module, name in self.params:
                prune.ln_structured(
                    module,
                    name,
                    amount=epoch_ptg,  # percentage of REMAINING (non-pruned) channels to prune
                    n=2,  # prunes channels based on L2 norm
                    dim=1,  # prune rows
                )
        elif self.method == "L1Unstructured":
            prune.global_unstructured(
                self.params,
                pruning_method=prune.L1Unstructured,
                amount=epoch_ptg,  # percentage of REMAINING (non-pruned) params to prune
            )
        else:
            raise ValueError(f"Unsupported pruning method: {self.method}")

    # Reports sparsity of pruning-eligible params, as well as overall model sparsity
    def report_sparsity(self):
        total = 0
        zeros = 0
        for module, name in self.params:
            weight = getattr(module, name)
            mask = getattr(module, f"{name}_mask", None)
            total += weight.numel()
            if mask is not None:
                # Use masked weights to calculate sparsity
                zeros += (weight * mask == 0).sum().item()
                # layer_sparsity = (weight * mask == 0).sum().item() / weight.numel()
        
        print(f'Pruning-eligible params: {total}')
        print(f'Sparsity of pruning-eligible params: {zeros / total:.2%}')
        print(f'Overall model sparsity: {zeros / self.total_params:.2%}')
        
    # Removes the pruning reparameterization to make pruning permanent
    # Necessary in order to consolidate pruning changes permanently or export the model
    # Do not execute until all pruning iterations have been completed
    def remove(self):
        for (module, name) in self.params:
            prune.remove(module, name)
