"""
Custom callback for progressively pruning pretrained weights during fine-tuning with HF Trainer
"""


from transformers import TrainerCallback
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn


class PruningCallback(TrainerCallback):
    
    def __init__(self, 
                 model, 
                 method, 
                 lora, 
                 sparsity_target, 
                 num_epochs, 
                 schedule : str,
                 prune_every_epoch : int = 1):
        self.model = model
        self.num_epochs = num_epochs
        self.sparsity_target = sparsity_target
        # for the interleaving methods: how much to prune the pretrained weights before each epoch
        # more specifically, this is the ptg by which sparsity of pruning-eligible params will increase before each epoch
        self.lora = lora  # whether the model is being fine-tuned with lora
        # pruning method
        if method == "L1Unstructured":
            self.method = prune.L1Unstructured
        # support other methods here
        else:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        if schedule == "linear":
            # Linear schedule: prune a fixed percentage of remaining weights at each epoch
            self.schedule = [(sparsity_target/((num_epochs)/prune_every_epoch)) * (i % prune_every_epoch == 0) for i in range(num_epochs-1)]
        elif schedule == "agp":
            n_pruning_epochs = (num_epochs) // prune_every_epoch
            cumulative_pruning = [(sparsity_target * (1-(1 - i / n_pruning_epochs) ** 3))  for i in range(n_pruning_epochs)]
            self.schedule = np.zeros(num_epochs-1)
            self.schedule[1::prune_every_epoch] =  np.diff(cumulative_pruning) 
        else:
            raise ValueError(f"Unsupported pruning schedule: {schedule}")
        
        assert(sum(self.schedule)==sparsity_target)
    
        self.schedule.append(0) # don't prune after final epoch

        self.params = []  # params to prune, list of tuples: (module, param_name)

        # Iterate through all modules in the model
        # and get the frozen, non-LoRA parameters to be pruned
        # this includes all pretrained model weights, excludes trainable, non-LoRA classifier head
        # skip embedding layers to mitigate accuracy degradation
        for name, module in self.model.named_modules():
            # Ensure module has a weight attribute and parameter
            if hasattr(module, 'weight') and 'weight' in module._parameters:
                if not isinstance(module, nn.Embedding):  # Skip embedding layers
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
        if state.epoch == self.num_epochs:
            # Don't prune after final epoch
            return
        epoch_ptg = self.schedule[state.epoch]
        self.prune_pretrained(state.epoch, epoch_ptg)
        self.report_sparsity()

    # Applies pruning `weight_mask` to pretrained, non-LoRA model parameters
    # Retains previous mask, allowing cumulative pruning
    def prune_pretrained(self, epoch, epoch_ptg=None):
        print(f"\nPruning {epoch_ptg * 100:.2f}% of remaining pretrained weights after epoch {epoch}")

        prune.global_unstructured(
            self.params,
            pruning_method=self.method,
            amount=epoch_ptg,  # percentage of REMAINING (non-pruned) params to prune
        )
        
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
