"""
Subclasses of FineTuneEvaluator for evaluating finetuning methods on particular model types, e.g. BERT
"""


from src.evaluator import FineTuneEvaluator


class BertBaseFineTuneEvaluator(FineTuneEvaluator):
    def __init__(self,
                 dataset,
                 training_args,
                 max_length,
                 lora_config,
                 pruning_method,
                 sparsity_target,
                 alpha,
                 temp,
                 device,
                 save_dir,
                 eval_ppl):
        model_name = 'bert-base-uncased'
        super().__init__(model_name, dataset, training_args, max_length, lora_config, pruning_method, sparsity_target, alpha, temp, device, save_dir, eval_ppl)
        
    def get_target_modules(self):
        target_modules = [
            # Self-Attention Modules
            "query",
            "key",
            "value",
            "attention.output.dense",

            # Feed-Forward Modules
            "intermediate.dense",
            "output.dense",

            #additional!

            # Pooler Layer
            "pooler.dense",
            
            # Task-Specific Heads
            "classifier.dense",
            "classifier.out_proj",
            "predictions.transform.dense",
            "predictions.decoder"
        ]
        return target_modules

"""Found linear layer: query
Found linear layer: key
Found linear layer: value
Found linear layer: dense
Found linear layer: classifier"""


class DistilBertFineTuneEvaluator(FineTuneEvaluator):
    def __init__(self,
                 dataset,
                 training_args,
                 max_length,
                 lora_config,
                 pruning_method,
                 sparsity_target,
                 alpha,
                 temp,
                 device,
                 save_dir,
                 eval_ppl,
                 num_samples):
        model_name = 'distilbert-base-uncased'
        super().__init__(model_name, dataset, training_args, max_length, lora_config, pruning_method, sparsity_target, alpha, temp, device, save_dir, eval_ppl, num_samples)
        
    def get_target_modules(self):
        return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"] #, "pre_classifier", "classifier"] #['q_lin', 'k_lin', 'v_lin', 'lin1', 'lin2']
    