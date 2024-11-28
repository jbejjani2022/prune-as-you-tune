"""
Subclasses of FineTuneEvaluator for evaluating finetuning methods on particular model types, e.g. BERT
"""


from src.evaluator import FineTuneEvaluator


class BertBaseFineTuneEvaluator(FineTuneEvaluator):
    def __init__(self, dataset, training_args, lora_config, pruning_method, device):
        model_name = 'bert-base-uncased'
        super().__init__(model_name, dataset, training_args, lora_config, pruning_method, device)
        
    def get_target_modules(self):
        return ['query', 'key', 'value', 'intermediate.dense']


class DistilBertFineTuneEvaluator(FineTuneEvaluator):
    def __init__(self, dataset, training_args, lora_config, pruning_method, device, num_samples): #TODO: make num_samples an optional parameter
        model_name = 'distilbert-base-uncased'
        super().__init__(model_name, dataset, training_args, lora_config, pruning_method, device, num_samples)
        
    def get_target_modules(self):
        return ['q_lin', 'k_lin', 'v_lin', 'lin1', 'lin2']
    