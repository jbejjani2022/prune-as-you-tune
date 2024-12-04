from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel


class PPL:
    
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token_id = tokenizer.mask_token_id
        # load in "wikitext" test set for evaluating masked LM task
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")  # https://huggingface.co/datasets/Salesforce/wikitext
        print(f'num samples = {test_dataset.num_rows}')
        # join test set into one long sequence, and encode
        self.encodings = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")

    def is_peft_model(self, model):
        """
        Checks if the given model is a PEFT model with LoRA adapters.
        
        Parameters:
            model: A model instance to check.
            
        Returns:
            bool: True if the model is a PEFT model with LoRA adapters, False otherwise.
        """
        # Check if the model is an instance of PeftModel
        return isinstance(model, PeftModel)

    # evaluates pseudo perplexity of a model finetuned for sequence classification
    # if `path` is provided, loads in model from saved checkpoint. otherwise, a loaded model must be passed as `model` arg
    # model must be for SequenceClassification
    # `path` currently only works as expected if it points to a non-pruned model checkpoint (from_pretrained(path) does not properly load in the weight_mask module buffers from pruning)
    def eval(self, path=None, model=None):
        if path:
            # load in finetuned model from a checkpoint
            print(f'loading model from {path}...')
            finetuned_seq_model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        elif model:
            finetuned_seq_model = model
        else:
            raise ValueError("eval() expects a loaded model or path to a model checkpoint")

        # if the model was finetuned with LoRA, merge the adapters into the bert backbone
        if self.is_peft_model(finetuned_seq_model):
            print('merging lora adapters into bert...')
            finetuned_seq_model = finetuned_seq_model.merge_and_unload()

        # extract bert parameters from the finetuned seq classifier
        #base_model = finetuned_seq_model.bert
        base_model = finetuned_seq_model.distilbert
        
        # copy over finetuned seq model parameters to a masked LM architecture
        masked_lm_model = AutoModelForMaskedLM.from_pretrained(self.model_name, config=base_model.config).to(self.device)
        #masked_lm_model.bert = base_model
        masked_lm_model.distilbert = base_model
        
        # calculate pseudo perplexity of masked LM
        print(f'calculating (pseudo) perplexity...')
        ppl = self.calculate_perplexity_pseudo(masked_lm_model)
        return ppl

    # calculates negative log likelihood for `model` on one `input` sentence by masking each token one at a time and running a batched forward pass
    # does not give true perplexity, because full sentence is used in predicting each masked token (i.e. bidirectional)
    # as opposed to a next-token prediction model like GPT, which only uses tokens to the left
    # https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
    # https://stackoverflow.com/questions/63030692/how-do-i-use-bertformaskedlm-or-bertmodel-to-calculate-perplexity-of-a-sentence
    def score(self, model, input):
        repeat_input = input.repeat(input.size(-1)-2, 1)
        mask = torch.ones(input.size(-1) - 1).diag(1)[:-2].to(self.device)
        masked_input = repeat_input.masked_fill(mask == 1, self.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != self.mask_token_id, -100)
        with torch.inference_mode():
            output = model(masked_input, labels=labels)
        return output.loss
    
    # same as `score` but computes loss via two batches of half the size
    # useful if running out of CUDA memory
    def score_save_mem(self, model, input):
        repeat_input = input.repeat(input.size(-1)-2, 1)
        mask = torch.ones(input.size(-1) - 1).diag(1)[:-2].to(self.device)
        masked_input = repeat_input.masked_fill(mask == 1, self.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != self.mask_token_id, -100)
        
        # Split the inputs into two halves
        split_size = masked_input.size(0) // 2
        masked_input_1, masked_input_2 = torch.split(masked_input, [split_size, masked_input.size(0) - split_size])
        labels_1, labels_2 = torch.split(labels, [split_size, labels.size(0) - split_size])
        
        with torch.inference_mode():
            # First forward pass
            output1 = model(masked_input_1, labels=labels_1)
            # Second forward pass
            output2 = model(masked_input_2, labels=labels_2)
        
        # Combine the losses
        combined_loss = (output1.loss * masked_input_1.size(0) + output2.loss * masked_input_2.size(0)) / masked_input.size(0)
        
        return combined_loss

    # calculates pseudo perplexity over test dataset
    # sweeps over test dataset in steps of size `stride`; adjust to balance accuracy of estimation with efficiency
    # `model` must be for masked LM
    def calculate_perplexity_pseudo(self, model):
        max_length = model.config.max_position_embeddings  # model's max context length - determines size of input to `score`
        print(f'max length = {max_length}')
        stride = 512
        seq_len = self.encodings.input_ids.size(1)  # full length of encoded test dataset
        print(f'seq len = {seq_len}')

        nlls = []  # negative log likelihoods for each eval sentence
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            score = self.score(model, input_ids)
            nlls.append(score)
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl

    def calculate_perplexity_true(self, model):
        # calculates true perplexity; expects model to be a causal LM, e.g. GPT2LMHeadModel
        # source: https://huggingface.co/docs/transformers/en/perplexity
        max_length = model.config.n_positions
        print(f'max length = {max_length}')
        stride = 512
        seq_len = self.encodings.input_ids.size(1)
        
        print(f'encodings.input_ids.size = {self.encodings.input_ids.size}')
        print(f'seq len = {seq_len}')

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_name = 'bert-base-uncased'
    model_name = 'distilbert-base-uncased'
    
    # Get perplexity of base, non-finetuned bert model
    # model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    ppl = PPL(model_name, device)
    # ppl = evaluator.calculate_perplexity_pseudo(model)
    # print(f'perplexity = {ppl}')
    
    # Test loading finetuned model checkpoints and evaluating pseudo perplexity for each
    # Each checkpoint is an instance of AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    # finetuned on 'imdb' for sequence classification
    
    # root = 'bert-imdb-r32-nomaxlen/'
    # paths = ['50pct-sparsity-5epochs/checkpoints/full_finetune/ckpt-epoch-5',
    #         '50pct-sparsity-10epochs/checkpoints/full_finetune/ckpt-epoch-10',
    #         '80pct-sparsity-8epochs/checkpoints/full_finetune/ckpt-epoch-8',
    #         '80pct-sparsity-16epochs/checkpoints/full_finetune/ckpt-epoch-2']
    
    root = 'distilbert-imdb-r32-nomaxlen/'
    paths = ['50pct-sparsity-5epochs/checkpoints/lora_finetune/ckpt-epoch-5']#,
            #'50pct-sparsity-10epochs/checkpoints/lora_finetune/ckpt-epoch-10',
            #'80pct-sparsity-8epochs/checkpoints/lora_finetune/ckpt-epoch-8',
            #'80pct-sparsity-16epochs/checkpoints/lora_finetune/ckpt-epoch-14']
    
    for ckpt in paths:
        perplexity = ppl.eval(path = root + ckpt)
        print(f'perplexity = {perplexity}')
