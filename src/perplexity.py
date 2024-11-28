from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from tqdm import tqdm

from datasets import load_dataset


class PPL:
    
    def __init__(self, model_name, device):
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")  # https://huggingface.co/datasets/Salesforce/wikitext
        print(f'num samples = {test_dataset.num_rows}')
        s = "\n\n".join(test_dataset["text"])
        print(len(s))
        print(s[:100])
        self.encodings = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")

    def score(self, model, input):
        # https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
        # https://stackoverflow.com/questions/63030692/how-do-i-use-bertformaskedlm-or-bertmodel-to-calculate-perplexity-of-a-sentence
        repeat_input = input.repeat(input.size(-1)-2, 1)
        mask = torch.ones(input.size(-1) - 1).diag(1)[:-2].to(device)
        masked_input = repeat_input.masked_fill(mask == 1, self.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != self.mask_token_id, -100)
        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss
        return loss.item()

    def calculate_perplexity(self, model):
        max_length = model.config.max_position_embeddings
        print(f'max length = {max_length}')
        stride = 512
        seq_len = self.encodings.input_ids.size(1)
        print(f'seq len = {seq_len}')

        nlls = []
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            score = self.score(model, input_ids)
            nlls.append(score)
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl

    def calculate_perplexity_v1(self, model):
        # https://huggingface.co/docs/transformers/en/perplexity
        max_length = model.config.max_position_embeddings
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
            
            # decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # print('*******************************')
            # print(begin_loc)
            # print(decoded_text)

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
    model_name = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    
    evaluator = PPL(model_name, device)
    ppl = evaluator.calculate_perplexity(model)
    print(f'perplexity = {ppl}')
