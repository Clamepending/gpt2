import lm_eval
import json
from lm_eval.utils import handle_non_serializable


from lm_eval.api.model import LM
import torch
import tiktoken
from torch.nn import functional as F

class MyGPT2LM(LM):
    def __init__(self, model, batch_size=1, device=None):
        super().__init__()
        self.model = model
        self._batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._device = device or next(model.parameters()).device

    def loglikelihood(self, requests):
        # Return list of (logprob, is_greedy) tuples
        
        
        for request in requests:
            prompt = request[0]
            completion = request[1]
            token_ids = torch.tensor(self.tokenizer.encode(prompt + completion), dtype=torch.long, device=self._device).unsqueeze(0)  # B T1+T2
            token_ids = torch.cat((torch.tensor([[50256]], device=self._device), token_ids), dim=1)  # add <|endoftext|> token so B T1+T2+1
            logits, _ = self.model(token_ids[:, :-1]) # B T1+T2 V -> B T1+T2 V
            logprobs = F.log_softmax(logits, dim=-1) # B T1+T2 V -> B T1+T2 LP
            per_token_log_probs = torch.gather(logprobs, dim=2, index=token_ids[:, 1:].unsqueeze(2)) # (B T1+T2 LP), (B T1+T2 1) -> B T1+T2 1
            total_log_probs = per_token_log_probs.sum(dim=1) # B T1+T2 1 -> B 1
            
            # calculate if completion would have been greedy:
            greedy_indices = torch.argmax(logits, dim=-1) # B T1+T2 1
            is_greedy_decoding = (greedy_indices == token_ids[:, 1:]).all(dim=1)
            
            return total_log_probs.item(), is_greedy_decoding.item()

    def generate_until(self, requests):
        # Return list of generated strings
        pass

    def loglikelihood_rolling(self, requests):
        # Return list of (logprob, is_greedy) tuples
        pass


    @property
    def batch_size(self):
        return self._batch_size

from model import GPT2, GPT2config, load_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = load_model()
model = model.to(device)
model.eval()
my_gpt2_lm = MyGPT2LM(model, batch_size=8, device=device)

print(my_gpt2_lm.loglikelihood(
    [("Hello, I'm a language model,", " I'm a language model, and I'm here to help you.", "cat")]
    ))



# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=gpt2,dtype=float32",
#     tasks=["hellaswag"],
#     num_fewshot=5,
#     batch_size=8,
#     limit=100,
#     device="mps",
# )

# with open("results.json", "w") as f:
#     json.dump(results, f, default=handle_non_serializable, indent=2)
# # print the results
# print(results["results"]["hellaswag"])