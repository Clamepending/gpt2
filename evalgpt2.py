import lm_eval
import json
from lm_eval.utils import handle_non_serializable


from lm_eval.api.model import LM
import torch
import tiktoken
from lm_eval.api.model import LM
from torch.nn import functional as F

EOT = 50256


class MyGPT2LM(LM):
    def __init__(self, model, batch_size=1, device=None):
        super().__init__()
        self.model = model
        self._batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self._device = device or next(model.parameters()).device

    def loglikelihood(self, requests):
        # INPUT:  requests = list[Instance], each req.args = (context: str, continuation: str)
        # OUTPUT: list[tuple[float, bool]] — (log P(continuation|context), is_greedy_match)
        out = []
        for i in range(len(requests), self._batch_size):
            batch_size = min(self._batch_size, len(requests) - i)
            
            batch_requests = requests[i:i+batch_size]
            padded_tokens, batch_mask = self.calculate_padded_tokens_and_mask(batch_requests)
            batch_log_probs, batch_is_greedy_decoding = self._batch_calculate_loglikelihood(padded_tokens, batch_mask)
            
            out.extend([(log_prob.item(), is_greedy_decoding.item()) for log_prob, is_greedy_decoding in zip(batch_log_probs, batch_is_greedy_decoding)])
        return out
    
    def _calculate_padded_tokens_and_mask(self, batch_requests):
        batch_context_tokens = [self.tokenizer.encode(req.args[0]) for req in batch_requests]
        batch_continuation_tokens = [self.tokenizer.encode(req.args[1]) for req in batch_requests]
        batch_context_lengths = [len(tokens) for tokens in batch_context_tokens]
        idx_add_eot = batch_context_lengths == 0
        batch_context_tokens[idx_add_eot] = [EOT] + batch_context_tokens[idx_add_eot]
        batch_context_lengths[idx_add_eot] = 1
        batch_full_requests = [torch.tensor(batch_context_tokens[i] + batch_continuation_tokens[i], dtype=torch.long, device=self._device) for i in range(len(batch_requests))]
        padded_tokens = torch.nn.utils.rnn.pad_sequence(batch_full_requests, batch_first=True, padding_value=self.tokenizer.eot_token)
        
        # generate the batch mask
        batch_masks = []
        for context_tokens, full_tokens in zip(batch_context_tokens, batch_full_requests):
            mask = torch.ones(len(full_tokens), dtype=torch.bool, device=self._device)
            mask[:len(context_tokens)] = False
            batch_masks.append(mask)
            
        batch_mask = torch.nn.utils.rnn.pad_sequence(batch_masks, batch_first=True, padding_value=False)
        
        return padded_tokens, batch_mask
    
    def _batch_calculate_loglikelihood(self, batch_tokens, batch_mask):
        token_ids = torch.tensor(batch_tokens, dtype=torch.long, device=self._device)  # B T
        logits, _ = self.model(token_ids[:, :-1]) # B T-1 V -> B T-1 V
        logprobs = F.log_softmax(logits, dim=-1) # B T-1 V -> B T-1 LP
        per_token_log_probs = torch.gather(logprobs, dim=2, index=token_ids[:, 1:].unsqueeze(2)) # (B T-1 LP), (B T-1 1) -> B T-1 1
        per_token_log_probs = per_token_log_probs * batch_mask[:, 1:].unsqueeze(2) # B T-1 1
        
        is_greedy_decoding = (torch.argmax(logits, dim=-1) == token_ids[:, 1:]).all(dim=1) # B 1
        return per_token_log_probs.sum(dim=1), is_greedy_decoding # B 1, B 1

    def loglikelihood_rolling(self, requests):
        # INPUT:  requests = list[Instance], each req.args = (text: str,)
        # OUTPUT: list[tuple[float]] — (log P(text | EOT),)
        out = []
        for req in requests:
            (text,) = req.args
            if not text:
                out.append((0.0,))
                continue
            enc = self.tokenizer.encode(text)
            # TODO: run model on [EOT] + enc[:-1], sum log P(enc), append (ll,)
            out.append((0.0,))  # placeholder
        return out

    def generate_until(self, requests):
        # INPUT:  requests = list[Instance], each req.args = (prefix: str, gen_kwargs: dict)
        #        gen_kwargs has "until" (str or list[str]) and "max_gen_toks" (int)
        # OUTPUT: list[str] — generated continuation for each request (stop strings stripped)
        out = []
        for req in requests:
            prefix, kw = req.args
            until = kw.get("until", ["\n"])
            until = [until] if isinstance(until, str) else until
            max_toks = kw.get("max_gen_toks", 256)
            # TODO: encode prefix, autoregressively sample until EOT or until or max_toks,
            #       decode and strip any until string from the end, append to out
            out.append("")  # placeholder
        return out

    @property
    def batch_size(self):
        return self._batch_size

from model import GPT2, GPT2config, load_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = load_model()
model = model.to(device)
model.eval()
my_gpt2_lm = MyGPT2LM(model, batch_size=8, device=device)

from lm_eval.api.instance import Instance

padded_tokens, batch_mask = my_gpt2_lm._calculate_padded_tokens_and_mask(
    [Instance(arguments=("Hello, I'm a language model,", " and I'm here to help you."), request_type=None, doc=None, idx=None), Instance(arguments=("cat", " is an animal."), request_type=None, doc=None, idx=None)]
    )
batch_log_probs, batch_is_greedy_decoding = my_gpt2_lm._batch_calculate_loglikelihood(padded_tokens, batch_mask)
print(batch_log_probs.tolist())
print(batch_is_greedy_decoding.tolist())

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