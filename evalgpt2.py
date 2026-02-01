import lm_eval
import json
from lm_eval.utils import handle_non_serializable
import os

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
        for i in range(0, len(requests), self._batch_size):
            batch_size = min(self._batch_size, len(requests) - i)
            
            batch_requests = requests[i:i+batch_size]
            padded_tokens, batch_mask = self._calculate_padded_tokens_and_mask(batch_requests)
            batch_log_probs, batch_is_greedy_decoding = self._batch_calculate_loglikelihood(padded_tokens, batch_mask)
            
            out.extend([(log_prob.item(), is_greedy_decoding.item()) for log_prob, is_greedy_decoding in zip(batch_log_probs, batch_is_greedy_decoding)])
        return out
    
    def _calculate_padded_tokens_and_mask(self, batch_requests):
        batch_context_tokens = [self.tokenizer.encode(req.args[0]) for req in batch_requests]
        batch_continuation_tokens = [self.tokenizer.encode(req.args[1]) for req in batch_requests]
        batch_context_lengths = []
        for i in range(len(batch_requests)):
            if len(batch_context_tokens[i]) == 0:
                batch_context_tokens[i] = [EOT]
            batch_context_lengths.append(len(batch_context_tokens[i]))
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
        token_ids = batch_tokens.to(self._device) if batch_tokens.device != self._device else batch_tokens  # B T
        logits, _ = self.model(token_ids[:, :-1])  # B T-1 V
        logprobs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = torch.gather(logprobs, dim=2, index=token_ids[:, 1:].unsqueeze(2))
        cont_mask = batch_mask[:, 1:].unsqueeze(2)  # True only on continuation positions
        per_token_log_probs = per_token_log_probs * cont_mask

        # is_greedy: True iff model would greedily generate the *continuation* (not context)
        match = torch.argmax(logits, dim=-1) == token_ids[:, 1:]
        is_greedy_decoding = (match | ~batch_mask[:, 1:]).all(dim=1)
        return per_token_log_probs.sum(dim=1), is_greedy_decoding

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling is not implemented")

    def generate_until(self, requests):
        raise NotImplementedError("generate_until is not implemented")

    @property
    def batch_size(self):
        return self._batch_size

def get_hellaswag_estimates(model, batch_size=8, device=None, limit=100):
    model = model.to(device)
    model.eval()
    my_gpt2_lm = MyGPT2LM(model, batch_size=batch_size, device=device)
    results = lm_eval.simple_evaluate(
        model=my_gpt2_lm,
        model_args="pretrained=gpt2,dtype=float32",
        tasks=["hellaswag"],
        num_fewshot=5,
        batch_size=batch_size,
        limit=limit,
        device=device,
    )
    return results["results"]["hellaswag"]["acc,none"], results["results"]["hellaswag"]["acc_norm,none"], results["results"]["hellaswag"]["acc_stderr,none"], results["results"]["hellaswag"]["acc_norm_stderr,none"]

if __name__ == "__main__":
    limit = 100

    from model import GPT2, GPT2config, load_model

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model()
    model = model.to(device)
    model.eval()
    my_gpt2_lm = MyGPT2LM(model, batch_size=8, device=device)

    from lm_eval.api.instance import Instance

    # padded_tokens, batch_mask = my_gpt2_lm._calculate_padded_tokens_and_mask(
    #     [Instance(arguments=("Hello, I'm a language model,", " and I'm here to help you."), request_type=None, doc=None, idx=None), Instance(arguments=("cat", " is an animal."), request_type=None, doc=None, idx=None)]
    #     )
    # batch_log_probs, batch_is_greedy_decoding = my_gpt2_lm._batch_calculate_loglikelihood(padded_tokens, batch_mask)
    # print(batch_log_probs.tolist())
    # print(batch_is_greedy_decoding.tolist())

    model = model.to(device)
    model.eval()
    my_gpt2_lm = MyGPT2LM(model, batch_size=8, device=device)
    import time
    start_time = time.time()
    my_gpt2_lm_results = lm_eval.simple_evaluate(
        model=my_gpt2_lm,
        model_args="pretrained=gpt2,dtype=float32",
        tasks=["hellaswag"],
        num_fewshot=5,
        batch_size=8,
        limit=limit,
        device=device,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    
    # save to eval fodler
    os.makedirs("eval", exist_ok=True)
    with open("eval/hellaswag_mygpt2_results.json", "w") as f:
        json.dump(my_gpt2_lm_results, f, default=handle_non_serializable, indent=2)
    # print the results
    print(f"MyGPT2 LM Hellaswag accuracy: {my_gpt2_lm_results['results']['hellaswag']['acc,none']} (+/- {my_gpt2_lm_results['results']['hellaswag']['acc_stderr,none']}) Normalized: {my_gpt2_lm_results['results']['hellaswag']['acc_norm,none']} (+/- {my_gpt2_lm_results['results']['hellaswag']['acc_norm_stderr,none']})")



    hf_gpt2_lm_results = lm_eval.simple_evaluate(
        model="hf",
        model_args="pretrained=gpt2,dtype=float32",
        tasks=["hellaswag"],
        num_fewshot=5,
        batch_size=8,
        limit=limit,
        device=device,
    )

    with open("eval/hellaswag_hf_results.json", "w") as f:
        json.dump(hf_gpt2_lm_results, f, default=handle_non_serializable, indent=2)
    # print the results
    print(f"HF GPT2 LM Hellaswag accuracy: {hf_gpt2_lm_results['results']['hellaswag']['acc,none']} (+/- {hf_gpt2_lm_results['results']['hellaswag']['acc_stderr,none']}) Normalized: {hf_gpt2_lm_results['results']['hellaswag']['acc_norm,none']} (+/- {hf_gpt2_lm_results['results']['hellaswag']['acc_norm_stderr,none']})")