import lm_eval
import json
from lm_eval.utils import handle_non_serializable

eval = False
if eval:
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args="pretrained=gpt2,dtype=float32",
        tasks=["hellaswag"],
        num_fewshot=5,
        batch_size=8,
        limit=100,
        device="cuda:0",
    )

    with open("results.json", "w") as f:
        json.dump(results, f, default=handle_non_serializable, indent=2)
else:
    with open("results.json", "r") as f:
        results = json.load(f)
# print the results
print(results["results"]["hellaswag"])