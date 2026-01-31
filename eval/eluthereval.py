import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2,dtype=float32",
    tasks=["hellaswag"],
    num_fewshot=5,
    batch_size=8,
    limit=100,
    device="cuda:0",
)