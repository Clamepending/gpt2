





```
uv run download_fineweb_shards.py
uv run torchrun --nproc_per_node=8 train_gpt2.py
```


sample (make sure the change the model name to the latest checkpoint):
```
uv run test_trained_model.py
```