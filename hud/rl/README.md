We suggest running hud rl (or with the --local flag) for optimal hyperparameters and native HuggingFace running.

However, to run this independently, sping up an instance with at least 2 GPUs and run:
```bash
sudo apt-get update -y && sudo apt-get install -y cuda-toolkit-12-6
uv pip install -e .[rl]
uv pip install ninja
uv pip install flash-attn --no-build-isolation
```

Launch a vllm server with:
```bash
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=7 # Set this to your last GPU

uv run vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --api-key token-abc123 --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --trust-remote-code \
    --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests \
    --dtype auto
```

And training with (replace 2 with your spare GPUs):
```bash
hud get hud-evals/2048-basic
torchrun --nproc-per-node 2 -m hud.rl.train --tasks 2048-basic.json --verbose
```

Add a `--config path/to/config.json` flag to run a specific configuration (or change the defaults in config.py)
