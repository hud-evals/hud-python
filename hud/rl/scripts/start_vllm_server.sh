#!/bin/bash
# Start vLLM server with OpenAI-compatible API

echo "Starting vLLM server for Qwen2.5-VL-3B-Instruct..."

export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0 uv run vllm serve \
    Qwen/Qwen2.5-VL-3B-Instruct \
    --api-key token-abc123 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template /home/ubuntu/hud-python/hud/rl/chat_template.jinja \
    --enable-log-requests \
    --uvicorn-log-level=debug 2>&1 | tee vllm_debug.log
