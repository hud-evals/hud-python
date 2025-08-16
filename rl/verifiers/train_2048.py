"""
Training Script for 2048 (2 GPUs)
Terminal 1 - Start vLLM server:
  CUDA_VISIBLE_DEVICES=0 vf-vllm \
      --model Qwen/Qwen2.5-3B-Instruct \
      --enforce-eager \
      --disable-log-requests

  Terminal 2 - Run training:
  CUDA_VISIBLE_DEVICES=1 python train_2048.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verifiers as vf

vf_env = vf.load_environment(
    env_id="hud_vf_gym",
    taskset="hud-evals/2048-taskset",  # HuggingFace dataset
    config_path="verifiers/configs/2048.yaml",
    num_tasks=1,
)

# Model configuration
model_name = "Qwen/Qwen2.5-3B-Instruct"
run_name = "2048-grpo_" + model_name.split("/")[-1].lower()

# Load model and tokenizer
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Get default GRPO training arguments
training_args = vf.grpo_defaults(run_name=run_name)
training_args.gradient_accumulation_steps = 2

training_args.per_device_train_batch_size = 8
training_args.num_generations = 16
training_args.max_tokens = 1024
training_args.max_seq_len = 2048
training_args.max_prompt_length = 2048
training_args.learning_rate = 1e-6

training_args.async_generation_timeout = 900

training_args.max_steps = 100
training_args.save_strategy = "steps"
training_args.save_steps = 10
training_args.logging_steps = 1

training_args.mask_env_responses = True

# Create GRPO trainer with LoRA
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults(),
)

# Start training
print(f"Starting training for {run_name}")  # noqa: T201
print(f"Dataset size: {len(vf_env.dataset)}")  # noqa: T201
trainer.train()
