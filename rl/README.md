# HUD environments for Reinforcement Learning

[hud-vf-gym](https://github.com/hud-evals/hud-vf-gym) module exposes CUA environments built with HUD MCP for training and evaluating RL agents using the [Verifiers](https://github.com/willccbb/verifiers) framework. It provides a standardized interface for agents to interact with computer interfaces through tool calls.


Need help? Join the Discord.
[![Discord](https://img.shields.io/discord/1327447144772407390?label=Discord&logo=discord&style=flat-square)](https://discord.gg/wkjtmHYYjm)


## Installation

You can directly install hud-vf-gym in this workspace `hud-python/rl`

```bash
# Clone hud-vf-gym
git clone https://github.com/hud-evals/hud-vf-gym.git

# Install dependencies (we recommend using uv for managing python envs)
uv sync

# Activate venv
source .venv/bin/activate

# Export environment variables
export OPENAI_API_KEY="YOUR_API_KEY"  # for running evals in openai models
export HUD_API_KEY="YOUR_API_KEY"   # for telemetry
```

if you don't have a hud api key, you can get one through the [HUD platform](https://app.hud.so).

## Running Evaluations

Use the Verifiers CLI to run evaluations such as hud-evals/2048-taskset.

For this, first build the base docker image locally:

```bash
cd ../environments/text_2048/
docker build -t hud-text-2048 .
```

Switch back to the workspace,

```bash
cd ../../rl
```

This will load in the taskset and run the gym via the config at ./configs/2048.yaml:
```bash
vf-eval hud-vf-gym \
    --model gpt-4.1-mini \
    --env-args '{"taskset": "hud-evals/2048-taskset", "config_path": "./configs/2048.yaml"}' \
    --num-examples 2 \
    --rollouts-per-example 3
```

Or use a custom config with a custom taskset:
```bash
# Use a custom config with custom taskset
vf-eval hud-vf-gym \
    --env-args '{"taskset": "your-org/your-taskset", "config_path": "custom_config.yaml"}' \
    --model gpt-4.1-mini \
    --num-examples 5 \
    --rollouts-per-example 3
```

You can also load tasks from a local JSON/JSONL file by passing a file path in `taskset`:
```bash
vf-eval hud-vf-gym \
    --model gpt-4.1-mini \
    --env-args '{"taskset": "./tasks.json", "config_path": "./configs/2048.yaml"}' \
    --num-examples 2 \
    --rollouts-per-example 3
```
Supported formats are a JSON array of tasks, a dict with a top-level `data` list, or JSON Lines (one JSON object per line). Each task should follow the HUD Task format.

Example using the included browser 2048 files:
```bash
vf-eval hud-vf-gym \
    --model gpt-4.1-mini \
    --env-args '{"taskset": "./data/browser_2048.json", "config_path": "./configs/browser_2048.yaml"}' \
    --num-examples 2 \
    --rollouts-per-example 3
```

## Training with GRPO

Verifier's GRPOtrainer is optimized for at least 2 GPUs. You can rent GPUs on marketplaces for [<$1/hr](https://app.primeintellect.ai).

HUD Gym supports training with the GRPO (Group Relative Policy Optimization) trainer:

Make sure you have the training dependencies installed:

```python
uv pip install 'verifiers[train]' && uv pip install flash-attn --no-build-isolation
```

Either just run:

```bash
python train_2048.py
```

Or configure your own training:
```python
from verifiers.trainers import GRPOTrainer, GRPOConfig
from verifiers import load_environment

# Load environment (both taskset and config_path are required)
env = load_environment(
    taskset="hud-evals/gmail-taskset",
    config_path="./configs/default.yaml"
)

# Configure training
config = GRPOConfig(
    model_name_or_path="your-model",
    per_device_train_batch_size=4,
    # ... other training parameters
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    env=env,
    args=config,
    processing_class=tokenizer,
)

# Train
trainer.train()
```

To train the 2048 agent on 2 A100 GPUs, use `train_2048.py` with one GPU for inference and one for training (see script header for setup commands).

For any issues related to Verifiers, see [their docs](https://verifiers.readthedocs.io/en/latest/training.html).

## Configuration

HUDGym consumes a simple YAML file. These keys are supported:

- job: name and optional metadata (dataset, experiment, dataset_link) for HUD telemetry
- system_prompt: multiline instructions passed to the agent
- defaults: max_turns integer
- rubric: optional weights dict
- allowed_tools: list of tool names to allow

Generate a starter template:
```bash
hudvf-config-init --output ./configs/template.yaml
```

Example minimal config:
```yaml
job:
  name: "My Run"
  metadata:
    dataset: "your-dataset"
system_prompt: |
  You are an AI assistant.
defaults:
  max_turns: 30
allowed_tools: []
```

## Dataset Format

HUD Gym uses HuggingFace datasets or JSON files with hud.Task format:

```python
{
    "id": "task-001",
    "prompt": "Click on the submit button",
    "mcp_config": {...},  # MCP configuration as JSON string
    "setup_tool": {...},   # Setup tool call as JSON string
    "evaluate_tool": {...}, # Evaluation tool call as JSON string
    "metadata": {...}       # Additional metadata as JSON string
}
```

When loading from a file path in `taskset`, place objects in a JSON array, a `{ "data": [...] }` structure, or use JSON Lines where each line is a task object.

## Reward Functions

By default, HUDBaseRubric combines:

1. Task Completion - Primary reward from HUD evaluation
2. Tool Execution - Success rate of tool calls

You can adjust weights via the `rubric.weights` section in the config.

### Adding new HUD Environments

To work with a new environment:

1. Create a config file (`configs/my_env.yaml`) with at least:
```yaml
job:
  name: "My Env"
system_prompt: |
  Instructions for the agent...
defaults:
  max_turns: 30
allowed_tools: []  # populate as needed
```

2. Run with your config:
```bash
vf-eval hud-vf-gym \
  --env-args '{"taskset": "your-org/your-taskset", "config_path": "configs/my_env.yaml"}' \
  --model gpt-4o-mini
```

### Adding New Tools to Existing Environments

1. Update the system prompt to describe the tool and how to use it.
2. Add the tool name to `allowed_tools` in your config.
3. Ensure your dataset's `mcp_config` enables/points to the MCP server that implements the tool.

### Creating Datasets

Convert tasks to HuggingFace format:

```python
from datasets import Dataset
import json

# Load your tasks
tasks = [...]

# Convert to HF format
dataset_dict = {
    "id": [t["id"] for t in tasks],
    "prompt": [t["prompt"] for t in tasks],
    "mcp_config": [json.dumps(t["mcp_config"]) for t in tasks],
    "setup_tool": [json.dumps(t["setup_tool"]) for t in tasks],
    "evaluate_tool": [json.dumps(t["evaluate_tool"]) for t in tasks],
    "metadata": [json.dumps(t.get("metadata", {})) for t in tasks],
}

dataset = Dataset.from_dict(dataset_dict)
dataset.push_to_hub("your-org/your-dataset")
```

## Troubleshooting

### Common Issues

1. **"Unknown tool" errors**: Ensure action mappings are correctly configured
2. Tool-calling failures: Ensure your model supports tool calls and `allowed_tools` is set correctly in the config
3. **MCP connection issues**: Verify MCP configuration in dataset
4. **Low rewards**: Review rubric weights and ensure evaluation tool returns grades

## License

See LICENSE file in the hud-vf-gym directory.
