# HUD environments for Reinforcement Learning through Verifiers

[hud-vf-gym](https://github.com/hud-evals/hud_vf_gym) module exposes CUA environments built with HUD MCP for training and evaluating RL agents using the [Verifiers](https://github.com/willccbb/verifiers) framework. It provides a standardized interface for agents to interact with computer interfaces through tool calls.

## Installation

```bash
# 1. Initialize and update the submodule (if not already)
git submodule update --init --recursive examples/rl/hud_vf_gym

# 2. Install RL dependencies
uv sync --group rl

# 3. Install hud_vf_gym in the your environment
vf-install examples/rl/hud_vf_gym
```

## Running Evaluations

Use the Verifiers CLI to run evaluations:

```bash
# Basic evaluation with default Gmail taskset
vf-eval hud_vf_gym --model gpt-4o-mini --num-examples 5

# Specify custom taskset (Huggingface Dataset for HUD tasks)
vf-eval hud_vf_gym \
    --model gpt-4o-mini \
    --env-args '{"taskset": "hud-evals/gmail-taskset"}' \
    --num-examples 10

# Use a custom config
vf-eval hud_vf_gym \
    --env-args '{"config_path": "custom_config.yaml"}' \
    --model gpt-4o-mini \
    --num-examples 5 \
    --rollouts-per-example 3
```

These might not work right now

```bash
# Save evaluation results locally
vf-eval hud_vf_gym \
    --model gpt-4o-mini \
    --num-examples 20 \
    --save-dataset \
    --save-path "./eval_results"

# Save to HuggingFace Hub
vf-eval hud_vf_gym \
    --model gpt-4o-mini \
    --env-args '{"taskset": "hud-evals/gmail-taskset", "num_tasks": 50}' \
    --num-examples 50 \
    --save-to-hf-hub \
    --hf-hub-dataset-name "your-org/gmail-eval-results"
```

## Training with GRPO

HUD Gym supports training with the GRPO (Group Relative Policy Optimization) trainer:

```python
from verifiers.trainers import GRPOTrainer, GRPOConfig
from hud_vf_gym import load_environment

# Load environment
env = load_environment(taskset="hud-evals/gmail-taskset")

# Configure training
config = GRPOConfig(
    model_name_or_path="your-model",
    num_train_epochs=3,
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

## Configuration

### Environment Configuration (`configs/default.yaml`)

The main configuration file controls various aspects of the environment:

#### System Prompt
Defines the instructions given to the agent, including available tools and usage guidelines.

```yaml
system_prompt: |
  You are an AI assistant that can interact with computer interfaces.
  
  Screen Information:
  - Resolution: 1280x800 pixels
  ...
```

#### Default Settings that are also Verifiers **kwargs
```yaml
defaults:
  max_turns: 30  # Maximum number of turns per task
```

#### Action Mappings
Maps agent action names defined in the system prompt to MCP computer tool format.

```yaml
action_mappings:
  screenshot:
    action: "screenshot"
  
  click:
    action: "click"
    x:
      from_arg: "x"
    y:
      from_arg: "y"
  
  type:
    action: "type"
    text:
      from_arg: "text"
      default: ""
  
  key:
    action: "press"
    keys:
      from_arg: "key"
      transform: "split_plus"  # Transforms "ctrl+a" to ["ctrl", "a"]
  
  scroll:
    action: "scroll"
    # ... scroll configuration
  
  wait:
    action: "wait"
    time:
      from_arg: "seconds"
      transform: "seconds_to_ms"  # Converts seconds to milliseconds
```

#### Parser Configuration
Controls XML parsing and action validation:

```yaml
parser:
  xml_weight: 0.6    # Weight for XML format validation
  action_weight: 0.4  # Weight for action syntax validation
```

#### Rubric Configuration
Defines reward function weights:

```yaml
rubric:
  weights:
    task_completion: 0.75      # Primary task completion from HUD evaluation
    tool_execution: 0.05       # Successful tool execution rate
    format_compliance: 0.1     # XML format and action syntax
    screenshot_behavior: 0.05  # Taking screenshot first
    thinking_quality: 0.05     # Concise, quality thinking
```

## Available Tools

Agents interact with the environment through the system prompt, which defines available actions. These actions are mapped to the MCP computer tool provided by HUD.

### Computer Tool Actions
The following actions are defined in the system prompt and mapped to the HUD Computer Tool:

- **`screenshot()`** - Capture the current screen (maps to `computer` tool with `action: "screenshot"`)
- **`click(x, y)`** - Click at coordinates (maps to `computer` tool with `action: "click"`)
- **`type("text")`** - Type text at cursor position (maps to `computer` tool with `action: "type"`)
- **`key("key")`** - Press a key or combination (maps to `computer` tool with `action: "press"`)
  - Examples: `key("enter")`, `key("ctrl+a")`, `key("escape")`
- **`scroll("direction", amount)`** - Scroll the screen (maps to `computer` tool with `action: "scroll"`)
  - Direction: "up" or "down"
- **`wait(seconds)`** - Wait for specified time (maps to `computer` tool with `action: "wait"`)
- **`done()`** - Signal task completion (special action, not sent to MCP)

### Tool Format
Agents must use XML format for tool calls as specified in the system prompt:

```xml
<think>First, I need to see the current screen</think>
<tool>screenshot()</tool>
```

The action mappings in `configs/default.yaml` translate these agent actions to the MCP computer tool format that HUD expects.

## Dataset Format

HUD Gym uses HuggingFace datasets with hud.TaskConfig format:

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

## Reward Functions

The rubric system combines multiple reward functions:

1. **Task Completion** (75%) - Primary reward from HUD evaluation tool
2. **Tool Execution** (5%) - Success rate of tool calls
3. **Format Compliance** (10%) - Proper XML format and action syntax
4. **Screenshot Behavior** (5%) - Taking screenshot as first action
5. **Thinking Quality** (5%) - Quality of reasoning in `<think>` tags

## Custom Environments

Extend HUD Gym for custom tasks:

```python
from hud_vf_gym import HUDGym

class CustomGym(HUDGym):
    def setup_state(self, state, **kwargs):
        state = super().setup_state(state, **kwargs)
        # Add custom state tracking
        state["custom_metric"] = 0
        return state
    
    def env_response(self, messages, state, **kwargs):
        # Custom response logic
        return super().env_response(messages, state, **kwargs)
```

## Development

### Running Tests
```bash
# Run pre-commit hooks
uv run pre-commit run --all-files

# Test with a single task
vf-eval --env hud_vf_gym --num-tasks 1 --model gpt-4o-mini
```

### Adding New Actions

To add new computer tool actions:

1. Update the system prompt to describe the new action to agents

2. Add action mapping in `configs/default.yaml` to map to HUD Computer Tool:
```yaml
action_mappings:
  new_action:
    action: "computer_tool_action"  # The actual MCP computer tool action
    param:
      from_arg: "agent_param"        # Map agent's parameter
      transform: "optional_transform" # Optional transformation
```

3. Update parser in `parsers.py` to parse the new action syntax:
```python
elif action_name == "new_action":
    # Parse new action arguments from agent's call
    return {"name": "new_action", "arguments": {...}}
```

Note: All actions ultimately map to the MCP `computer` tool with different action parameters.

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
2. **XML parsing failures**: Check that agents use proper `<tool>` and `<think>` tags
3. **MCP connection issues**: Verify MCP configuration in dataset
4. **Low rewards**: Review rubric weights and ensure evaluation tool returns grades

### Debug Mode

Enable detailed logging:

```yaml
logging:
  level: "DEBUG"
```

## License

See LICENSE file in the hud_vf_gym directory.