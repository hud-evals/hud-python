# Examples

A collection of examples demonstrating HUD SDK usage patterns.

## Quick Start

### 00_agent_env.py
Minimal MCP server and client in one file. Shows the basic agent-environment communication pattern using `hud.eval()`.

```bash
python examples/00_agent_env.py
```

### 01_agent_lifecycle.py
Complete agent lifecycle demonstrating:
- v5 Task format with Environment and scenario
- `hud.eval()` context for connection and tracing
- Agent initialization and execution
- Automatic scenario setup/evaluation

```bash
python examples/01_agent_lifecycle.py
```

> Requires `HUD_API_KEY` and `ANTHROPIC_API_KEY` environment variables.

## Agent Examples

### 02_claude_agent.py
Claude agent with computer use capabilities for browser automation.

```bash
python examples/02_claude_agent.py
```

> Requires `HUD_API_KEY` and `ANTHROPIC_API_KEY`.

### 03_openai_compatible_agent.py
OpenAI-compatible chat.completions agent with both text and browser 2048 environments.

```bash
export OPENAI_API_KEY=your-key
# export OPENAI_BASE_URL=http://localhost:8000/v1  # for local servers (e.g., vllm)

python examples/03_openai_compatible_agent.py --mode text     # text environment
python examples/03_openai_compatible_agent.py --mode browser  # browser environment
```

> Requires Docker for local environment execution.

### 04_grounded_agent.py
Grounded agent that separates visual grounding (element detection) from high-level reasoning.

```bash
export OPENAI_API_KEY=your-key
export OPENROUTER_API_KEY=your-key

python examples/04_grounded_agent.py
```

> Requires Docker and API keys for both OpenAI and OpenRouter.

### 05_custom_agent.py
Build a custom MCPAgent using HUD Gateway for unified model access:
- No need for individual provider API keys
- Works with Anthropic, OpenAI, Gemini, OpenRouter models
- Automatic tracing with `@hud.instrument`

```bash
HUD_API_KEY=sk-hud-... python examples/05_custom_agent.py
```

## Dataset Evaluation

### run_evaluation.py
Generic dataset evaluation runner using the programmatic API.

```bash
# Run all tasks in a dataset
python examples/run_evaluation.py hud-evals/SheetBench-50

# Run specific tasks by index
python examples/run_evaluation.py hud-evals/SheetBench-50 --task-ids 0 1 2

# Use different agent and concurrency
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent operator --max-concurrent 50
```

For production evaluations, prefer the CLI: `hud eval --help`

## Key Concepts

### v5 Task Format

The v5 Task format is the recommended way to define evaluation tasks:

```python
from hud.eval.task import Task

# Simple task with hub environment
task = Task(
    env={"name": "browser"},  # Connect to browser hub
    scenario="checkout",       # Scenario to run
    args={"user_id": "alice"}, # Scenario arguments
)

# Task with local Docker environment
env = hud.Environment("my-env")
env.connect_local(command="docker", args=["run", "--rm", "-i", "my-image"])
task = Task(env=env, scenario="test")
```

### Using hud.eval()

All examples use `hud.eval()` as the primary entry point:

```python
async with hud.eval(task, name="my-eval", variants={"model": "gpt-4o"}) as ctx:
    result = await agent.run(ctx, max_steps=10)
    print(f"Reward: {ctx.reward}")
```

The context manager handles:
- Environment connection (MCP servers start)
- Scenario setup execution
- Telemetry and tracing
- Automatic scenario evaluation on exit
