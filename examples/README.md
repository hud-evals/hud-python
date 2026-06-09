# Examples

A collection of examples demonstrating HUD SDK usage patterns.

## Quick Start

### 00_agent_env.py
Minimal environment and agent in one file. Shows the `Task` lifecycle: define a task,
enter it to get a `Run`, let an agent fill the trace, and read the reward.

```bash
uv run examples/00_agent_env.py
```

## Coding Agents

### 01_codex_coding_agent.py
Build your own Codex - a 1:1 recreation of OpenAI's Codex CLI using HUD's `ShellTool` and `ApplyPatchTool`. Supports local mode (tools run on your machine) and hub mode (sandboxed cloud execution with full telemetry).

```bash
# Local mode - just like running `codex` on your machine
uv run python examples/01_codex_coding_agent.py --local

# Hub mode - sandboxed cloud execution
uv run python examples/01_codex_coding_agent.py

# Custom task
uv run python examples/01_codex_coding_agent.py --local \
  --task "Create a Python script that prints the Fibonacci sequence"
```

> Requires `HUD_API_KEY`. Uses HUD Gateway for inference.

## Key Concepts

### Tasks, tasksets, jobs

Create concrete tasks by calling an `@env.task` function. Group tasks into a
`Taskset` when you want to evaluate a batch:

```python
from hud import Taskset

taskset = Taskset.from_tasks("my-eval", [count_letter(word="strawberry")])
job = await taskset.run(agent)
print(job.runs[0].reward)
```

Each `Run` owns the agent trace and grade result.
