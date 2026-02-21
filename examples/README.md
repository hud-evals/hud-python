# Examples

A collection of examples demonstrating HUD SDK usage patterns.

## Quick Start

### 00_agent_env.py
Minimal MCP server and client in one file. Shows the basic agent-environment communication pattern using `hud.eval()`.

```bash
python examples/00_agent_env.py
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

### 02_opencode_agent.py
OpenCode-style coding agent with `EditTool`, `ShellTool`, and filesystem exploration tools (`ReadTool`, `GrepTool`, `GlobTool`, `ListTool`). Includes a read-only plan mode for safe codebase exploration.

```bash
# Build mode - full coding capabilities
uv run python examples/02_opencode_agent.py --task "Fix the bug in main.py"

# Plan mode - read-only exploration
uv run python examples/02_opencode_agent.py --plan --task "How does auth work?"
```

> Requires `HUD_API_KEY`. Works with any model via `--model`.

## Key Concepts

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
