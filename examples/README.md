# Examples

A collection of examples demonstrating HUD SDK usage patterns.

## Quick Start Examples

### 00_agent_env.py
Minimal MCP server and client in one file. Shows the basic agent-environment communication pattern.

```bash
python examples/00_agent_env.py
```

### 01_hello_2048.py
Complete agent evaluation on the 2048 environment using Claude.

```bash
python examples/01_hello_2048.py
```

> | Requires Docker and `ANTHROPIC_API_KEY` environment variable.

### 03_browser_agent_loop.py
Quick start for the browser environment (Claude). Supports multiple demo apps.

```bash
# 2048 (default)
python examples/03_browser_agent_loop.py

# Todo app
python examples/03_browser_agent_loop.py --app todo
```

> | Requires Docker (exposes port 8080) and `ANTHROPIC_API_KEY`.

## Core Patterns

### 01_agent_lifecycle.py
Demonstrates the full agent lifecycle with telemetry and state management.
- Task creation using LegacyTask format
- Trace context for debugging
- Setup and evaluation tool calls

### run_evaluation.py
Generic dataset evaluation runner using the programmatic API.

```bash
# Run all tasks
python examples/run_evaluation.py hud-evals/SheetBench-50

# Run specific tasks by index
python examples/run_evaluation.py hud-evals/SheetBench-50 --task-ids 0 1 2

# Use different agent
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent operator
```

## Integration Examples

### claude_agent.py
Direct usage of Claude agent without environments.

### integration_mcp_use.py
Using the legacy `mcp_use` client for multi-server setups.

### integration_otel.py
Custom OpenTelemetry backend integration (e.g., Jaeger).

### openai_compatible_agent.py
OpenAI-compatible chat.completions agent with both text and browser 2048 environments.

```bash
export OPENAI_API_KEY=your-key           # or dummy value for local servers
# export OPENAI_BASE_URL=http://localhost:8000/v1  # e.g., vllm
python examples/openai_compatible_agent.py --mode text     # text environment
python examples/openai_compatible_agent.py --mode browser  # browser environment
```
