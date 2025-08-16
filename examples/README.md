# HUD SDK – Examples Overview

These examples show **modern v3 patterns** for building, running and evaluating agents with the HUD SDK.

---

## Quick navigation

| Example | Level | Purpose |
|---------|-------|---------|
| 00_minimal_fastmcp.py | Beginner | Server **and** stdio-client in one file (sum two numbers) |
| 01_hello_2048.py | Beginner | Launch `text_2048` environment (Docker) + Claude agent |
| task_with_setup_eval.py | Beginner | Manual call of `setup` / `evaluate` without LLM |
| 02_agent_lifecycle.py | Intermediate | Save / resume runs, HUD trace & telemetry |
| 03_cloud_vs_local.py | Intermediate | Run the same image locally vs remotely on HUD |
| mcp_use_agent.py | Intermediate | Using the HTTP `mcp_use` client |
| playwright_screenshot.py | Intermediate | Capture a browser screenshot via `remote_browser` |
| rl/hud_vf_gym/ | Advanced | Reinforcement-learning loop with value-function gym |
| sheet_bench*.py | Advanced | Google-sheet style benchmark via remote browser |
| claude_agent.py / openai_agent.py | Quickstart | Direct chat agents (no environment) |

_Notebooks_: exploratory notebooks are optional and live next to the scripts.

Run an example:
```bash
python examples/00_minimal_fastmcp.py
```

---
---

## 2 · Prerequisites

| Requirement | Why |
|-------------|-----|
| **Docker** | Needed for local browser examples. |
| **HUD_API_KEY** | Required for cloud routes. |
| **OPENAI / ANTHROPIC API keys** | Only if you run those LLM agents. |

---

## 3 · Running Visual Examples

```bash
# Build the browser image once
docker pull hudpython/hud-browser:latest

# Start a visual task
python task_with_setup_eval.py
# Open http://localhost:8080/vnc.html to watch the agent
```

---

## 4 · Pattern Cheat-Sheet

```python
from hud.telemetry import trace
from hud.datasets import TaskConfig
from hud.clients import MCPClient
from hud.agents import ClaudeAgent

with trace("My Demo"):
    task = TaskConfig(...)
    client = MCPClient(mcp_config=task.mcp_config)
    agent  = ClaudeAgent(mcp_client=client, ...)
    result = await agent.run(task)
```

*️⃣  Every example follows this structure – once you understand it, you can mix-and-match components to build your own flows.

---

## 5 · Alternative Tracing Backends

Want to use Jaeger or another OpenTelemetry backend instead of HUD's? See `custom_otel_backend.py` for a simple example that sends `hud.trace()` spans to Jaeger.

---

Happy hacking 🚀
