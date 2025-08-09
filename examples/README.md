# HUD SDK – Examples Overview

These examples show **modern v3 patterns** for building, running and evaluating agents with the HUD SDK.

---

## 1 · Quick Start

```bash
# Install core SDK (+extras for examples)
pip install "hud-python[examples]"

# Set your cloud key (skip for local-only)
export HUD_API_KEY=your_api_key

# Build the 2048 game image
docker build -t hud-text-2048 environments/text_2048

# Run the tiniest demo
python 01_hello_2048.py
```

---

## 2 · What You’ll Find

| File / Notebook | What it Proves | Notes |
|-----------------|----------------|-------|
| **01_hello_2048.py** | Local Docker game + agent | Simple 2048 game with text interface.
| **02_agent_lifecycle.py** | *Full* task lifecycle <br>setup → agent loop → evaluate | Wrapped in `hud.trace()` – generates RUN_ID.
| **03_cloud_vs_local.py** | Same agent, two deployments | Compare HUD cloud vs local Docker.
| **claude_agent.py** | Claude-3.7, screenshots & reasoning | Uses *anthropic_computer* tool.
| **openai_agent.py** | GPT-4o with function calling | Uses *openai_computer* tool.
| **mcp_use_agent.py** | Provider-agnostic via LangChain | Works with any LLM key you have.
| **task_with_setup_eval.py** | Visual todo / Wikipedia tasks | See the browser at `http://localhost:8080/vnc.html`.
| **resources_exploration.py** | List `setup://`, `evaluate://`, etc. | Discover what an MCP server can do.
| **agent_lifecycle_exploration.ipynb** | Step-through debug of an agent | Inspect messages, tool calls, screenshots.
| **dataset_evaluation_pipeline.ipynb** | Run a HF dataset end-to-end | Auto-traced; outputs CSV + MD report.

---

## 3 · Prerequisites

| Requirement | Why |
|-------------|-----|
| **Docker** | Needed for local browser examples. |
| **HUD_API_KEY** | Required for cloud routes. |
| **OPENAI / ANTHROPIC API keys** | Only if you run those LLM agents. |

---

## 4 · Running Visual Examples

```bash
# Build the browser image once
docker pull hudpython/hud-browser:latest

# Start a visual task
python task_with_setup_eval.py
# Open http://localhost:8080/vnc.html to watch the agent
```

---

## 5 · Pattern Cheat-Sheet

```python
from hud.telemetry import trace
from hud.datasets import TaskConfig
from hud.mcp.client import MCPClient
from hud.mcp import ClaudeMCPAgent

with trace("My Demo"):
    task = TaskConfig(...)
    client = MCPClient(mcp_config=task.mcp_config)
    agent  = ClaudeMCPAgent(mcp_client=client, ...)
    result = await agent.run(task)
```

*️⃣  Every example follows this structure – once you understand it, you can mix-and-match components to build your own flows.

---

Happy hacking 🚀
