<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg">
    <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 24px;"/>
  </picture>
</div>

OSS RL environment + evals toolkit. Wrap software as environments, run benchmarks, and train with RL – locally or at scale.

[![PyPI version](https://img.shields.io/pypi/v/hud-python?style=flat-square)](https://pypi.org/project/hud-python/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Add docs to Cursor](https://img.shields.io/badge/Add%20docs%20to-Cursor-black?style=flat-square)](https://cursor.com/en/install-mcp?name=docs-hud-python&config=eyJ1cmwiOiJodHRwczovL2RvY3MuaHVkLmFpL21jcCJ9)
[![Discord](https://img.shields.io/discord/1327447144772407390?label=Discord&logo=discord&style=flat-square)](https://discord.gg/wkjtmHYYjm)
[![X Follow](https://img.shields.io/twitter/follow/hud_evals?style=social)](https://x.com/intent/user?screen_name=hud_evals)
[![Shop](https://img.shields.io/badge/_-white.svg?label=shop&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAJCAYAAAAywQxIAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxMAAAsTAQCanBgAAAF6SURBVChTlZA9ixNhFIWf8yaTpFHRRMXCKpAZhCAYFvwoLHZhwUKw9A9YCJb+Bq0sxGbBQrTxX1j41dvIRAjGZbdwRUUGIzPMeyw2swS3WZ/ynHvP5VylafoAWAd+5Xm+wX+SpukmcMf29RDCZrD9BViz3f53+CjYngKZpD5A2/Y7SQBMJpOkKIprdV1vdzqdHzHGblmW9Ww2+5pl2TmAxWKxmM/nP8fj8cmqqtZijJ9sb0u6ABBWjh0riuIt8CqE8LGu66e2d5MkeQ8QY3xme7fb7T4ZjUbrZVl+jjFuSXoEXGxCDgIl9WzfAO5LSmzvNB771R6vzG4Bx0MIt/M8vwV8aLyDQNt70+n0G1AspaTxVln+aghQluVsKbvxVysflT9NQK/XO7R/SGiQ9Nt2aftElmWXJd1kv0kbeANQVdWl4XB4XtJouXaqNRgMHkrqS+r0+/3XwD1JXdungRfAVWBi+6WkK8D3EMJz22cl3W21WgNgx3YAzvwFd0Chdq03gKUAAAAASUVORK5CYII=&style=social)](https://shop.hud.ai)
[![Scarf](https://static.scarf.sh/a.png?x-pxid=6530ff33-4945-452b-81f9-626872593933)](https://scarf.sh)


### Are you an enterprise building agents?

[📅 Hop on a call](https://cal.com/jay-hud) or [📧 founders@hud.ai](mailto:founders@hud.ai)

## Highlights

- 🚀 **[MCP environment skeleton](https://docs.hud.ai/core-concepts/mcp-protocol)** – any agent can call any environment.
- ⚡️ **[Live telemetry](https://hud.ai)** – inspect every tool call, observation, and reward in real time.
- 🗂️ **[Public benchmarks](https://hud.ai/leaderboards)** – OSWorld-Verified, SheetBench-50, and more.
- 🌐 **[Cloud browsers](environments/remote_browser/)** – AnchorBrowser, Steel, BrowserBase integrations for browser automation.
- 🛠️ **[Hot-reload dev loop](environments/README.md#phase-5-hot-reload-development-with-cursor-agent)** – `hud dev` for iterating on environments without rebuilds.
- 🎓 **[One-click RL](https://hud.ai/models)** – Run `hud rl` to get a trained model on any environment.

> We welcome contributors and feature requests – open an issue or hop on a call to discuss improvements!

## Installation

```bash
# SDK - MCP servers, telemetry, evaluation
pip install hud-python

# CLI - RL pipeline, environment design
uv tool install hud-python@latest --python 3.12
# uv tool update-shell
```

> See [docs.hud.ai](https://docs.hud.ai), or add docs to any MCP client:
> `claude mcp add --transport http docs-hud https://docs.hud.ai/mcp`

Before starting, get your HUD_API_KEY at [hud.ai](https://hud.ai).


## Quickstart: Evals

For a tutorial that explains the agent and evaluation design, run:

```python
uvx hud-python quickstart
```

Or just write your own agent loop (more [examples here](examples/)).

```python
import asyncio, hud, os
from hud.settings import settings
from hud.clients import MCPClient
from hud.agents import ClaudeAgent
from hud.datasets import Task  # See docs: https://docs.hud.ai/reference/tasks

async def main() -> None:
    with hud.trace("Quick Start 2048"): # All telemetry works for any MCP-based agent (see https://hud.ai)
        task = {
            "prompt": "Reach 64 in 2048.",
            "mcp_config": {
                "hud": {
                    "url": "https://mcp.hud.ai/v3/mcp",  # HUD's cloud MCP server (see https://docs.hud.ai/core-concepts/architecture)
                    "headers": {
                        "Authorization": f"Bearer {settings.api_key}",  # Get your key at https://hud.ai
                        "Mcp-Image": "hudpython/hud-text-2048:v1.2"  # Docker image from https://hub.docker.com/u/hudpython
                    }
                }
            },
            "evaluate_tool": {"name": "evaluate", "arguments": {"name": "max_number", "arguments": {"target": 64}}},
        }
        task = Task(**task)

        # 1. Define the client explicitly:
        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-sonnet-4-5",  # requires ANTHROPIC_API_KEY
        )

        result = await agent.run(task)

        # 2. Or just:
        # result = await ClaudeAgent().run(task)

        print(f"Reward: {result.reward}")
        await client.shutdown()

asyncio.run(main())
```

The above example let's the agent play 2048 ([See replay](https://hud.ai/trace/6feed7bd-5f67-4d66-b77f-eb1e3164604f))

![Agent playing 2048](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/2048_1.gif)

## Quickstart: Training

RL using GRPO a Qwen2.5-VL model on any hud dataset:

```bash
hud get hud-evals/2048-basic # from HF
hud rl 2048-basic.json
```

> See [agent training docs](https://docs.hud.ai/train-agents/quickstart)

Or make your own environment and dataset:

```bash
hud init my-env && cd my-env
hud dev --interactive
# When ready to run:
hud rl
```

> See [environment design docs](https://docs.hud.ai/build-environments)

## Benchmarking Agents

This is Claude Computer Use running on our proprietary financial analyst benchmark [SheetBench-50](https://huggingface.co/datasets/hud-evals/SheetBench-50):

![Trace screenshot](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/trace_sheet.gif)

> [See this trace on _hud.ai_](https://hud.ai/trace/9e212e9e-3627-4f1f-9eb5-c6d03c59070a)

This example runs the full dataset (only takes ~20 minutes) using [run_evaluation.py](examples/run_evaluation.py):

```bash
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --agent claude
```

Or in code:

```python
import asyncio
from hud.datasets import run_dataset
from hud.agents import ClaudeAgent

results = await run_dataset(
    name="My SheetBench-50 Evaluation",
    dataset="hud-evals/SheetBench-50",      # <-- HuggingFace dataset
    agent_class=ClaudeAgent,                # <-- Your custom agent can replace this (see https://docs.hud.ai/evaluate-agents/create-agents)
    agent_config={"model": "claude-sonnet-4-5"},
    max_concurrent=50,
    max_steps=30,
)
print(f"Average reward: {sum(r.reward for r in results) / len(results):.2f}")
```

> Running a dataset creates a job and streams results to the [hud.ai](https://hud.ai) platform for analysis and [leaderboard submission](https://docs.hud.ai/evaluate-agents/leaderboards).

## Building Environments (MCP)

This is how you can make any environment into an interactable one in 5 steps:

1. Define MCP server layer using [`MCPServer`](https://docs.hud.ai/reference/environments)

```python
from hud.server import MCPServer
from hud.tools import HudComputerTool

mcp = MCPServer("My Environment")

# Add hud tools (see all tools: https://docs.hud.ai/reference/tools)
mcp.tool(HudComputerTool())

# Or custom tools (see https://docs.hud.ai/build-environments/adapting-software)
@mcp.tool("launch_app"):
def launch_app(name: str = "Gmail")
...

if __name__ == "__main__":
    mcp.run()
```

2. Write a simple Dockerfile that installs packages and runs:

```python
CMD ["python", "-m", "hud_controller.server"]
```

And build the image:

```bash
hud build # runs docker build under the hood
```

Or run it in interactible mode

```bash
hud dev
```

3. Debug it with the CLI to see if it launches:

```console
$ hud debug my-name/my-environment:latest

✓ Phase 1: Docker image exists
✓ Phase 2: MCP server responds to initialize 
✓ Phase 3: Tools are discoverable
✓ Phase 4: Basic tool execution works
✓ Phase 5: Parallel performance is good

Progress: [█████████████████████] 5/5 phases (100%)
✅ All phases completed successfully!
```

Analyze it to see if all tools appear:

```console
$ hud analyze hudpython/hud-remote-browser:latest
⠏ ✓ Analysis complete
...
Tools
├── Regular Tools
│   ├── computer
│   │   └── Control computer with mouse, keyboard, and screenshots
...
└── Hub Tools
    ├── setup
    │   ├── navigate_to_url
    │   ├── set_cookies
    │   ├── ...
    └── evaluate
        ├── url_match
        ├── page_contains
        ├── cookie_exists
        ├── ...

📡 Telemetry Data
 Live URL  https://live.anchorbrowser.io?sessionId=abc123def456
```

4. When the tests pass, push it up to the docker registry:

```bash
hud push # needs docker login, hud api key
```

5. Now you can use `mcp.hud.ai` to launch 100s of instances of this environment in parallel with any agent, and see everything live on [hud.ai](https://hud.ai):

```python
from hud.agents import ClaudeAgent

result = await ClaudeAgent().run({  # See all agents: https://docs.hud.ai/reference/agents
    "prompt": "Please explore this environment",
    "mcp_config": {
        "my-environment": {
            "url": "https://mcp.hud.ai/v3/mcp",
            "headers": {
                "Authorization": f"Bearer {os.getenv('HUD_API_KEY')}",
                "Mcp-Image": "my-name/my-environment:latest"
            }
        }
        # "my-environment": { # or use hud run which wraps local and remote running
        #     "cmd": "hud",
        #     "args": [
        #         "run",
        #         "my-name/my-environment:latest",
        #     ]
        # }
    }
})

```

> See the full environment design guide and common pitfalls in [`environments/README.md`](environments/README.md)

## Leaderboards & benchmarks

All leaderboards are publicly available on [hud.ai/leaderboards](https://hud.ai/leaderboards) (see [docs](https://docs.hud.ai/evaluate-agents/leaderboards))

![Leaderboard](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/leaderboards_3.png)

We highly suggest running 3-5 evaluations per dataset for the most consistent results across multiple jobs.

Using the [`run_dataset`](https://docs.hud.ai/reference/tasks#run_dataset) function with a HuggingFace dataset automatically assigns your job to that leaderboard page, and allows you to create a scorecard out of it:

## Reinforcement Learning with GRPO

This is a Qwen‑2.5‑VL‑3B agent training a policy on the 2048-basic browser environment:

![RL curve](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/rl_2.png)

Train with the new interactive `hud rl` flow:

```bash
# Install CLI
uv tool install hud-python@latest --python 3.12

# Option A: Run directly from a HuggingFace dataset
hud rl hud-evals/2048-basic

# Option B: Download first, modify, then train
hud get hud-evals/2048-basic
hud rl 2048-basic.json

# Optional: baseline evaluation
hud eval 2048-basic.json
```

Supports multi‑turn RL for both:
- Language‑only models (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- Vision‑Language models (e.g., `Qwen/Qwen2.5-VL-3B-Instruct`)

By default, `hud rl` provisions a persistent server and trainer in the cloud, streams telemetry to `hud.ai`, and lets you monitor/manage models at `hud.ai/models`. Use `--local` to run entirely on your machines (typically 2+ GPUs: one for vLLM, the rest for training).

Any HUD MCP environment and evaluation works with our RL pipeline (including remote configurations). See the guided docs: `https://docs.hud.ai/train-agents/quickstart`.

Pricing: Hosted vLLM and training GPU rates are listed in the [Training Quickstart → Pricing](https://docs.hud.ai/train-agents/quickstart#pricing). Manage billing at the [HUD billing dashboard](https://hud.ai/project/billing).

## Architecture

```mermaid
%%{init: {"theme": "neutral", "themeVariables": {"fontSize": "14px"}} }%%
graph LR
    subgraph "Platform"
        Dashboard["📊 hud.ai"]
        API["🔌 mcp.hud.ai"]
    end
  
    subgraph "hud"
        Agent["🤖 Agent"]
        Task["📋 Task"]
        SDK["📦 SDK"]
    end
  
    subgraph "Environments"
        LocalEnv["🖥️ Local Docker<br/>(Development)"]
        RemoteEnv["☁️ Remote Docker<br/>(100s Parallel)"]
    end
  
    subgraph "otel"
        Trace["📡 Traces & Metrics"]
    end
  
    Dataset["📚 Dataset<br/>(HuggingFace)"]
  
    AnyMCP["🔗 Any MCP Client<br/>(Cursor, Claude, Custom)"]
  
    Agent <--> SDK
    Task --> SDK
    Dataset <-.-> Task
    SDK <-->|"MCP"| LocalEnv
    SDK <-->|"MCP"| API
    API  <-->|"MCP"| RemoteEnv
    SDK  --> Trace
    Trace --> Dashboard
    AnyMCP -->|"MCP"| API
  
```

## CLI reference

| Command                 | Purpose                                    | Docs |
| ----------------------- | ------------------------------------------ | ---- |
| [`hud init`](https://docs.hud.ai/reference/cli/init)            | Create new environment with boilerplate.  | [📖](https://docs.hud.ai/reference/cli/init) |
| [`hud dev`](https://docs.hud.ai/reference/cli/dev)              | Hot-reload development with Docker.        | [📖](https://docs.hud.ai/reference/cli/dev) |
| [`hud build`](https://docs.hud.ai/reference/cli/build)          | Build image and generate lock file.       | [📖](https://docs.hud.ai/reference/cli/build) |
| [`hud push`](https://docs.hud.ai/reference/cli/push)            | Share environment to registry.            | [📖](https://docs.hud.ai/reference/cli/push) |
| [`hud pull <target>`](https://docs.hud.ai/reference/cli/pull)   | Get environment from registry.            | [📖](https://docs.hud.ai/reference/cli/pull) |
| [`hud analyze <image>`](https://docs.hud.ai/reference/cli/analyze) | Discover tools, resources, and metadata.   | [📖](https://docs.hud.ai/reference/cli/analyze) |
| [`hud debug <image>`](https://docs.hud.ai/reference/cli/debug)   | Five-phase health check of an environment. | [📖](https://docs.hud.ai/reference/cli/debug) |
| [`hud run <image>`](https://docs.hud.ai/reference/cli/run)       | Run MCP server locally or remotely.       | [📖](https://docs.hud.ai/reference/cli/run) |

## Roadmap

- Merging our forks in to the main `mcp`, `mcp_use` repositories
- Helpers for building new environments (see [current guide](environments/README.md))
- Integrations with every major agent framework
- Evaluation environment registry
- MCP opentelemetry standard

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas:
- [Environment examples](environments/) - Add new MCP environments
- [Agent implementations](hud/agents/) - Add support for new LLM providers
- [Tool library](hud/tools/) - Extend the built-in tool collection
- [RL training](hud/rl/) - Improve reinforcement learning pipelines

Thanks to all our contributors!

<a href="https://github.com/hud-evals/hud-python/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hud-evals/hud-python&max=50" />
</a>

## Citation

```bibtex
@software{hud2025agentevalplatform,
  author = {HUD and Jay Ram and Lorenss Martinsons and Parth Patel and Govind Pimpale and Dylan Bowman and Jaideep and Nguyen Nhat Minh},
  title  = {HUD: An Evaluation and RL Envrionments Platform for Agents},
  date   = {2025-04},
  url    = {https://github.com/hud-evals/hud-python},
  langid = {en}
}
```

> **License**: HUD is released under the MIT License – see the [LICENSE](LICENSE) file for details.
