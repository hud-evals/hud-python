<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg">
    <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 24px;"/>
  </picture>
</div>

HUD is a platform for building RL environments for AI agents. Define an environment, write tasks that prompt and grade an agent, run evaluations at scale, and train models on the results.

To learn more, check out our [Documentation](https://docs.hud.ai) and [API Reference](https://docs.hud.ai/reference).

[![PyPI](https://img.shields.io/pypi/v/hud-python?style=flat-square)](https://pypi.org/project/hud-python/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Add docs to Cursor](https://img.shields.io/badge/Add%20docs%20to-Cursor-black?style=flat-square)](https://cursor.com/en/install-mcp?name=docs-hud-python&config=eyJ1cmwiOiJodHRwczovL2RvY3MuaHVkLmFpL21jcCJ9)
[![Discord](https://img.shields.io/discord/1327447144772407390?label=Discord&logo=discord&style=flat-square)](https://discord.gg/wkjtmHYYjm)
[![X Follow](https://img.shields.io/twitter/follow/hud_evals?style=social)](https://x.com/intent/user?screen_name=hud_evals)
[![Scarf](https://static.scarf.sh/a.png?x-pxid=6530ff33-4945-452b-81f9-626872593933)](https://scarf.sh)
[![Docs](https://img.shields.io/badge/docs-hud.ai-blue?style=flat-square)](https://docs.hud.ai)

## Install

```bash
# Install the CLI (recommended)
uv tool install hud-python --python 3.12

# …or as a library
pip install hud-python
```

Get your API key at [hud.ai/project/api-keys](https://hud.ai/project/api-keys) and set it:

```bash
export HUD_API_KEY=your-key-here
```

![Agent running on SheetBench](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/trace_sheet.gif)

## Environments

An environment is the harness an agent operates in. It declares **capabilities** (how the agent acts — shell, browser, MCP tools) and **tasks** (how the agent is prompted and graded). Each evaluation spins up a fresh, isolated instance.

```python
from hud.environment import Environment

env = Environment(name="my-env")

@env.task()
async def count(word: str, letter: str):
    # PROMPT — the agent runs its reasoning loop and sends back an answer.
    answer = yield f"How many '{letter}' in '{word}'?"

    # SCORE — return a reward (0.0–1.0).
    correct = str(word.lower().count(letter.lower()))
    yield 1.0 if answer and correct in answer else 0.0
```

A task has two yields. The first sends a prompt — the agent works between the yields, reasoning and calling tools. The second checks the answer and returns a reward. → [Core Concepts](https://docs.hud.ai/concepts)

## Run an Agent

Calling a task binds a **Variant** (a task + its args). Entering it launches the environment and yields a live **Run**; `await agent(run)` drives the agent, filling `run.trace`.

```python
from hud.agents import create_agent

agent = create_agent("claude-sonnet-4-5")

async with count(word="strawberry", letter="r") as run:
    await agent(run)

print(f"Reward: {run.reward}")        # 1.0 if the agent answers "3"
print(run.trace.content)              # the agent's final answer
```

`create_agent()` routes any model (Claude, GPT, Gemini, …) through the HUD gateway and picks the right native tools. Agents are stateless, so one instance can drive many concurrent rollouts. → [Agents](https://docs.hud.ai/quick-links/environments)

## Evaluate at Scale

Group many variants into a **Taskset** and evaluate one agent across them — with optional grouping and a concurrency cap. You get back a `Run` per rollout.

```python
from hud.eval import Taskset

ts = Taskset(count(word=w, letter="r") for w in ["strawberry", "raspberry", "blueberry"])
runs = await ts.run(agent, group=4, max_concurrent=16)

print(sum(r.reward for r in runs) / len(runs))   # mean reward
```

The same `agent(run)` primitive carries you from a single rollout to a full batch — no new concepts. → [Evaluation](https://docs.hud.ai/advanced/testing-environments)

## Workflow (CLI)

The CLI takes an environment from scaffold to deployed evals:

```bash
hud init my-env              # scaffold an environment (env.py + Dockerfile)
cd my-env
hud dev env:env              # serve the environment locally (control channel on :8765)
hud eval tasks.py claude     # run an agent over your tasks locally
hud build                    # build the image + lock (capabilities + tasks)
hud deploy                   # deploy to the platform
hud sync my-taskset          # sync your tasks to the platform
```

Run evals at scale from the [platform UI](https://hud.ai) once deployed.

→ [Deploy](https://docs.hud.ai/quick-links/deploy) · [CLI Reference](https://docs.hud.ai/reference/cli/overview)

## Capabilities & Tools

Agents act through **capabilities** the environment declares. For shell access, expose an SSH capability backed by a sandboxed `Workspace` — the agent drives `bash` over SSH:

```python
from hud.environment import Environment, Workspace

ws = Workspace("/workspace")                                   # bwrap-isolated SSH + SFTP
env = Environment(name="coder", capabilities=[ws.capability()])

@env.initialize
async def _serve_shell():
    await ws.start()                                           # capability declared above
```

For arbitrary MCP tools, register HUD's standalone tools on your own `MCPServer` and attach it as an `mcp` capability:

```python
from hud.server import MCPServer
from hud.native.tools import JupyterTool, MemoryTool, PlaywrightTool

server = MCPServer(name="my-tools")
server.add_tool(JupyterTool())     # also: MemoryTool, PlaywrightTool, BashTool, EditTool
```

→ [Capabilities](https://docs.hud.ai/concepts) · [Tools Reference](https://docs.hud.ai/tools/computer)

## Model Gateway

Use Claude, GPT, Gemini, or Grok through one OpenAI-compatible endpoint:

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=os.environ["HUD_API_KEY"])

response = await client.chat.completions.create(
    model="claude-sonnet-4-5",  # or gpt-4o, gemini-2.5-pro — see https://hud.ai/models
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Every call is traced at [hud.ai](https://hud.ai). → [Models](https://docs.hud.ai/quick-links/models)

## Links

- 📖 [Documentation](https://docs.hud.ai)
- ⌨️ [CLI Reference](https://docs.hud.ai/reference/cli/overview)
- 🏆 [Leaderboards](https://hud.ai/leaderboards)
- 🌐 [Environment Templates](https://hud.ai/environments)
- 🤖 [Supported Models](https://hud.ai/models)
- 💬 [Discord](https://discord.gg/wkjtmHYYjm)

## Enterprise

Building agents at scale? We work with teams on custom environments, benchmarks, and training.

[📅 Book a call](https://cal.com/jay-hud) · [📧 founders@hud.ai](mailto:founders@hud.ai)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

Key areas: [Agents](hud/agents/) · [Environments](hud/environment/) · [Native Tools](hud/native/tools/)

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

MIT License · [LICENSE](LICENSE)
