<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg">
    <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 24px;"/>
  </picture>
</div>

HUD is a platform for building RL environments for AI agents. Define agent-callable tools, write evaluation scenarios, run evals at scale, and train models on the results.

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
# Install CLI (recommended)
uv tool install hud-python --python 3.12

Get your API key at [hud.ai](https://hud.ai) and set it:

```bash
export HUD_API_KEY=your-key-here
```

Get your API key at [hud.ai/project/api-keys](https://hud.ai/project/api-keys).

> Or install as a library: `pip install hud-python`

![Agent running on SheetBench](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/trace_sheet.gif)

## Environments

An environment is the harness an agent operates in. It packages tools (functions agents can call) and scenarios (how agents are evaluated) into a single deployable unit. Each environment spins up fresh and isolated for every evaluation.

```python
from hud import Environment

env = Environment("my-env")

@env.scenario("count")
async def count(word: str, letter: str):
    # PROMPT — send a question to the agent.
    # The agent runs its reasoning loop and returns an answer.
    answer = yield f"How many '{letter}' in '{word}'?"

    # SCORE — check the agent's answer against the correct count.
    # Return a reward: 1.0 for correct, 0.0 for wrong.
    correct = str(word.lower().count(letter.lower()))
    yield 1.0 if answer and correct in answer else 0.0
```

A scenario has two yields. The first sends a prompt — the agent runs between the yields, calling tools and reasoning. The second checks the result and returns a reward (0.0 to 1.0). → [Core Concepts](https://docs.hud.ai/concepts)

## Run an Agent

```python
import hud
from hud.agents import create_agent

task = env("count", word="strawberry", letter="r")
agent = create_agent("claude-sonnet-4-5")

async with hud.eval(task) as ctx:
    result = await agent.run(ctx)

print(f"Reward: {result.reward}")  # 1.0 if agent answers "3"
```

`create_agent()` picks the right agent class and native tools for each model. → [Environments](https://docs.hud.ai/quick-links/environments)

## Workflow

```bash
hud init my-env          # Scaffold environment
cd my-env
hud dev env:env -w env.py    # Run locally with hot-reload
hud eval tasks.py claude     # Run evals locally
hud deploy                   # Deploy to platform
hud sync tasks my-taskset    # Sync tasks to platform
```

Once deployed, run evals at scale from the CLI or the [platform UI](https://hud.ai):

```bash
hud eval my-taskset claude --remote --full
```

→ [Deploy](https://docs.hud.ai/quick-links/deploy) · [Testing & Evaluation](https://docs.hud.ai/advanced/testing-environments)

## Pre-built Tools

HUD ships tools for computer control, shell execution, file editing, browser automation, and web search. Add them to any environment:

```python
from hud.tools import AnthropicComputerTool, BashTool, EditTool

env.add_tool(AnthropicComputerTool())  # Mouse, keyboard, screenshots
env.add_tool(BashTool())               # Persistent bash shell
env.add_tool(EditTool())               # File viewing and editing
```

HUD adapts each tool to the model's native format — Claude gets `computer_20250124`, OpenAI gets `computer_use_preview`, Gemini gets `ComputerUse`. → [Tools Reference](https://docs.hud.ai/tools/computer)

## Model Gateway

Use Claude, GPT, Gemini, or Grok through one OpenAI-compatible endpoint:

```python
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(
    base_url="https://inference.hud.ai",
    api_key=os.environ["HUD_API_KEY"]
)

response = await client.chat.completions.create(
    model="claude-sonnet-4-5",  # or gpt-4o, gemini-2.5-pro (https://hud.ai/models)
    messages=[{"role": "user", "content": "Hello!"}]
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

Key areas: [Agents](hud/agents/) · [Tools](hud/tools/) · [Environments](https://hud.ai/environments)

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
