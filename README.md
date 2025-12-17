<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg">
    <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 24px;"/>
  </picture>
</div>

The HUD SDK is an open-source Python toolkit for building, evaluating, and training AI agents. Use a unified API for any model provider, wrap your code as MCP environments, run A/B evals at scale, and train with reinforcement learning.

To learn more, check out our [Documentation](https://docs.hud.ai) and [API Reference](https://docs.hud.ai/reference).

[![PyPI](https://img.shields.io/pypi/v/hud-python?style=flat-square)](https://pypi.org/project/hud-python/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Add docs to Cursor](https://img.shields.io/badge/Add%20docs%20to-Cursor-black?style=flat-square)](https://cursor.com/en/install-mcp?name=docs-hud-python&config=eyJ1cmwiOiJodHRwczovL2RvY3MuaHVkLmFpL21jcCJ9)
[![Discord](https://img.shields.io/discord/1327447144772407390?label=Discord&logo=discord&style=flat-square)](https://discord.gg/wkjtmHYYjm)
[![X Follow](https://img.shields.io/twitter/follow/hud_evals?style=social)](https://x.com/intent/user?screen_name=hud_evals)
[![Shop](https://img.shields.io/badge/_-white.svg?label=shop&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAJCAYAAAAywQxIAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxMAAAsTAQCanBgAAAF6SURBVChTlZA9ixNhFIWf8yaTpFHRRMXCKpAZhCAYFvwoLHZhwUKw9A9YCJb+Bq0sxGbBQrTxX1j41dvIRAjGZbdwRUUGIzPMeyw2swS3WZ/ynHvP5VylafoAWAd+5Xm+wX+SpukmcMf29RDCZrD9BViz3f53+CjYngKZpD5A2/Y7SQBMJpOkKIprdV1vdzqdHzHGblmW9Ww2+5pl2TmAxWKxmM/nP8fj8cmqqtZijJ9sb0u6ABBWjh0riuIt8CqE8LGu66e2d5MkeQ8QY3xme7fb7T4ZjUbrZVl+jjFuSXoEXGxCDgIl9WzfAO5LSmzvNB771R6vzG4Bx0MIt/M8vwV8aLyDQNt70+n0G1AspaTxVln+aghQluVsKbvxVysflT9NQK/XO7R/SGiQ9Nt2aftElmWXJd1kv0kbeANQVdWl4XB4XtJouXaqNRgMHkrqS+r0+/3XwD1JXdungRfAVWBi+6WkK8D3EMJz22cl3W21WgNgx3YAzvwFd0Chdq03gKUAAAAASUVORK5CYII=&style=social)](https://shop.hud.ai)
[![Scarf](https://static.scarf.sh/a.png?x-pxid=6530ff33-4945-452b-81f9-626872593933)](https://scarf.sh)
[![Docs](https://img.shields.io/badge/docs-hud.ai-blue?style=flat-square)](https://docs.hud.ai)

## Install

```bash
pip install hud-python
```

Get your API key at [hud.ai](https://hud.ai) and set it:

```bash
export HUD_API_KEY=your-key-here
```

> For CLI tools (`hud init`, `hud dev`, etc.): `uv tool install hud-python --python 3.12`

![Agent running on SheetBench](https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/src/images/trace_sheet.gif)

## Usage

### Unified Model API

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

Every call is traced at [hud.ai](https://hud.ai). → [Docs](https://docs.hud.ai/quick-links/gateway)

### Environments

Turn your code into tools agents can call. Define how to evaluate them:

```python
from hud import Environment

env = Environment("my-env")

@env.tool()
def search(query: str) -> str:
    """Search the knowledge base."""
    return db.search(query)

@env.scenario("find-answer")
async def find_answer(question: str, answer: str):
    response = yield f"Find: {question}"       # Prompt
    yield 1.0 if answer in response else 0.0  # Reward
```

The agent runs between the yields. First yield sends the prompt, second yield scores the result. → [Docs](https://docs.hud.ai/quick-links/environments) · [Templates](https://hud.ai/environments)

### A/B Evals

Test different models. Repeat runs to see the distribution:

```python
import hud

task = env("find-answer", question="What is 2+2?", answer="4")

async with hud.eval(task, variants={"model": ["gpt-4o", "claude-sonnet-4-5"]}, group=5) as ctx:
    response = await client.chat.completions.create(
        model=ctx.variants["model"],
        messages=[{"role": "user", "content": ctx.prompt}]
    )
    await ctx.submit(response.choices[0].message.content)
```

**Variants** test configurations. **Groups** repeat for distribution. Results stream to [hud.ai](https://hud.ai). → [Docs](https://docs.hud.ai/quick-links/ab-testing)

### Deploy & Train

Push to GitHub, connect on hud.ai, run at scale:

```bash
hud init                  # Scaffold environment
git push                  # Push to GitHub
# Connect on hud.ai → New → Environment
hud eval my-org/my-eval --model gpt-4o --group-size 100
# Or create and run tasks on the platform
```

Every run generates training data. Use it to fine-tune or run RL. → [Docs](https://docs.hud.ai/quick-links/deploy)

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
