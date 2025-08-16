<div align="left">
  <img src="https://raw.githubusercontent.com/hud-evals/hud-python/main/docs/logo/hud_logo.svg" alt="HUD" width="150" style="margin-bottom: 20px;"/>
</div>

Evaluate and improve agent–environment systems. Wrap software as environments, define measurable evaluations, and train with RL – locally or at scale.

### Are you a startup building agents?

[📅 Hop on a call](https://cal.com/jay-ram-z6st6w/demo) or [📧 founders@hud.so](mailto:founders@hud.so)

We're here to help with eval strategies, custom environments, or improving your agent via RL!

[![PyPI version](https://img.shields.io/pypi/v/hud-python)](https://pypi.org/project/hud-python/)

## What we help you do

- Environments: turn any app (browser, game, API) into a controllable environment with tools
- Evaluations: define setup/evaluate logic and run benchmarks (e.g., SheetBench)
- RL: train and evaluate via HUD’s Verifier-based gym (hud-vf-gym)

```text
Agent (LLM/Policy)  ── tool calls (MCP JSON-RPC) ──►  Environment (your software)
                                   │                        ▲
                                   ▼                        │
                           HUD Client/SDK           Tools you expose (setup/evaluate/act)
```

MCP (Model-Context Protocol) is the wire format we use to send tool calls. You rarely “see” MCP directly – it’s just the protocol that lets any agent talk to any environment, locally or in Docker or in the cloud.

---

## 1) Environments

Wrap your software as an environment by exposing a small set of tools. We use FastMCP to register tool functions and (usually) run them in Docker so they’re reproducible and scalable.

Minimal call example (MCP client calling a local Docker image):

```python
import asyncio
import hud
from hud.client import MCPClient

async def main():
    with hud.trace("env-quickstart"):
        client = MCPClient(mcp_config={
            "local": {
                "command": "docker",
                "args": ["run","--rm","-i","hud-text-2048"]
            }
        })
        # Setup 2048 board, make a move, then evaluate
        await client.call_tool("setup", {"name": "board", "arguments": {"board_size": 4}})
        await client.call_tool("move", {"direction": "up"})
        result = await client.call_tool("evaluate", {"name": "max_number", "arguments": {"target": 64}})
        print(result)
        await client.close()

asyncio.run(main())
```

Start here:
- environments/README.md – how to build an environment (with BaseHub + tools)
- examples/01_hello_2048.py – end-to-end hello world using the 2048 environment
- examples/task_with_setup_eval.py – minimal setup/evaluate without an LLM

---

## 2) Evaluations

Evaluations turn tasks into measurable outcomes. Your `evaluate` tool returns a reward (0–1), a done flag, and optional info.

Return shapes:

```python
from hud.tools.types import SetupResult, EvaluationResult

SetupResult(content="Seeded 5 items", info={"items": 5})
EvaluationResult(reward=0.8, done=False, content="Found 8/10 entries")
```

Recommended starting points:
- examples/sheet_bench.py – SheetBench demo using a dataset → TaskConfig → agent run
- environments/remote_browser – rich setup/evaluate hubs (navigate, cookies, page checks)

Run with Claude (quick demo):

```python
import asyncio
import hud
from datasets import load_dataset
from hud.datasets import to_taskconfigs
from hud.client import MCPClient
from hud.agents import ClaudeAgent

async def main():
    with hud.trace("sheetbench-claude-demo"):
        # Load a small SheetBench split from HF and convert to TaskConfig
        dataset = load_dataset("hud-evals/SheetBench-50", split="train")
        task = to_taskconfigs(dataset)[0]

        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )
        try:
            print(task.prompt)
            result = await agent.run(task, max_steps=15)
            print(result.reward)
        finally:
            await client.close()

asyncio.run(main())
```

Tip: use `with hud.trace("run-name"):` to send traces to app.hud.so for debugging.

---

## 3) RL (hud-vf-gym + Verifiers)

HUD integrates with the Verifiers framework via `hud-vf-gym` (config-driven). A YAML config defines the agent-facing tools and maps them to MCP tools (action_mappings). You can run evaluations with the Verifiers CLI and train with GRPO.

Run evals (CLI):

```bash
vf-eval hud_vf_gym \
  --model gpt-4.1-mini \
  --env-args '{"taskset":"hud-evals/gmail-taskset","config_path":"./configs/default.yaml"}' \
  --num-examples 5
```

Train with GRPO (programmatic):

```python
from verifiers.trainers import GRPOTrainer, GRPOConfig
from hud_vf_gym import load_environment

env = load_environment(taskset="hud-evals/gmail-taskset", config_path="./configs/default.yaml")
cfg = GRPOConfig(model_name_or_path="your-model", num_train_epochs=3, per_device_train_batch_size=4)
trainer = GRPOTrainer(model=model, env=env, args=cfg, processing_class=tokenizer)
trainer.train()
```

Action mapping (YAML):

```yaml
action_mappings:
  screenshot:
    _parser: { positional: [] }
    _tool: "computer"
    action: "screenshot"
  key:
    _parser: { positional: ["key"] }
    _tool: "computer"
    action: "press"
    keys: { from_arg: "key", transform: "lambda x: x.split('+')" }
```

Learn more:
- rl/README.md – overview of RL frameworks (ART and Verifiers)
- rl/verifiers/README.md – how hud-vf-gym + Verifiers work, configs and datasets
- rl/verifiers/configs/default.yaml (browser/computer), rl/verifiers/configs/2048.yaml (2048)

---

## Tooling & telemetry

### CLI Tools (`hud`)

**What's the difference?**

- **`hud debug <IMAGE>`** – 5-phase checker (startup → MCP handshake → tools → telemetry → stress).
- **`hud analyze <IMAGE>`** – Explore tools/resources (`--format json|markdown`). Requires debug phase 3.

```bash
hud debug hudpython/hud-remote-browser:latest
hud analyze hudpython/hud-remote-browser:latest
```

### Tracing & Debugging

- **Tracing**: `with hud.trace("run-name"):` - Stream any MCP-based trace to app.hud.so for debugging and visualization

---

## Next steps

- environments/ – build an environment (BaseHub + tools)
- examples/ – hello world, agent lifecycle, SheetBench
- rl/ – RL training with ART and Verifiers frameworks

Install: `pip install hud-python`  •  Issues/feedback welcome.


[MIT License](LICENSE)

```bibtex
@software{hud2025agentevalplatform,
  author = {HUD and Jay Ram and Lorenss Martinsons and Parth Patel and Oskars Putans and Govind Pimpale and Mayank Singamreddy and Nguyen Nhat Minh},
  title = {{HUD: An Evaluation Platform for Agents}},
  date = {2025-04},
  url = {https://github.com/hud-evals/hud-python},
  langid = {en}
}
```