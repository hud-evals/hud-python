"""Templates for hud init command."""

DOCKERFILE_HUD = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
COPY . .

# Serve the Environment's control channel (tcp JSON-RPC) on 8765.
EXPOSE 8765
CMD ["uv", "run", "python", "-m", "hud", "dev", "env:env", "--port", "8765"]
"""

# fmt: off
ENV_PY = '''\
"""{env_name} - HUD Environment"""

import asyncio

from hud.environment import Environment

env = Environment(name="{env_name}")


# =============================================================================
# 1. TASKS - a prompt for the agent, then how to score its answer
# =============================================================================

@env.task(id="count")
async def count(sentence: str, letter: str):
    """Agent must count a letter; we check if it got the answer right."""
    # Yield the prompt, receive the agent's final answer back via ``asend``.
    answer = yield f"How many times does '{{letter}}' appear in: '{{sentence}}'?"

    # Score: 1.0 if correct, else 0.0.
    correct = str(sentence.lower().count(letter.lower()))
    yield 1.0 if correct in (answer or "") else 0.0


# =============================================================================
# 2. CAPABILITIES (optional) - give the agent a way to act
# =============================================================================
# Capabilities are how the agent interacts with the environment. For shell
# access, expose an SSH capability (a sandboxed Workspace) — the agent drives
# bash over SSH, no in-process "bash tool" required. Declare it at create time;
# @env.initialize only starts the daemon:
#
#   from hud.environment import Workspace
#
#   ws = Workspace("/workspace")          # bwrap-isolated SSH + SFTP (binds at create)
#   env = Environment(name="{env_name}", capabilities=[ws.capability()])
#
#   @env.initialize
#   async def _serve_shell():
#       await ws.start()
#
# For arbitrary MCP tools, run them on your own MCPServer and attach it:
#
#   from hud.server import MCPServer
#   from hud.native.tools import JupyterTool
#   server = MCPServer(name="{env_name}-tools")
#   server.add_tool(JupyterTool())
#   env.add_capability(Capability.mcp(name="tools", url="http://127.0.0.1:8765/mcp"))


# =============================================================================
# TEST - run with: python env.py
# =============================================================================

async def test():
    from hud.agents.claude import ClaudeAgent

    agent = ClaudeAgent()

    # Calling a task binds a runnable Task; entering it launches the env.
    async with count(sentence="Strawberry world", letter="r") as run:
        await agent(run)          # fills run.trace; answer is run.trace.content

    print("reward:", run.reward)


if __name__ == "__main__":
    asyncio.run(test())


# =============================================================================
# RUN AT SCALE
# =============================================================================
# Group many parameterizations into a Taskset and evaluate one (stateless) agent
# across them, with optional GRPO-style grouping + a concurrency cap:
#
#   from hud.eval import Taskset
#   from hud.agents.claude import ClaudeAgent
#
#   ts = Taskset.from_tasks(
#       "letters",
#       [count(sentence=s, letter="r") for s in ["strawberry", "raspberry"]],
#   )
#   job = await ts.run(ClaudeAgent(), group=4, max_concurrent=8)
'''
# fmt: on

TASKS_PY = '''\
"""Tasks for {env_name} — run with: hud eval tasks.py <agent>   (e.g. claude)."""

from env import count

# ``hud eval`` collects these Tasks — each is the ``count`` task bound to
# concrete args. Add your own, or build them in a loop.
tasks = [
    count(sentence="Strawberry world", letter="r"),
    count(sentence="banana", letter="a"),
]
'''

PYPROJECT_TOML = """\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["hud-python", "openai"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
