"""File templates written by ``hud init``."""

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
CMD ["uv", "run", "python", "-m", "hud", "dev", "env:env", "--host", "0.0.0.0", "--port", "8765"]
"""

# fmt: off
ENV_PY = '''\
"""{env_name} - HUD Environment"""

import asyncio
import tempfile
from pathlib import Path

from hud.environment import Environment

env = Environment(name="{env_name}")

# =============================================================================
# 1. WORKSPACE - give the agent a bash shell and file system
# =============================================================================
# The workspace is an isolated directory the agent can read/write over SSH.
# ``network=True`` lets the agent's shell reach the internet (curl, pip, etc.).
# The path is created fresh each run; change it to a fixed path if you need
# to pre-populate files (e.g. a git clone, dataset, or config).

WORKSPACE = Path(tempfile.mkdtemp(prefix="hud-{env_name}-"))
ws = env.workspace(WORKSPACE, network=True)


# =============================================================================
# 2. TASKS - a prompt for the agent, then how to score its answer
# =============================================================================

@env.template(id="count")
async def count(sentence: str, letter: str):
    """Agent must count a letter; we check if it got the answer right."""
    # Yield the prompt, receive the agent\'s final answer back via ``asend``.
    answer = yield f"How many times does \'{{letter}}\' appear in: \'{{sentence}}\'?"

    # Score: 1.0 if correct, else 0.0.
    correct = str(sentence.lower().count(letter.lower()))
    yield 1.0 if correct in (answer or "") else 0.0


# =============================================================================
# 3. MCP TOOLS (optional) - expose custom tools to the agent
# =============================================================================
# Run a FastMCP server in @env.initialize and register it as a capability.
# The agent gets the tools on its next manifest negotiation.
#
#   from fastmcp import FastMCP
#   from hud.capabilities import Capability
#
#   server = FastMCP(name="{env_name}-tools")
#
#   @server.tool()
#   async def my_tool(arg: str) -> str: ...
#
#   @env.initialize
#   async def _start():
#       import asyncio
#       asyncio.create_task(server.run_http_async(host="127.0.0.1", port=8765))
#       await asyncio.sleep(0.2)  # let the server bind
#       env.add_capability(Capability.mcp(name="tools", url="http://127.0.0.1:8765/mcp"))


# =============================================================================
# TEST - run with: python env.py
# =============================================================================

async def test():
    from hud.agents.claude import ClaudeAgent
    from hud import LocalRuntime

    agent = ClaudeAgent()

    # Calling a task binds a runnable Task; ``runtime=LocalRuntime(__file__)`` serves this
    # file in a child process and runs the task against it over the wire.
    task = count(sentence="Strawberry world", letter="r")
    job = await task.run(agent, runtime=LocalRuntime(__file__))

    print("reward:", job.reward)


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
#   ts = Taskset(
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
requires-python = ">=3.11"
dependencies = ["hud-python"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
