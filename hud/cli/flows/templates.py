"""Templates for hud init command."""

DOCKERFILE_HUD = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
COPY . .

CMD ["uv", "run", "python", "-m", "hud", "dev", "hud:env", "--stdio"]
"""

# fmt: off
HUD_PY = '''\
"""{env_name} - HUD Environment"""

import asyncio
import os

from hud.environment import Environment

env = Environment("{env_name}")


# =============================================================================
# 1. ADD FUNCTIONS AS TOOLS
# =============================================================================
# Decorate any function with @env.tool() to expose it as a tool.

@env.tool()
def hud(query: str) -> str:
    """A tool that returns the answer to any question."""
    return f"Oh, I know the answer to '{{query}}', it's 42."


# =============================================================================
# 2. IMPORT FROM EXISTING SERVERS
# =============================================================================

# --- FastAPI app ---
# from my_app import app
# env.connect_fastapi(app)

# --- FastMCP / MCPServer ---
# from my_server import mcp
# env.connect_server(mcp)

# --- OpenAPI spec (URL or file path) ---
# env.connect_openapi("https://api.example.com/openapi.json")


# =============================================================================
# 3. CONNECT REMOTE SERVERS
# =============================================================================

# --- MCP config (stdio or SSE) ---
# env.connect_mcp_config({{
#     "my-server": {{"command": "uvx", "args": ["some-mcp-server"]}}
# }})

# --- HUD hub (requires deployment, see below) ---
# env.connect_hub("my-org/my-env", prefix="remote")


# =============================================================================
# TEST - Run with: python hud.py
# =============================================================================

async def test():
    from openai import AsyncOpenAI

    async with env.task("test") as ctx:
        # 1. List tools
        tools = await env.list_tools()
        print(f"Tools: {{[t.name for t in tools]}}")

        # 2. Call the hud tool
        result = await env.call_tool("hud", query="What is HUD?")
        print(f"HUD result: {{result}}")

        # 3. Call inference.hud.ai
        client = AsyncOpenAI(
            base_url="https://inference.hud.ai/v1",
            api_key=os.environ.get("HUD_API_KEY", ""),
        )
        response = await client.chat.completions.create(
            model="claude-sonnet-4-5",
            messages=[{{"role": "user", "content": "Say hello in one word."}}],
        )
        print(f"LLM: {{response.choices[0].message.content}}")

        # 4. Assign reward
        ctx.reward = 1.0 if "42" in str(result) else 0.0
        print(f"Reward: {{ctx.reward}}")


if __name__ == "__main__":
    asyncio.run(test())


# =============================================================================
# DEPLOYMENT
# =============================================================================
# To deploy this environment on HUD:
#
# 1. Push this repo to GitHub
# 2. Go to hud.ai -> New -> Environment
# 3. Choose "From GitHub URL" and paste your repo URL
# 4. This deploys the environment for remote connection
#
# Once deployed, connect to it from other environments:
#   env.connect_hub("{env_name}")
#
# Remote deployment enables:
# - Parallelized evaluations (run many agents simultaneously)
# - Training data collection at scale
# - Shared environments across team members
#
# Note: The test() function above is just for local testing.
# It's not required for the deployed environment.
'''
# fmt: on

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
