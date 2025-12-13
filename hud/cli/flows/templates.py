"""Templates for hud init command."""

DOCKERFILE_HUD = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
COPY . .

# Most of the time this command should not change, except if you change your env path
# or launch some other service before running the environment
CMD ["uv", "run", "python", "-m", "hud", "dev", "env:env", "--stdio"]
"""

# fmt: off
ENV_PY = '''\
"""{env_name} - HUD Environment"""

import asyncio

import hud
from hud.settings import settings
from openai import AsyncOpenAI, Omit
from hud.environment import Environment

env = Environment("{env_name}")


# =============================================================================
# 1. TOOLS - Functions the agent can call
# =============================================================================

@env.tool()
def count_letter(text: str, letter: str) -> int:
    """Count occurrences of a letter in text."""
    return text.lower().count(letter.lower())


# =============================================================================
# 2. SCRIPTS - Define prompts and evaluation logic
# =============================================================================

@env.scenario("count")
async def count_script(sentence: str, letter: str, fmt: str = "integer"):
    """Agent must count a letter. We check if they got it right."""
    # Yield the prompt, receive the agent's final answer
    answer = yield f"How many times does '{{letter}}' appear in: '{{sentence}}'? Format: {{fmt}}."

    # Score: 1.0 if correct, 0.0 otherwise
    correct = str(sentence.lower().count(letter.lower()))
    yield correct in answer


# =============================================================================
# 3. CONNECT EXISTING SERVERS (optional)
# =============================================================================

# --- FastAPI app ---
# from my_app import app
# env.connect_fastapi(app)

# --- FastMCP / MCPServer ---
# from my_server import mcp
# env.connect_server(mcp)

# --- OpenAPI spec (URL or file path) ---
# env.connect_openapi("https://api.example.com/openapi.json")

# --- MCP config (stdio or SSE) ---
# env.connect_mcp_config({{
#     "my-server": {{"command": "uvx", "args": ["some-mcp-server"]}}
# }})

# --- HUD hub (requires deployment, see below) ---
# env.connect_hub("my-org/my-env", prefix="remote")


# =============================================================================
# TEST - Run with: python env.py
# =============================================================================

async def test():
    client = AsyncOpenAI(
        base_url=settings.hud_gateway_url,
        api_key=settings.api_key,
    )

    # Create a task from the scenario
    task = env("count", sentence="Strawberry world", letter="r")

    # Test with and without tools
    async with hud.eval(task, variants={{"tools": [True, False]}}) as ctx:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{{"role": "user", "content": ctx.prompt}}],
            tools=ctx.as_openai_chat_tools() if ctx.variants["tools"] else Omit(),
        )

        # Handle tool calls if present
        message = response.choices[0].message
        if message.tool_calls:
            result = await ctx.call_tool(message.tool_calls[0])
            answer = str(result["content"])
        else:
            answer = message.content

        await ctx.submit(answer or "")


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
