from __future__ import annotations

import asyncio
from pathlib import Path

from hud.agents.openrouter import OpenRouterAgent
from hud.utils.hud_console import HUDConsole


async def main() -> None:
    hud_console = HUDConsole()

    # Inline FastMCP sum task (no external JSON needed)
    server_path = Path(__file__).parent / "mcp_sum_server.py"
    task = {
        "id": "sum-demo",
        "prompt": "Call the `sum` tool to add 7 and 5, then reply with the total in natural language.",
        "mcp_config": {
            "local": {
                "command": "python",
                "args": [str(server_path)],
            }
        },
        "agent_config": {
            "allowed_tools": ["sum"],
            "system_prompt": (
                "You are a concise math assistant. Always call the `sum` tool when asked to add "
                "numbers, wait for the result, then explain the answer in one sentence."
            ),
        },
    }

    # Instantiate the OpenRouter agent (uses OPENROUTER_API_KEY from env)
    agent = OpenRouterAgent(model_name="z-ai/glm-4.5v", verbose=True)

    hud_console.info("Running task with OpenRouter agent...")
    result = await agent.run(task, max_steps=3)

    hud_console.info("\nFinal content:")
    hud_console.info(result.content or "<empty>")
    hud_console.success(f"Reward: {result.reward}")


if __name__ == "__main__":
    asyncio.run(main())


