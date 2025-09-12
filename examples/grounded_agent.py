"""
Grounded Agent Example

This example demonstrates the GroundedOpenAIChatAgent that separates:
- Visual grounding (element detection) using a specialized vision model
- High-level reasoning using GPT-4o or similar

Prerequisites:
1. Set your API keys:
   export OPENAI_API_KEY=your_openai_key
   export OPENROUTER_API_KEY=your_openrouter_key
   export HUD_API_KEY=your_hud_key
"""

import asyncio
import os

import hud
from hud.agents.grounded_openai import GroundedOpenAIChatAgent
from hud.settings import settings
from hud.tools.grounding import GrounderConfig
from openai import AsyncOpenAI


async def main():
    """Run the grounded agent example."""

    with hud.trace("Grounded Agent Demo"):
        # Configure the grounding model
        grounder_config = GrounderConfig(
            api_base="https://openrouter.ai/api/v1",  # OpenRouter API
            model="qwen/qwen-2.5-vl-7b-instruct",  # Vision model for grounding
            api_key=settings.openrouter_api_key,
        )

        # MCP configuration for environment
        mcp_config = {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "-p", "8080:8080", "hudevals/hud-browser:0.1.3"],
            }
        }

        # Create OpenAI client for planning
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        )  # can use any OpenAI-compatible endpoint

        agent = GroundedOpenAIChatAgent(
            grounder_config=grounder_config,
            openai_client=openai_client,
            model_name="gpt-4o-mini",  # Planning model
        )

        try:
            # Create a task with MCP config
            from hud.datasets import Task

            form_url = "https://hb.cran.dev/forms/post"

            form_prompt = f"""
            Fill out the form:
            1. Enter "Grounded Test" in the customer name field
            2. Enter "555-9876" in the telephone field
            3. Type "Testing grounded agent with separated vision and reasoning" in comments
            4. Select medium pizza size
            5. Choose mushroom as a topping
            6. Submit the form
            """

            task = Task(
                prompt=form_prompt,
                mcp_config=mcp_config,
                setup_tool={
                    "name": "playwright",
                    "arguments": {"action": "navigate", "url": form_url},
                },
            )

            print(f"📋 Task: Form interaction")
            print(f"🚀 Running grounded agent...\n")

            result = await agent.run(task, max_steps=10)
            print(f"Result: {result.content}\n")

        except Exception as e:
            print(f"Error during agent execution: {e}")

    print("\n✨ Grounded agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
