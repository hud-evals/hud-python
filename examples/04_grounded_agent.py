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
from hud.eval.task import Task
from hud.settings import settings
from hud.tools.grounding import GrounderConfig
from openai import AsyncOpenAI


async def main() -> None:
    """Run the grounded agent example."""

    # Configure the grounding model
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is required for grounding model")

    grounder_config = GrounderConfig(
        api_base="https://openrouter.ai/api/v1",  # OpenRouter API
        model="qwen/qwen-2.5-vl-7b-instruct",  # Vision model for grounding
        api_key=openrouter_key,
    )

    # Create OpenAI client for planning
    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", settings.openai_api_key)
    )  # can use any OpenAI-compatible endpoint

    agent = GroundedOpenAIChatAgent.create(
        grounder_config=grounder_config,
        openai_client=openai_client,
        checkpoint_name="gpt-4o-mini",  # Planning model
    )

    form_url = "https://hb.cran.dev/forms/post"

    form_prompt = """
    Fill out the form:
    1. Enter "Grounded Test" in the customer name field
    2. Enter "555-9876" in the telephone field
    3. Type "Testing grounded agent with separated vision and reasoning" in comments
    4. Select medium pizza size
    5. Choose mushroom as a topping
    6. Submit the form
    """

    # Create v5 Task with local Docker environment
    env = hud.Environment("browser-grounded")
    env.connect_image(
        "hudevals/hud-browser:0.1.6",
        docker_args=["-p", "8080:8080"],
    )

    task = Task(
        env=env,
        scenario="form_fill",
        args={"url": form_url},
        agent_config={"system_prompt": form_prompt},
    )

    print("📋 Task: Form interaction with grounded agent")
    print("🚀 Running grounded agent...\n")

    try:
        async with hud.eval(task, name="grounded-form-demo") as ctx:
            result = await agent.run(ctx, max_steps=10)
        print(f"Result: {result.content}\n")
    except Exception as e:
        print(f"Error during agent execution: {e}")

    print("\n✨ Grounded agent demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
