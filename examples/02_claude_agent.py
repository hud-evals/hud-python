#!/usr/bin/env python3
"""
Claude Agent Example

This example showcases Claude-specific features:
- Initial screenshot capture
- Thinking/reasoning display
- Computer tool usage
- Model-specific parameters

Claude is particularly good at visual understanding and
multi-step reasoning tasks.
"""

import asyncio

import hud
from hud.agents.claude import ClaudeAgent
from hud.eval.task import Task


async def main() -> None:
    # For any environment, you can run:
    # hud debug <IMAGE_NAME> to see the logs
    # hud analyze <IMAGE_NAME> to get a report about its capabilities

    initial_url = "https://httpbin.org/forms/post"

    prompt = f"""
    Please help me test a web form:
    1. Navigate to {initial_url}
    2. Fill in the customer name as "Claude Test"
    3. Enter the telephone as "555-0123"
    4. Type "Testing form submission with Claude" in the comments
    5. Select a small pizza size
    6. Choose "bacon" as a topping
    7. Set delivery time to "20:30"
    8. Submit the form
    9. Verify the submission was successful
    """

    # Create v5 Task with Environment config
    task = Task(
        env={"name": "browser"},  # Connect to browser hub
        scenario="form_fill",  # Scenario name
        args={"url": initial_url},  # Scenario args
        agent_config={"system_prompt": prompt},  # Pass prompt via agent config
    )

    # Create Claude-specific agent
    agent = ClaudeAgent.create(
        model="claude-sonnet-4-20250514",
        allowed_tools=["anthropic_computer"],
        initial_screenshot=True,
    )

    print("📋 Task: Multi-step form interaction")
    print("🚀 Running Claude agent...\n")

    # Run with hud.eval() context
    async with hud.eval(task, name="claude-form-demo") as ctx:
        result = await agent.run(ctx, max_steps=15)

    print("\n✨ Claude agent demo complete!")
    print(f"   Reward: {result.reward}")
    print(f"   Done: {result.done}")


if __name__ == "__main__":
    asyncio.run(main())
