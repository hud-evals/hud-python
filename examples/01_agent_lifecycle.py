#!/usr/bin/env python3
"""
Complete Agent Lifecycle Example

This example demonstrates the full agent lifecycle using the v5 Task format:
- Task definition with Environment and scenario
- hud.eval() context for connection and tracing
- Agent initialization and execution
- Automatic scenario setup/evaluation
- Result collection

For simpler usage, just use `await agent.run(ctx)` which handles everything.
This example shows what happens under the hood.
"""

import asyncio

import hud
from hud.agents.claude import ClaudeAgent
from hud.eval.task import Task


async def main() -> None:
    print("🚀 Agent Lifecycle Example")
    print("=" * 50)

    # Phase 1: Define task using v5 Task format
    # The Task holds environment config and scenario info
    print("📋 Defining task...")
    task = Task(
        # Environment config - connects to HUD browser hub
        env={"name": "browser"},
        # Scenario to run (defined on the environment)
        scenario="checkout",
        # Scenario arguments
        args={"product": "laptop", "quantity": 1},
        # Optional: agent configuration
        agent_config={"system_prompt": "You are a helpful shopping assistant."},
    )

    # Phase 2: Create agent
    print("🤖 Creating Claude agent...")
    agent = ClaudeAgent.create(
        model="claude-sonnet-4-20250514",
    )

    # Phase 3: Enter eval context and run agent
    # The context manager handles:
    # - Environment connection (MCP servers start)
    # - Scenario setup execution
    # - Trace creation for telemetry
    print("🔧 Entering eval context...")
    async with hud.eval(task, name="agent-lifecycle-demo") as ctx:
        print("   ✅ Environment connected")
        print(f"   📝 Prompt: {ctx.prompt[:50] if ctx.prompt else 'N/A'}...")

        # Phase 4: Run the agent
        # agent.run() handles the agentic loop:
        # - Gets system messages
        # - Sends prompt to model
        # - Processes tool calls
        # - Continues until done or max_steps
        print("\n🏃 Running agent loop...")
        result = await agent.run(ctx, max_steps=10)

        print(f"\n   Agent finished:")
        print(f"   - Done: {result.done}")
        print(f"   - Has error: {result.isError}")
        if result.content:
            print(f"   - Response: {result.content[:100]}...")

    # Phase 5: After exit, scenario evaluation was automatically called
    # and ctx.reward is set based on the evaluation
    print("\n📊 Evaluation complete")
    print(f"   Reward: {ctx.reward}")
    print(f"   Success: {ctx.success}")

    print("\n✨ Agent lifecycle demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
