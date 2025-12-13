#!/usr/bin/env python3
"""
Complete Agent Lifecycle Example

This example demonstrates the full agent lifecycle using Task.from_v4():
- Task definition with setup and evaluation tools (v4 LegacyTask format)
- Conversion to v5 Task using Task.from_v4()
- hud.eval() context for connection and tracing
- Agent initialization and execution
- Automatic setup/evaluate tool execution
- Result collection

For simpler usage, just use `await agent.run(ctx)` which handles everything.
This example shows what happens under the hood.
"""

import asyncio
import hud
from hud.datasets import LegacyTask
from hud.eval.task import Task
from hud.agents import ClaudeAgent


async def main():
    print("🚀 Agent Lifecycle Example")
    print("=" * 50)

    # Phase 1: Define task in v4 LegacyTask format
    # This format includes setup_tool and evaluate_tool
    print("📋 Defining task...")
    legacy_task = LegacyTask(
        prompt="Create a new todo item with the title 'Buy groceries' and description 'Milk, eggs, bread'",
        mcp_config={
            "hud": {
                "url": "https://mcp.hud.ai/v3/mcp",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",  # Auto-resolved from env
                    "Mcp-Image": "hudevals/hud-browser:latest",
                },
            }
        },
        setup_tool={"name": "launch_app", "arguments": {"app_name": "todo"}},
        evaluate_tool={
            "name": "evaluate",
            "arguments": {"name": "todo_exists", "arguments": {"title": "Buy groceries"}},
        },
    )

    # Phase 2: Convert to v5 Task
    # Task.from_v4() creates an Environment with:
    # - mcp_config connection (connects on context entry)
    # - setup_tool calls (run on context entry)
    # - evaluate_tool calls (run on context exit)
    print("🔄 Converting to v5 Task...")
    task = Task.from_v4(legacy_task)

    # Phase 3: Create agent
    print("🤖 Creating Claude agent...")
    agent = ClaudeAgent.create(
        checkpoint_name="claude-sonnet-4-5",
        allowed_tools=["anthropic_computer"],
        initial_screenshot=True,
    )

    # Phase 4: Enter eval context and run agent
    # The context manager handles:
    # - Environment connection (MCP servers start)
    # - Setup tools execution (launch_app)
    # - Trace creation for telemetry
    print("🔧 Entering eval context...")
    async with task as ctx:
        print(f"   ✅ Environment connected")
        print(f"   ✅ Setup tools executed")
        print(f"   📝 Prompt: {ctx.prompt[:50]}...")

        # Phase 5: Run the agent
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

    # Phase 6: After exit, evaluate_tool was automatically called
    # and ctx.reward is set based on the evaluation
    print("\n📊 Evaluation complete (via evaluate_tool)")
    print(f"   Reward: {ctx.reward}")
    print(f"   Success: {ctx.success}")

    print("\n✨ Agent lifecycle demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
