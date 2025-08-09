#!/usr/bin/env python3
"""
Complete Agent Lifecycle Example

This example demonstrates the full agent lifecycle:
- Task definition with setup and evaluation tools
- Agent initialization
- Setup phase
- Agent execution loop
- Tool call handling
- Evaluation phase
- Cleanup

The entire flow is wrapped in hud.trace() to provide RUN_ID context.
"""

import asyncio
import hud
from hud.datasets import TaskConfig
from hud.mcp import ClaudeMCPAgent
from hud.mcp.client import MCPClient


async def main():
    # Wrap everything in trace to provide RUN_ID for the task
    with hud.trace("Agent Lifecycle Demo"):
        # Define a complete task with setup and evaluation
        task = TaskConfig(
            prompt="Create a new todo item with the title 'Buy groceries' and description 'Milk, eggs, bread'",
            mcp_config={
                "hud": {
                    "url": "https://mcp.hud.so/v3/mcp",
                    "headers": {
                        "Authorization": "Bearer ${HUD_API_KEY}",
                        "Mcp-Image": "hudpython/hud-remote-browser:latest",
                        "Run-Id": "${RUN_ID}",  # Automatically filled from trace
                    },
                }
            },
            setup_tool={"name": "setup", "arguments": {"name": "todo_creation_test"}},
            evaluate_tool={"name": "evaluate", "arguments": {"name": "todo_creation_test"}},
        )

        # Create MCP client with resolved config
        client = MCPClient(mcp_config=task.mcp_config)

        # Create agent
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            # Phase 1: Initialize agent with task context
            print("🔧 Initializing agent...")
            await agent.initialize(task)

            # Phase 2: Run setup tool
            print("📋 Running setup...")
            setup_result = await agent.call_tool(task.setup_tool)
            if setup_result.isError:
                print(f"❌ Setup failed: {setup_result.content}")
                return
            print(f"✅ Setup complete")

            # Phase 3: Run agent loop
            print(f"\n🤖 Running task: {task.prompt}")
            messages = await agent.create_initial_messages(task.prompt, None)

            done = False
            steps = 0
            max_steps = 10

            while not done and steps < max_steps:
                # Get model response
                response = await agent.get_model_response(messages)
                print(f"\n   Step {steps + 1}:")

                if response.content:
                    print(f"   💭 Agent: {response.content[:100]}...")

                if response.tool_calls:
                    # Execute tool calls
                    tool_results = []
                    for tool_call in response.tool_calls:
                        print(f"   🔧 Calling tool: {tool_call.name}")
                        result = await agent.call_tool(tool_call)
                        tool_results.append(result)

                        # Show result preview
                        if not result.isError:
                            preview = str(result.content)[:100]
                            print(f"      ✓ Result: {preview}...")

                    # Format results back into messages
                    messages.extend(
                        await agent.format_tool_results(response.tool_calls, tool_results)
                    )
                else:
                    # No more tool calls, we're done
                    done = True

                steps += 1

            # Phase 4: Run evaluation
            print("\n📊 Running evaluation...")
            eval_result = await agent.call_tool(task.evaluate_tool)

            if eval_result.isError:
                print(f"❌ Evaluation failed: {eval_result.content}")
            else:
                # Extract reward from evaluation
                eval_data = eval_result.content[0] if eval_result.content else {}
                reward = (
                    eval_data.get("text", "").split("reward=")[-1].split()[0]
                    if isinstance(eval_data, dict) and "text" in eval_data
                    else "unknown"
                )
                print(f"✅ Evaluation complete - Reward: {reward}")

            # Summary
            print(f"\n📈 Summary:")
            print(f"   Total steps: {steps}")
            print(f"   Task completed: {done}")

        finally:
            # Phase 5: Cleanup
            print("\n🧹 Cleaning up...")
            await client.close()

    print("\n✨ Agent lifecycle demo complete!")


if __name__ == "__main__":
    print("🚀 Agent Lifecycle Example")
    print("=" * 50)
    asyncio.run(main())
