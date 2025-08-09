#!/usr/bin/env python3
"""
Task with Setup and Evaluation Example

This example demonstrates the complete TaskConfig lifecycle:
- Defining tasks with setup and evaluation phases
- Running setup to prepare the environment
- Executing the main task
- Evaluating success with specific criteria
- Getting rewards/scores

This pattern is essential for:
- Reproducible evaluations
- Benchmarking agents
- Automated testing
"""

import asyncio
import hud
from hud.datasets import TaskConfig
from hud.mcp import ClaudeMCPAgent
from hud.mcp.client import MCPClient


async def main():
    print("📋 Task with Setup & Evaluation Example")
    print("=" * 50)

    with hud.trace("Task Lifecycle Demo"):
        # Define tasks using local browser environment
        tasks = [
            {
                "name": "Todo Creation Task",
                "config": TaskConfig(
                    prompt="Create a new todo item with the title 'Complete project' and description 'Finish the final report by Friday'",
                    mcp_config={
                        "browser": {
                            "command": "docker",
                            "args": [
                                "run",
                                "--rm",
                                "-i",
                                "-p",
                                "8080:8080",  # VNC port for viewing
                                "-e",
                                "LAUNCH_APPS=todo",  # Launch todo app
                                "hud-browser",
                            ],
                        }
                    },
                    setup_tool={"name": "setup", "arguments": {"name": "todo_creation"}},
                    evaluate_tool={"name": "evaluate", "arguments": {"name": "todo_creation"}},
                    metadata={"category": "todo", "difficulty": "easy"},
                ),
            },
            {
                "name": "Web Navigation Task",
                "config": TaskConfig(
                    prompt="Navigate to Wikipedia, search for 'Artificial Intelligence', and take a screenshot of the first paragraph",
                    mcp_config={
                        "browser": {
                            "command": "docker",
                            "args": [
                                "run",
                                "--rm",
                                "-i",
                                "-p",
                                "8080:8080",  # VNC port for viewing
                                "hud-browser",
                            ],
                        }
                    },
                    setup_tool={"name": "setup", "arguments": {"name": "blank_browser"}},
                    evaluate_tool={
                        "name": "evaluate",
                        "arguments": {"name": "wikipedia_ai_search"},
                    },
                    metadata={"category": "web_navigation", "difficulty": "medium"},
                ),
            },
        ]

        # Let user choose a task
        print("\nAvailable tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task['name']} - {task['config'].metadata['difficulty']}")

        choice = input("\nSelect task (1 or 2, default=1): ").strip() or "1"
        selected = tasks[int(choice) - 1] if choice in ["1", "2"] else tasks[0]

        task_config = selected["config"]
        print(f"\n✅ Selected: {selected['name']}")
        print(f"📝 Task: {task_config.prompt}")

        # Show browser viewing instructions
        print(f"\n👀 View the browser at: http://localhost:8080/vnc.html")
        print(f"   (Password: 'secret' if prompted)")

        # Create client and agent
        client = MCPClient(mcp_config=task_config.mcp_config)
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            # Run the complete task lifecycle
            print("\n🚀 Starting task lifecycle...\n")

            # The agent.run() method handles all phases when given a TaskConfig
            result = await agent.run(task_config, max_steps=15)

            # Extract results
            print("\n📊 Task Results:")
            print(f"   ✅ Success: {result.success}")
            print(f"   🏆 Reward: {result.reward}")
            print(f"   📈 Steps taken: {len(result.tool_calls)}")

            # Show evaluation details if available
            if result.evaluation_result:
                print(f"\n📋 Evaluation Details:")
                if isinstance(result.evaluation_result, dict):
                    for key, value in result.evaluation_result.items():
                        print(f"   - {key}: {value}")

            # Task metadata
            print(f"\n🏷️  Task Metadata:")
            print(f"   Category: {task_config.metadata.get('category')}")
            print(f"   Difficulty: {task_config.metadata.get('difficulty')}")

            # Performance analysis
            if result.reward == 1.0:
                print("\n🎉 Perfect score! Task completed successfully.")
            elif result.reward > 0:
                print(f"\n⚠️  Partial success. Score: {result.reward}")
            else:
                print("\n❌ Task failed. Review the steps to see what went wrong.")

        finally:
            await client.close()

    print("\n✨ Task lifecycle demo complete!")
    print("\n💡 Key Takeaways:")
    print("   - Setup ensures consistent starting conditions")
    print("   - Evaluation provides objective success metrics")
    print("   - Rewards enable benchmarking and comparison")
    print("   - TaskConfig encapsulates the complete workflow")


if __name__ == "__main__":
    asyncio.run(main())
