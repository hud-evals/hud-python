#!/usr/bin/env python3
"""
MCP-Use Agent Example

This example shows how to use the mcp_use library for a
provider-agnostic approach to MCP agents. Works with any
LangChain-compatible LLM.

Key benefits:
- Use any LLM provider (OpenAI, Anthropic, Cohere, etc.)
- Leverage LangChain's ecosystem
- Flexible tool filtering
"""

import asyncio
import os
import hud
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from mcp_use import MCPAgent, MCPClient


async def main():
    print("🤖 MCP-Use Agent Example")
    print("=" * 50)

    # Choose your LLM provider
    print("\nSelect LLM provider:")
    print("1. OpenAI (gpt-4o)")
    print("2. Anthropic (claude-3)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    # Configure LLM based on choice
    if choice == "2":
        llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0.3)
        print("✅ Using Claude-3")
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        print("✅ Using GPT-4")

    with hud.trace("MCP Use Agent Demo"):
        # Configure MCP connection
        # Can be local or cloud
        config = {
            "mcp_config": {
                "hud": {
                    "url": "https://mcp.hud.so/v3/mcp",
                    "headers": {
                        "Authorization": f"Bearer {os.getenv('HUD_API_KEY')}",
                        "Mcp-Image": "hudpython/hud-browser:latest",
                    },
                }
            }
        }

        # Create MCP client and agent
        client = MCPClient.from_dict(config)
        agent = MCPAgent(
            llm=llm,
            mcp_client=client,
            max_steps=10,
            verbose=True,  # Show detailed execution
        )

        try:
            # Interactive task
            print("\n📝 Enter a task (or press Enter for default):")
            task = input("> ").strip()

            if not task:
                task = "Go to Wikipedia and find out when the Eiffel Tower was built"
                print(f"Using default task: {task}")

            print(f"\n🚀 Running agent...\n")

            # Execute task
            result = await agent.run(task)

            print(f"\n✅ Task completed!")
            print(f"\n📊 Results:")
            print(f"   Final answer: {result}")

            # MCP-Use provides execution history
            if hasattr(agent, "history"):
                print(f"\n🔍 Execution history:")
                for i, step in enumerate(agent.history, 1):
                    print(f"   Step {i}: {step.get('action', 'unknown')}")

        finally:
            # Cleanup
            await client.close_all_sessions()

    print("\n✨ MCP-Use demo complete!")
    print("\n💡 Benefits of mcp_use:")
    print("   - Works with any LangChain LLM")
    print("   - Full tool discovery and filtering")
    print("   - Built-in execution history")
    print("   - Verbose mode for debugging")


if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("HUD_API_KEY"):
        print("⚠️  HUD_API_KEY not set")
        print("   Get your key at: https://app.hud.so")
        exit(1)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  No LLM API key found")
        print("   Set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        exit(1)

    asyncio.run(main())
