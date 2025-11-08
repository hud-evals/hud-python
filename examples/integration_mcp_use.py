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
from mcp_use import MCPAgent, MCPClient


async def main():
    print("🤖 MCP-Use Agent Example")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    print("✅ Using GPT-4")

    with hud.trace("MCP Use Agent Demo"):
        # Configure MCP connection
        # Can be local or cloud
        config = {
            "mcp_config": {
                "hud": {
                    "url": "https://mcp.hud.ai/v3/mcp",
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
            client=client,
            max_steps=10,
            verbose=True,
        )

        try:
            # Interactive task
            print("\n📝 Enter a task (or press Enter for default):")
            task = input("> ").strip()

            if not task:
                task = "Go to Wikipedia and find out when the Eiffel Tower was built"
                print(f"Using default task: {task}")

            print("\n🚀 Running agent...")

            # Execute task
            result = await agent.run(task)

            print("\n✅ Task completed!")
            print("\n📊 Results:")
            print(f"   Final answer: {result}")

            # MCP-Use provides execution history
            print("\n🔍 Execution history:")
            for i, step in enumerate(agent.get_conversation_history(), 1):
                print(f"   Step {i}: {step.content}")

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
        print("   Get your key at: https://hud.ai")
        exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No LLM API key found")
        print("   Set OPENAI_API_KEY")
        exit(1)

    asyncio.run(main())
