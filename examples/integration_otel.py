"""Example: Running HUD agents with Jaeger as the tracing backend.

This example shows how to run a normal HUD agent (playing 2048 game)
but send all traces to Jaeger instead of HUD's backend.

To run:
1. Build the 2048 game:
   docker build -t hud-text-2048 ../environments/text_2048

2. Start Jaeger:
   docker run -d --name jaeger \
     -e COLLECTOR_OTLP_ENABLED=true \
     -p 16686:16686 -p 4318:4318 \
     jaegertracing/all-in-one:latest

3. Run this example:
   python custom_otel_backend.py

4. View traces at http://localhost:16686
   - Service: "hud-2048-jaeger"
   - You'll see the agent's get_model_response and execute_tools spans

5. Cleanup:
   docker stop jaeger && docker rm jaeger
"""

import asyncio

# Configure telemetry BEFORE importing agents to use Jaeger
from hud.otel import configure_telemetry

configure_telemetry(
    service_name="hud-2048-jaeger",
    enable_otlp=True,
    otlp_endpoint="localhost:4318",  # Jaeger's OTLP HTTP endpoint
)

# Now import everything else
import hud
from hud.agents import ClaudeAgent
from hud.clients import MCPClient
from hud.datasets import Task


async def main():
    """Run 2048 game with Claude agent, traces go to Jaeger."""

    task_dict = {
        "prompt": "Play 2048 and try to get as high as possible. Do not stop even after 2048 is reached.",
        "mcp_config": {
            "local": {"command": "docker", "args": ["run", "--rm", "-i", "hud-text-2048"]}
        },
        "setup_tool": {
            "name": "setup",
            "arguments": {"name": "board", "arguments": {"board_size": 4}},
        },
        "evaluate_tool": {
            "name": "evaluate",
            "arguments": {"name": "max_number"},
        },
    }
    task = Task(**task_dict)

    # Create client and agent
    mcp_client = MCPClient(mcp_config=task.mcp_config)
    # Create agent - its methods are already instrumented with @hud.instrument
    agent = ClaudeAgent(
        mcp_client=mcp_client,
    )

    # Run with hud.trace() - this creates the root span in Jaeger
    with hud.trace("play_2048_game"):
        print(f"🎮 Starting 2048 game")

        # Agent will play the game with setup and evaluate phases
        # Each call to get_model_response() and execute_tools()
        # will create child spans in Jaeger automatically
        result = await agent.run(task, max_steps=20)

        print(f"\n🏁 Game finished!")
        print(f"   Final reward: {result.reward}")
        print(f"   Success: {not result.isError}")

    print("\n✅ All traces sent to Jaeger!")
    print("🔍 View at: http://localhost:16686")
    print("   - Service: 'hud-2048-jaeger'")
    print("   - You'll see the agent's reasoning and tool calls")


if __name__ == "__main__":
    asyncio.run(main())
