import asyncio
import uuid

import hud
from hud.datasets import Task
from hud.rl.actor import Actor
from hud.rl.config import ActorConfig
from hud.rl.logger import configure_logging, console
from hud.rl.utils import save_traces
from openai import AsyncOpenAI


async def test_actor() -> None:
    """Test the actor with a single 2048 task using a local hud-browser image."""
    config = ActorConfig()
    config.max_parallel_episodes = 16
    config.max_steps_per_episode = 10
    config.temperature = 0.6
    config.force_tool_choice = False
    config.verbosity = 1
    configure_logging(verbosity=config.verbosity)

    task_data = {
        "id": "test_2048_128",
        "prompt": "Play the browser-based 2048 game and try to reach the 128 tile. Start by taking a screenshot, then make strategic moves using arrow keys.",
        "mcp_config": {
            "hud": {
                "url": "https://mcp.hud.so/v3/mcp",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Mcp-Image": "hudevals/hud-browser:0.1.7",
                    "Env-Agent-Display-Width": "588",
                    "Env-Agent-Display-Height": "336",
                },
            }
        },
        "setup_tool": {"name": "launch_app", "arguments": {"app_name": "2048"}},
        "evaluate_tool": {
            "name": "evaluate",
            "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}},
        },
        "agent_config": {
            "system_prompt": """You are an expert 2048 game player using a browser interface. Your goal is to reach the tile specified by the user.
HOW 2048 WORKS:
- 4x4 grid with numbered tiles (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048...)
- When you move, all tiles slide in that direction
- When two tiles with SAME number touch, they merge into one (2+2=4, 4+4=8, etc.)
- After each move, a new tile (2 or 4) appears randomly
- Game ends when grid is full and no merges possible

BROWSER INTERACTION USING THE COMPUTER TOOL:
1. FIRST TURN ONLY - TAKE SCREENSHOT:
   Use: {"name": "computer", "arguments": {"action": "screenshot"}}
   After that, the environment returns an image with each successful move.

2. MAKE MOVES - Use the computer tool with action="press" and keys parameter (list of strings):
   - Move up: {"name": "computer", "arguments": {"action": "press", "keys": ["up"]}}
   - Move down: {"name": "computer", "arguments": {"action": "press", "keys": ["down"]}}
   - Move left: {"name": "computer", "arguments": {"action": "press", "keys": ["left"]}}
   - Move right: {"name": "computer", "arguments": {"action": "press", "keys": ["right"]}}

IMPORTANT GAME PLAYING GUIDELINES:
- You can make MULTIPLE actions in a turn (press multiple keys)
- ONLY USE THE PRESS ACTION
- After each turn, you will see the result of the tool calls and the new game state. CONTINUE playing until the objective is clearly achieved
- Do NOT stop playing just because you made several moves - keep going until you reach the target
- There is NO "terminate" or "stop" action - you continue until the task is complete
- If you're unsure whether the target is reached, make one more move to verify, then continue if needed
- The game/environment will naturally end when appropriate - you don't need to manually stop it
- Keep taking actions until you can clearly see the objective has been met

Strategy: keep highest tiles in a corner; maintain order; avoid random moves.
""",
            "allowed_tools": ["computer"],
        },
    }

    task = Task(**task_data)
    client = AsyncOpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY", timeout=30.0)

    actor = Actor(config, client=client)

    console.info(f"Testing actor with task: {task.id}")
    console.info(f"Model: {config.base_model}")
    console.info(f"Client base URL: {client.base_url}")

    job_id = str(uuid.uuid4())
    async with hud.async_job("Test Actor", job_id=job_id):
        traces = await actor.run_tasks([task] * 32, job_id=job_id)

    for trace in traces:
        console.info(f"Trace completed - Reward: {trace.reward}")

    output_file = f"tests/data/traces_{job_id}.json"
    save_traces(traces, output_file)

    from hud.utils.task_tracking import wait_all_tasks

    await wait_all_tasks()


if __name__ == "__main__":
    asyncio.run(test_actor())
