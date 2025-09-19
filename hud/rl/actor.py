"""Actor for episode collection using vLLM and HUD."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import hud
from hud.types import Task, Trace
from hud.utils.agent_factories import create_openai_agent
from hud.utils.hud_console import HUDConsole

from .config import Config

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)


class Actor:
    """Collects episodes using vLLM-served models via HUD agents."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.actor_config = config.actor

    def create_agent(self):
        """Create an agent with the current adapter."""
        return create_openai_agent(
            base_url=self.actor_config.vllm_base_url.replace("localhost", "127.0.0.1"),
            api_key=self.actor_config.vllm_api_key,
            request_timeout=self.actor_config.request_timeout,
            model_name=self.config.model.base_model,
            allowed_tools=self.actor_config.allowed_tools,
            append_setup_output=False,
            verbose=self.config.verbose,
            completion_kwargs={
                "temperature": self.actor_config.temperature,
                "max_tokens": self.actor_config.max_new_tokens,
                "tool_choice": "required" if self.actor_config.force_tool_choice else "auto",
            },
        )
    
    async def run_tasks(self, tasks: list[Task], job_id: str) -> list[Trace]:
        """Run tasks and collect traces using semaphore for concurrency control."""
        # Create semaphore to limit concurrent episodes
        semaphore = asyncio.Semaphore(self.actor_config.max_parallel_episodes)
        
        async def run_with_semaphore(task: Task) -> Trace:
            """Run a single task with semaphore and timeout protection."""
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._run_task(task, job_id),
                        timeout=self.actor_config.episode_timeout_sec,
                    )
                except asyncio.TimeoutError:
                    hud_console.warning_log(f"Episode timed out for task {task.id}")
                    return Trace(isError=True, content="Episode timeout")
                except Exception as e:
                    hud_console.warning_log(f"Episode error for task {task.id}: {e}")
                    return Trace(isError=True, content=str(e))
        
        # Run all tasks concurrently with semaphore limiting
        results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=True,
        )
        
        # Normalize any remaining exceptions to error traces
        traces = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                hud_console.warning_log(f"Unexpected error for task {tasks[i].id}: {result}")
                traces.append(Trace(isError=True, content=str(result)))
            else:
                traces.append(result)
        
        return traces

    async def _run_task(self, task: Task, job_id: str) -> Trace:
        """Run a single task."""
        agent = self.create_agent()

        # Run the task
        with hud.trace(f"Training | {task.id}", job_id=job_id):
            result = await agent.run(
                task,
                max_steps=self.actor_config.max_steps_per_episode
            )

        result.info["tool_spec"] = agent.get_tool_schemas()

        return result


def save_traces(traces: list[Trace], output_path: str) -> None:
    """Save traces to JSON file."""
    with open(output_path, "w") as f:
        json.dump([trace.model_dump(mode="json") for trace in traces], f, indent=2)
    print(f"Saved {len(traces)} traces to {output_path}")


if __name__ == "__main__":
    from hud.datasets import Task
    import uuid

    async def test_actor() -> None:
        """Test the actor with a single 2048 task using local hud-browser image."""
        config = Config()
        config.actor.max_parallel_episodes = 16
        config.actor.max_steps_per_episode = 10
        config.actor.temperature = 0.6
        config.actor.force_tool_choice = False
        config.verbose = True

        # Create test task with local hud-browser image
        task_data = {
            "id": "test_2048_128",
            "prompt": "Play the browser-based 2048 game and try to reach the 128 tile. Start by taking a screenshot, then make strategic moves using arrow keys.",  # noqa: E501
            "mcp_config": {
                "local": {
                    "command": "sh",
                    "args": ["-c", f"docker run --rm --platform linux/amd64 -i -e AGENT_DISPLAY_WIDTH=588 -e AGENT_DISPLAY_HEIGHT=336 hud-browser 2>/dev/null"]
                }
            },
            "setup_tool": {"name": "launch_app", "arguments": {"app_name": "2048"}},
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}},
            },
            "system_prompt": '''You are an expert 2048 game player using a browser interface. Your goal is to reach the tile specified by the user.
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
''',
            "agent_tools": ["computer"],
        }

        task = Task(**task_data)
        actor = Actor(config)

        logger.info("Testing actor with task: %s", task.id)
        logger.info("Model: %s", config.model.base_model)
        logger.info("VLLM: %s", config.actor.vllm_base_url)

        job_id = str(uuid.uuid4())
        with hud.job("Test Actor", job_id=job_id):
           traces = await actor.run_tasks([task]*32, job_id=job_id)

        for trace in traces:
            print(f"Trace completed - Reward: {trace.reward}")
        
        # Dump traces for training system testing
        output_file = f"traces_{job_id}.json"
        save_traces(traces, output_file)

    asyncio.run(test_actor())
