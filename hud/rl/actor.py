"""Actor for episode collection using vLLM and HUD."""

from __future__ import annotations

import asyncio
import logging

import httpx
from openai import AsyncOpenAI

import hud
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from hud.clients.utils.retry_transport import create_retry_httpx_client
from hud.types import Task, Trace
from hud.utils.hud_console import HUDConsole

from .config import Config

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)


class Actor:
    """Collects episodes using vLLM-served models via HUD agents."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.actor_config = config.actor
        
        # Setup OpenAI client for vLLM
        base_url = self.actor_config.vllm_base_url.replace("localhost", "127.0.0.1")
        self.openai_client = self._create_openai_client(base_url)

    def _create_openai_client(self, base_url: str) -> AsyncOpenAI:
        """Create OpenAI client with optimized settings for vLLM."""
        http_client = create_retry_httpx_client(
            timeout=httpx.Timeout(self.actor_config.request_timeout),   
        )
        return AsyncOpenAI(
            base_url=base_url,
            api_key=self.actor_config.vllm_api_key,
            http_client=http_client,
            max_retries=2,
        )

    def create_agent(self) -> GenericOpenAIChatAgent:
        """Create an agent with the current adapter."""
        return GenericOpenAIChatAgent(
            openai_client=self.openai_client,
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


if __name__ == "__main__":
    from hud.datasets import Task
    import uuid

    async def test_actor() -> None:
        """Test the actor with a single 2048 task using local hud-browser image."""
        config = Config()
        config.actor.max_parallel_episodes = 2
        config.actor.max_steps_per_episode = 6
        config.verbose = True

        # Create test task with local hud-browser image
        task_data = {
            "id": "test_2048_128",
            "prompt": "Play the browser-based 2048 game and try to reach the 128 tile. Start by taking a screenshot, then make strategic moves using arrow keys.",  # noqa: E501
            "mcp_config": {
                "local": {
                    "command": "sh",
                    "args": ["-c", "docker run --rm --platform linux/amd64 -i hudevals/hud-browser:0.1.6 2>/dev/null"]
                }
            },
            "setup_tool": {"name": "launch_app", "arguments": {"app_name": "2048"}},
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}},
            },
            "system_prompt": "You are an expert 2048 game player. Use arrow keys to reach the target tile. First take a screenshot, then make strategic moves.",
            "agent_tools": ["computer"],
        }

        task = Task(**task_data)
        actor = Actor(config)

        logger.info("Testing actor with task: %s", task.id)
        logger.info("Model: %s", config.model.base_model)
        logger.info("VLLM: %s", config.actor.vllm_base_url)

        job_id = str(uuid.uuid4())
        with hud.job("Test Actor", job_id=job_id):
            await actor.run_tasks([task]*5, job_id=job_id)

    asyncio.run(test_actor())
