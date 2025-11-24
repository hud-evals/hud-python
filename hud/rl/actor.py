"""Actor for episode collection using vLLM and HUD."""

from __future__ import annotations

import asyncio
import logging

import httpx
from openai import AsyncOpenAI

import hud
from hud.agents.openai_chat import OpenAIChatAgent
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
        self.current_adapter = config.model.base_model

        # Setup OpenAI client for vLLM
        base_url = self.actor_config.vllm_base_url.replace("localhost", "127.0.0.1")
        self.openai_client = self._create_openai_client(base_url)

    def _create_openai_client(self, base_url: str) -> AsyncOpenAI:
        """Create OpenAI client with optimized settings for vLLM."""
        # Match connection limits to parallel_episodes to avoid bottlenecks
        # Use shorter per-request timeout and keep retries modest to avoid long blocking
        http_client = create_retry_httpx_client(
            timeout=httpx.Timeout(60.0),
        )
        return AsyncOpenAI(
            base_url=base_url,
            api_key=self.actor_config.vllm_api_key,
            http_client=http_client,
            max_retries=2,
        )

    def create_agent(self) -> OpenAIChatAgent:
        """Create an agent with the current adapter."""
        return OpenAIChatAgent(
            openai_client=self.openai_client,
            model_name=self.current_adapter,
            allowed_tools=self.actor_config.allowed_tools,
            append_setup_output=False,
            system_prompt=self.actor_config.system_prompt,
            verbose=self.config.verbose,
            completion_kwargs={
                "temperature": self.actor_config.temperature,
                "max_tokens": self.actor_config.max_new_tokens,
                "tool_choice": "required" if self.actor_config.force_tool_choice else "auto",
            },
        )

    def update_adapter(self, adapter_name: str) -> None:
        """Update the current adapter being used."""
        self.current_adapter = adapter_name
        hud_console.info(f"[Actor] Using adapter: {adapter_name}")

    async def run_tasks(self, tasks: list[Task], job_id: str) -> list[Trace]:
        """Run tasks and collect traces."""
        traces = []

        # Process tasks in batches respecting max_parallel_episodes limit
        for batch_start in range(0, len(tasks), self.actor_config.max_parallel_episodes):
            batch_end = min(batch_start + self.actor_config.max_parallel_episodes, len(tasks))
            batch = tasks[batch_start:batch_end]

            # Run batch in parallel with per-episode timeout protection
            async def run_with_timeout(t: Task) -> Trace:
                try:
                    return await asyncio.wait_for(
                        self._run_task(t, job_id),
                        timeout=self.actor_config.episode_timeout_sec,
                    )
                except TimeoutError:
                    hud_console.warning_log(f"Episode timed out for task {t.id}")
                    # Attach task so buffer grouping has key
                    return Trace(isError=True, content="Episode timeout", task=t)

            results = await asyncio.gather(
                *[run_with_timeout(t) for t in batch],
                return_exceptions=True,
            )

            # Normalize exceptions to error traces and ensure task is attached
            for t, res in zip(batch, results, strict=False):
                if isinstance(res, Exception):
                    hud_console.warning_log(f"Episode error: {res}")
                    traces.append(Trace(isError=True, content=str(res), task=t))
                else:
                    traces.append(res)

        return traces

    async def _run_task(self, task: Task, job_id: str) -> Trace:
        """Run a single task."""
        agent = self.create_agent()

        # Run the task
        try:
            async with hud.async_trace(f"Training | {task.prompt}", job_id=job_id):
                result = await agent.run(task, max_steps=self.actor_config.max_steps_per_episode)

        except Exception:
            logger.info("GOT EXCEPTION")
            # Preserve task on exception for grouping
            return Trace(isError=True, task=task)

        result.info["tool_spec"] = agent.get_tool_schemas()

        return result


if __name__ == "__main__":
    from hud.types import Task

    async def test_actor() -> None:
        """Test the actor with a single 2048 task using local hud-browser image."""
        config = Config()
        config.actor.max_parallel_episodes = 1
        config.actor.max_steps_per_episode = 6
        config.actor.episode_timeout_sec = 120
        config.verbose = True

        # Create test task with local hud-browser image
        task_data = {
            "id": "test_2048_128",
            "prompt": "Play the browser-based 2048 game and try to reach the 128 tile. Start by taking a screenshot, then make strategic moves using arrow keys.",  # noqa: E501
            "mcp_config": {
                "local": {
                    "command": "sh",
                    "args": [
                        "-c",
                        "docker run --rm --platform linux/amd64 -i hud-browser:latest 2>/dev/null",
                    ],
                }
            },
            "setup_tool": {"name": "launch_app", "arguments": {"app_name": "2048"}},
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}},
            },
            "agent_config": {
                "system_prompt": "You are an expert 2048 game player. Use arrow keys to reach the target tile. First take a screenshot, then make strategic moves.",  # noqa: E501
            },
        }

        task = Task(**task_data)
        actor = Actor(config)

        logger.info("Testing actor with task: %s", task.id)
        logger.info("Model: %s", config.model.base_model)
        logger.info("VLLM: %s", config.actor.vllm_base_url)

        traces = await actor.run_tasks([task], job_id="test_2048")

        for trace in traces:
            if trace.isError:
                logger.info("Error: %s", trace.content)
            else:
                logger.info("Success!")
                logger.info("Trace info: %s", trace.info if hasattr(trace, "info") else "No info")
                # Check for evaluation in the trace info
                if hasattr(trace, "info") and "evaluation" in trace.info:
                    logger.info("  Evaluation: %s", trace.info["evaluation"])

    asyncio.run(test_actor())
