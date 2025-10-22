import asyncio

import hud
from hud.rl.config import ActorConfig
from hud.rl.logger import console
from hud.types import Task, Trace
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from openai import AsyncOpenAI


class Actor:
    """Collects episodes using GenericOpenAIChatAgent."""

    def __init__(self, config: ActorConfig, client: AsyncOpenAI) -> None:
        self.config = config
        self.client = client

    async def run_tasks(self, tasks: list[Task], job_id: str) -> list[Trace]:
        """Run tasks and collect traces using semaphore for concurrency control with timeout protection."""
        semaphore = asyncio.Semaphore(self.config.max_parallel_episodes)

        async def run_with_semaphore(task: Task) -> Trace:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._run_task(task, job_id),
                        timeout=self.config.episode_timeout_sec,
                    )
                except TimeoutError:
                    console.warning_log(f"Episode timed out for task {task.id}")
                    return Trace(isError=True, content="Episode timeout")
                except Exception as e:
                    console.warning_log(f"Episode error for task {task.id}: {e}")
                    return Trace(isError=True, content=str(e))

        results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=True,
        )

        traces = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                console.warning_log(f"Unexpected error for task {tasks[i].id}: {result}")
                traces.append(Trace(isError=True, content=str(result)))
            else:
                traces.append(result)

        return traces

    async def _run_task(self, task: Task, job_id: str) -> Trace:
        """Run a single task."""
        agent = GenericOpenAIChatAgent(
            openai_client=self.client,
            model_name=self.config.base_model,
            verbose=True if self.config.verbosity >= 1 else False,
            append_setup_output=False,
            completion_kwargs={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_new_tokens,
                "tool_choice": "required" if self.config.force_tool_choice else "auto",
                "logprobs": True,
                "top_p": 1.0,
                "extra_body": {
                    "prompt_logprobs": 0,
                    "top_k": -1,
                    "min_p": 0.0,
                }
            },
        )

        async with hud.async_trace(f"Actor | {task.prompt}", job_id=job_id):
            result = await agent.run(task, max_steps=self.config.max_steps_per_episode)
        result.info["tool_spec"] = agent.get_tool_schemas()
        result.info["temperature"] = self.config.temperature

        return result
