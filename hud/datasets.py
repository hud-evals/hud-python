"""Dataset utilities for working with HuggingFace datasets and TaskConfigs."""

from __future__ import annotations

import asyncio
import json
import logging
from string import Template
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset
from pydantic import BaseModel, Field, field_validator

from .types import MCPToolCall

if TYPE_CHECKING:
    from datasets import Dataset

    from hud.agent import MCPAgent

logger = logging.getLogger("hud.datasets")


class TaskConfig(BaseModel):
    """
    A task configuration that can be used to create a task.

    The mcp_config field supports environment variable substitution using
    template placeholders in the format ${VAR_NAME} or ${VAR_NAME:default_value}.

    Example:
        mcp_config: {
            "hud": {
                "url": "${HUD_MCP_URL:https://mcp.hud.so/v3/mcp}",
                "headers": {
                    "Authorization": "Bearer ${HUD_API_KEY}",
                    "Run-Id": "${RUN_ID}",
                    "Mcp-Image": "your-mcp-image"
                }
            }
        }
    """

    id: str | None = None
    prompt: str
    mcp_config: dict[str, Any]
    setup_tool: MCPToolCall | list[MCPToolCall] | None = None
    evaluate_tool: MCPToolCall | list[MCPToolCall] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mcp_config", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        Automatically resolve environment variables in mcp_config using Template.

        Supports ${VAR_NAME} syntax with variable substitution from:
        1. System environment variables (including HUD_API_KEY, etc.)
        2. Runtime context variables (e.g., RUN_ID from telemetry context)

        Missing variables resolve to empty strings.
        """
        import os

        from hud.otel import get_current_task_run_id

        # Start with current environment variables
        mapping = dict(os.environ)

        # Add runtime context variables if available
        run_id = get_current_task_run_id()
        if run_id:
            mapping["RUN_ID"] = run_id

        def substitute_in_value(obj: Any) -> Any:
            """Recursively substitute variables in nested structures."""
            if isinstance(obj, str):
                # Use Template's substitute with defaultdict - missing vars become empty strings
                from collections import defaultdict

                safe_mapping = defaultdict(str, mapping)
                return Template(obj).substitute(safe_mapping)
            elif isinstance(obj, dict):
                return {k: substitute_in_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_in_value(item) for item in obj]
            else:
                return obj

        return substitute_in_value(v)


def to_taskconfigs(dataset: Dataset) -> list[TaskConfig]:
    """
    Convert a HuggingFace dataset to TaskConfig objects.

    The dataset should have complex fields (mcp_config, setup_tool, etc.)
    stored as JSON strings to avoid null value pollution.

    Environment variables are resolved during TaskConfig instantiation.

    Args:
        dataset: HuggingFace Dataset with JSON string fields

    Returns:
        List of TaskConfig objects with env vars resolved

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("hud-evals/browser-taskconfigs", split="train")
        >>> tasks = to_taskconfigs(dataset)
        >>> tasks[0].mcp_config  # Env vars like ${HUD_API_KEY} are resolved
    """
    try:
        tasks = []
        for row in dataset:
            # Cast row to dict for type checker
            row = cast("dict[str, Any]", row)
            # Build TaskConfig dict, parsing JSON string fields
            tc_dict: dict[str, Any] = {
                "prompt": row["prompt"],
                "mcp_config": json.loads(row["mcp_config"]),
            }

            # Optional fields
            if row.get("id"):
                tc_dict["id"] = row["id"]

            if row.get("metadata"):
                tc_dict["metadata"] = json.loads(row["metadata"])

            if row.get("setup_tool"):
                tc_dict["setup_tool"] = json.loads(row["setup_tool"])

            if row.get("evaluate_tool"):
                tc_dict["evaluate_tool"] = json.loads(row["evaluate_tool"])

            # Create TaskConfig (triggers env var resolution)
            tasks.append(TaskConfig(**tc_dict))

        return tasks
    except TypeError as e:
        raise ValueError("Dataset must be a train or test HF split") from e
    except Exception as e:
        raise e


async def run_dataset(
    name: str,
    dataset: Dataset | list[TaskConfig],
    agent_class: type[MCPAgent],
    agent_config: dict[str, Any] | None = None,
    max_concurrent: int = 5,
    metadata: dict[str, Any] | None = None,
) -> list[Any]:
    """
    Run all tasks in a dataset with automatic job tracking.

    Args:
        name: Name for the job
        dataset: HuggingFace Dataset with task data OR list of TaskConfig objects
        agent_class: Agent class to instantiate (e.g., ClaudeMCPAgent)
        agent_config: Configuration for agent (model, etc.)
        max_concurrent: Maximum parallel task execution
        metadata: Optional metadata for the job

    Returns:
        List of results from agent.run() in dataset order

    Example:
        >>> from datasets import load_dataset
        >>> from hud.mcp import ClaudeMCPAgent
        >>> # Option 1: From HuggingFace dataset with JSON string fields
        >>> dataset = load_dataset("hud-evals/browser-taskconfigs", split="train")
        >>> tasks = to_taskconfigs(dataset)
        >>> results = await run_dataset(
        ...     "browser_eval",
        ...     tasks,
        ...     ClaudeMCPAgent,
        ...     {"model": "claude-3-5-sonnet-20241022"},
        ...     max_concurrent=3,
        ... )
        >>> # Option 2: Direct from loaded dataset
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("hud-evals/browser-taskconfigs", split="train")
        >>> results = await run_dataset("my_eval", dataset, ClaudeMCPAgent)
    """
    # Import here to avoid circular imports
    import hud
    from hud.client import MCPClient

    # Convert dataset to TaskConfigs if needed
    tasks = dataset if isinstance(dataset, list) else to_taskconfigs(dataset)

    # Create job context
    job_metadata = metadata or {}
    job_metadata["agent_class"] = agent_class.__name__
    if agent_config:
        job_metadata["agent_config"] = agent_config

    with hud.job(name, metadata=job_metadata) as job_obj:
        # Run tasks with semaphore for concurrency control
        sem = asyncio.Semaphore(max_concurrent)
        results: list[Any | None] = [None] * len(tasks)

        async def _worker(index: int, task: TaskConfig) -> None:
            async with sem:
                # Create trace for this task
                with hud.trace(f"task_{index}", job_id=job_obj.id):
                    # Create fresh MCP client per task
                    if task.mcp_config:
                        client = MCPClient(mcp_config=task.mcp_config)
                        agent = agent_class(mcp_client=client, **(agent_config or {}))

                        try:
                            results[index] = await agent.run(task)
                        finally:
                            await client.close()
                    else:
                        logger.warning("Task %d has no mcp_config defined", index)
                        results[index] = None

        # Execute all tasks
        await asyncio.gather(
            *[_worker(i, task) for i, task in enumerate(tasks)],
            return_exceptions=True,  # Don't fail entire batch on one error
        )

    return results


def save_taskconfigs(taskconfigs: list[dict[str, Any]], repo_id: str, **kwargs: Any) -> None:
    """
    Save TaskConfigs to HuggingFace dataset with JSON string serialization.

    Complex fields are serialized as JSON strings to maintain clean schema
    and avoid null value pollution in HuggingFace datasets.

    Args:
        taskconfigs: List of TaskConfig dicts (NOT TaskConfig objects, to preserve templates)
        repo_id: HuggingFace repository ID (e.g., "hud-evals/my-tasks")
        **kwargs: Additional arguments passed to dataset.push_to_hub()
    """
    from datasets import Dataset

    # Safety check: Ensure we're not saving TaskConfig objects (which have resolved env vars)
    if taskconfigs and isinstance(taskconfigs[0], TaskConfig):
        raise ValueError(
            "save_taskconfigs expects dictionaries, not TaskConfig objects. "
            "TaskConfig objects have resolved environment variables which would expose secrets. "
            "Please pass raw dictionaries with template strings like '${HUD_API_KEY}' preserved."
        )

    # Convert to rows with JSON string fields
    data = []
    for i, tc_dict in enumerate(taskconfigs):
        # Additional safety check for each item
        if isinstance(tc_dict, TaskConfig):
            raise ValueError(
                f"Item {i} is a TaskConfig object, not a dictionary. "
                "This would expose resolved environment variables. "
                "Please convert to dictionary format with template strings preserved."
            )
        row = {
            "prompt": tc_dict["prompt"],
            "mcp_config": json.dumps(tc_dict["mcp_config"]),
        }

        if tc_dict.get("id"):
            row["id"] = tc_dict["id"]

        if tc_dict.get("metadata"):
            row["metadata"] = json.dumps(tc_dict["metadata"])

        if tc_dict.get("setup_tool"):
            row["setup_tool"] = json.dumps(tc_dict["setup_tool"])

        if tc_dict.get("evaluate_tool"):
            row["evaluate_tool"] = json.dumps(tc_dict["evaluate_tool"])

        data.append(row)

    # Create and push dataset
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, **kwargs)
