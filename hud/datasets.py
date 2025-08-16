"""Dataset utilities for working with HuggingFace datasets and TaskConfigs."""

from __future__ import annotations

import asyncio
import json
import logging
from string import Template
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field, field_validator

from .types import MCPToolCall

if TYPE_CHECKING:
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

    @field_validator("mcp_config", "metadata", mode="before")
    @classmethod
    def parse_json_strings(cls, v: Any) -> Any:
        """Parse JSON strings into dictionaries."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}") from e
        return v

    @field_validator("setup_tool", "evaluate_tool", mode="before")
    @classmethod
    def convert_dict_to_tool_call(cls, v: Any) -> Any:
        """Convert dict to MCPToolCall instance, parsing JSON strings first."""
        if v is None:
            return None

        # Parse JSON string if needed
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}") from e

        if isinstance(v, dict):
            return MCPToolCall(**v)
        if isinstance(v, list):
            return [MCPToolCall(**item) if isinstance(item, dict) else item for item in v]
        return v

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


async def run_dataset(
    name: str,
    dataset: str | Dataset | list[dict[str, Any]],
    agent_class: type[MCPAgent],
    agent_config: dict[str, Any] | None = None,
    max_concurrent: int = 50,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 40,
    split: str = "train",
) -> list[Any]:
    """
    Run all tasks in a dataset with automatic job tracking.

    Args:
        name: Name for the job
        dataset: HuggingFace dataset identifier (e.g. "hud-evals/SheetBench-50"),
                Dataset object, OR list of TaskConfig objects
        agent_class: Agent class to instantiate (e.g., ClaudeAgent)
        agent_config: Configuration for agent (model, etc.)
        max_concurrent: Maximum parallel task execution
        metadata: Optional metadata for the job
        max_steps: Maximum steps per task
        split: Dataset split to use when loading from string (default: "train")

    Returns:
        List of results from agent.run() in dataset order

    Example:
        >>> from hud.agents import ClaudeAgent
        >>> # Option 1: From dataset string identifier
        >>> results = await run_dataset(
        ...     "SheetBench Eval",
        ...     "hud-evals/SheetBench-50",
        ...     ClaudeAgent,
        ...     {"model": "claude-3-5-sonnet-20241022"},
        ... )
        >>> # Option 2: From HuggingFace dataset object
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("hud-evals/browser-taskconfigs", split="train")
        >>> results = await run_dataset("my_eval", dataset, ClaudeAgent)
        >>> # Option 3: From list of dicts
        >>> tasks = [{"prompt": "...", "mcp_config": {...}, ...}, ...]
        >>> results = await run_dataset("browser_eval", tasks, ClaudeAgent)
    """
    # Import here to avoid circular imports
    import hud
    from hud.client import MCPClient

    dataset_link = None

    # Load dataset from string if needed
    if isinstance(dataset, str):
        logger.info("Loading dataset %s from HuggingFace...", dataset)
        dataset_link = dataset
        dataset = cast("Dataset", load_dataset(dataset, split=split))

    # Create job context
    job_metadata = metadata or {}
    job_metadata["agent_class"] = agent_class.__name__
    if agent_config:
        job_metadata["agent_config"] = agent_config

    # Extract dataset verification info if available
    if isinstance(dataset, Dataset) and not dataset_link:
        general_info = next(iter(dataset.info.__dict__["download_checksums"].keys())).split("/")
        project = general_info[3]
        dataset_name = general_info[4].split("@")[0]
        dataset_link = f"{project}/{dataset_name}"

    with hud.job(name, metadata=job_metadata, dataset_link=dataset_link) as job_obj:
        # Run tasks with semaphore for concurrency control
        sem = asyncio.Semaphore(max_concurrent)
        results: list[Any | None] = [None] * len(dataset)

        async def _worker(index: int, task_dict: Any, max_steps: int = 40) -> None:
            async with sem:
                # Create trace for this task
                task_name = task_dict.get("prompt") or f"Task {index}"
                with hud.trace(task_name, job_id=job_obj.id, task_id=task_dict.get("id")):
                    # Convert dict to TaskConfig here, at trace level
                    task = TaskConfig(**task_dict)

                    # Create fresh MCP client per task
                    if task.mcp_config:
                        client = MCPClient(mcp_config=task.mcp_config)
                        agent = agent_class(mcp_client=client, **(agent_config or {}))

                        try:
                            results[index] = await agent.run(task, max_steps=max_steps)
                        finally:
                            await client.close()
                    else:
                        logger.warning("Task %d has no mcp_config defined", index)
                        results[index] = None

        # Execute all tasks
        await asyncio.gather(
            *[_worker(i, task, max_steps=max_steps) for i, task in enumerate(dataset)],
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
