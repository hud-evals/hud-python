"""Dataset utilities for working with HuggingFace datasets and Tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from string import Template
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field, field_validator

from hud.agents.misc import ResponseAgent

from .types import MCPToolCall

if TYPE_CHECKING:
    from hud.agents import MCPAgent

logger = logging.getLogger("hud.datasets")


class Task(BaseModel):
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
    system_prompt: str | None = None
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

        Supports ${VAR_NAME} syntax with variable substitution from
        System environment variables (including HUD_API_KEY, etc.)

        Missing variables resolve to empty strings.
        """
        import os

        # Start with current environment variables
        mapping = dict(os.environ)

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


async def fetch_system_prompt_from_dataset(dataset_id: str) -> str | None:
    """
    Fetch system_prompt.txt from a HuggingFace dataset repository.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., "hud-evals/SheetBench-50")

    Returns:
        System prompt text if found, None otherwise
    """
    try:
        # Import here to avoid unnecessary dependency
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        # Try to download the system_prompt.txt file
        try:
            file_path = hf_hub_download(
                repo_id=dataset_id, filename="system_prompt.txt", repo_type="dataset"
            )

            # Read and return the content
            with open(file_path, encoding="utf-8") as f:  # noqa: ASYNC230
                content = f.read().strip()
                if content:
                    logger.info(
                        "Loaded system prompt from %s (length: %d chars)", dataset_id, len(content)
                    )
                    return content
                else:
                    logger.warning("System prompt file is empty in %s", dataset_id)
                    return None

        except EntryNotFoundError:
            logger.debug("No system_prompt.txt found in dataset %s", dataset_id)
            return None

    except ImportError:
        logger.warning(
            "huggingface_hub not installed. Install it to fetch system prompts from datasets."
        )
        return None
    except Exception as e:
        logger.error("Error fetching system prompt from %s: %s", dataset_id, e)
        return None


async def run_dataset(
    name: str,
    dataset: str | Dataset | list[dict[str, Any]],
    agent_class: type[MCPAgent],
    agent_config: dict[str, Any] | None = None,
    max_concurrent: int = 50,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 40,
    split: str = "train",
    auto_respond: bool = False,
    custom_system_prompt: str | None = None,
) -> list[Any]:
    """
    Run all tasks in a dataset with automatic job tracking.

    Args:
        name: Name for the job
        dataset: HuggingFace dataset identifier (e.g. "hud-evals/SheetBench-50"),
                Dataset object, OR list of Task objects
        agent_class: Agent class to instantiate (e.g., ClaudeAgent)
        agent_config: Configuration/kwargs for agent (model, etc.)
        max_concurrent: Maximum parallel task execution
        metadata: Optional metadata for the job
        max_steps: Maximum steps per task
        split: Dataset split to use when loading from string (default: "train")
        auto_respond: Whether to use auto-response agent

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
        >>> dataset = load_dataset("hud-evals/SheetBench-50", split="train")
        >>> results = await run_dataset("my_eval", dataset, ClaudeAgent)
        >>> # Option 3: From list of dicts
        >>> tasks = [{"prompt": "...", "mcp_config": {...}, ...}, ...]
        >>> results = await run_dataset("browser_eval", tasks, ClaudeAgent)
    """
    # Import here to avoid circular imports
    import hud

    dataset_link = None

    # Load dataset from string if needed
    if isinstance(dataset, str):
        logger.info("Loading dataset %s from HuggingFace...", dataset)
        dataset_link = dataset

        # Load dataset from HuggingFace
        dataset = cast("Dataset", load_dataset(dataset, split=split))

    # Create job context
    job_metadata = metadata or {}
    job_metadata["agent_class"] = agent_class.__name__
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
                if "system_prompt" not in task_dict:
                    task_dict["system_prompt"] = custom_system_prompt
                with hud.trace(task_name, job_id=job_obj.id, task_id=task_dict.get("id")):
                    # Convert dict to Task here, at trace level
                    task = Task(**task_dict)

                    agent = agent_class(**(agent_config or {}))

                    if auto_respond:
                        agent.response_agent = (
                            agent_config.get("response_agent")
                            if agent_config
                            else ResponseAgent()
                        )
                    results[index] = await agent.run(task, max_steps=max_steps)

        # Execute all tasks
        await asyncio.gather(
            *[_worker(i, task, max_steps=max_steps) for i, task in enumerate(dataset)],
            return_exceptions=True,  # Don't fail entire batch on one error
        )

    return results


def save_tasks(
    tasks: list[dict[str, Any]], repo_id: str, fields: list[str] | None = None, **kwargs: Any
) -> None:
    """
    Save data to HuggingFace dataset with JSON string serialization.

    Complex fields (dicts, lists) are serialized as JSON strings to maintain clean schema
    and avoid null value pollution in HuggingFace datasets.

    Args:
        tasks: List of dictionaries to save
        repo_id: HuggingFace repository ID (e.g., "hud-evals/my-tasks")
        fields: Optional list of fields to save. If None, saves all fields from each dict.
        **kwargs: Additional arguments passed to dataset.push_to_hub()
    """
    from datasets import Dataset

    # Safety check: Ensure we're not saving Task objects (which have resolved env vars)
    if tasks and isinstance(tasks[0], Task):
        raise ValueError(
            "save_tasks expects dictionaries, not Task objects. "
            "Task objects have resolved environment variables which would expose secrets. "
            "Please pass raw dictionaries with template strings like '${HUD_API_KEY}' preserved."
        )

    # Convert to rows with JSON string fields
    data = []
    for i, tc_dict in enumerate(tasks):
        # Additional safety check for each item
        if isinstance(tc_dict, Task):
            raise ValueError(
                f"Item {i} is a Task object, not a dictionary. "
                "This would expose resolved environment variables. "
                "Please convert to dictionary format with template strings preserved."
            )

        row = {}

        # Determine which fields to process
        fields_to_process = fields if fields is not None else list(tc_dict.keys())

        for field in fields_to_process:
            if field in tc_dict:
                value = tc_dict[field]
                # Serialize complex types as JSON strings
                if isinstance(value, (dict | list)):
                    row[field] = json.dumps(value)
                elif isinstance(value, (str | int | float | bool | type(None))):
                    row[field] = value if value is not None else ""
                else:
                    # For other types, convert to string
                    row[field] = str(value)

        data.append(row)

    # Create and push dataset
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, **kwargs)
