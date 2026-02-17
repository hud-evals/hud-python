"""Rollout collection utilities built on top of run_dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from hud.datasets import load_tasks, run_dataset
from hud.types import Trace

from .schema import RolloutRecord, make_rollout_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.types import AgentType


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _source_with_split(source: str, split: str) -> str:
    path = Path(source)
    if path.exists() or ":" in source or "/" not in source:
        return source
    if split != "train":
        return f"{source}:{split}"
    return source


def _load_raw_tasks(source: str | Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if isinstance(source, str):
        loaded = load_tasks(_source_with_split(source, split), raw=True)
        return cast("list[dict[str, Any]]", loaded)

    tasks: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            raise TypeError(f"Expected task dict, got {type(item)}")
        tasks.append(cast("dict[str, Any]", _to_jsonable(item)))
    return tasks


def _coerce_trace(result: Any) -> Trace:
    if isinstance(result, Trace):
        return result
    if result is None:
        return Trace(isError=True, content="No trace returned", reward=0.0, done=True)
    if isinstance(result, dict):
        try:
            return Trace.model_validate(result)
        except Exception:
            payload = json.dumps(_to_jsonable(result), ensure_ascii=False)
            return Trace(isError=True, content=payload, reward=0.0, done=True)

    if isinstance(result, BaseModel):
        try:
            return Trace.model_validate(result.model_dump(mode="json"))
        except Exception:
            return Trace(isError=True, content=str(result), reward=0.0, done=True)

    if hasattr(result, "trace_id") and hasattr(result, "reward"):
        error_obj = getattr(result, "error", None)
        info: dict[str, Any] = {}
        for key in ("trace_id", "job_id", "group_id", "eval_name"):
            value = getattr(result, key, None)
            if value is not None:
                info[key] = value
        if error_obj is not None:
            info["error"] = str(error_obj)

        return Trace(
            reward=float(getattr(result, "reward", 0.0) or 0.0),
            done=True,
            isError=error_obj is not None,
            content=getattr(result, "answer", None),
            info=info,
        )

    return Trace(isError=True, content=str(result), reward=0.0, done=True)


def _expand_tasks(tasks: Sequence[dict[str, Any]], group_size: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for task in tasks:
        expanded.extend(dict(task) for _ in range(group_size))
    return expanded


def build_rollout_records(
    *,
    source: str,
    tasks: Sequence[dict[str, Any]],
    results: Sequence[Any],
    group_size: int = 1,
) -> list[RolloutRecord]:
    """Convert run_dataset results into rollout records."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    records: list[RolloutRecord] = []
    expected = len(tasks) * group_size
    bounded = min(len(results), expected)

    for result_index in range(bounded):
        task_index = result_index // group_size
        repeat_index = result_index % group_size
        task_dict = dict(tasks[task_index])
        prompt = str(task_dict.get("prompt") or f"Task {task_index}")
        trace = _coerce_trace(results[result_index])
        task_id_raw = task_dict.get("id")

        records.append(
            RolloutRecord(
                rollout_id=make_rollout_id(source, task_index, repeat_index, prompt),
                source=source,
                task_index=task_index,
                repeat_index=repeat_index,
                task_id=str(task_id_raw) if task_id_raw is not None else None,
                prompt=prompt,
                reward=trace.reward,
                done=trace.done,
                is_error=trace.isError,
                content=trace.content,
                info=cast("dict[str, Any]", _to_jsonable(trace.info)),
                task=cast("dict[str, Any]", _to_jsonable(task_dict)),
                trace=[_to_jsonable(step) for step in trace.trace],
                messages=_to_jsonable(trace.messages),
            )
        )

    for result_index in range(bounded, len(results)):
        trace = _coerce_trace(results[result_index])
        fallback_prompt = "Unknown task"
        records.append(
            RolloutRecord(
                rollout_id=make_rollout_id(source, -1, result_index - bounded, fallback_prompt),
                source=source,
                task_index=-1,
                repeat_index=result_index - bounded,
                prompt=fallback_prompt,
                reward=trace.reward,
                done=trace.done,
                is_error=trace.isError,
                content=trace.content,
                info=cast("dict[str, Any]", _to_jsonable(trace.info)),
                task={},
                trace=[_to_jsonable(step) for step in trace.trace],
                messages=_to_jsonable(trace.messages),
            )
        )

    return records


async def collect_rollouts(
    *,
    name: str,
    source: str | Sequence[dict[str, Any]],
    agent_type: AgentType | str,
    agent_params: dict[str, Any] | None = None,
    max_concurrent: int = 30,
    metadata: dict[str, Any] | None = None,
    max_steps: int = 10,
    split: str = "train",
    group_size: int = 1,
    auto_respond: bool = True,
) -> list[RolloutRecord]:
    """Collect rollouts by executing tasks with run_dataset."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    raw_tasks = _load_raw_tasks(source, split=split)
    if not raw_tasks:
        return []

    expanded_tasks = _expand_tasks(raw_tasks, group_size=group_size)
    source_name = source if isinstance(source, str) else name
    results = await run_dataset(
        expanded_tasks,
        agent_type,
        agent_params=agent_params,
        group_size=1,
        quiet=True,
        max_concurrent=max_concurrent,
        max_steps=max_steps,
        taskset=metadata["taskset"] if metadata and "taskset" in metadata else None,
    )

    return build_rollout_records(
        source=source_name,
        tasks=raw_tasks,
        results=results,
        group_size=group_size,
    )


def write_rollouts_jsonl(records: Sequence[RolloutRecord], output_path: str | Path) -> Path:
    """Write rollout records to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
    return path
