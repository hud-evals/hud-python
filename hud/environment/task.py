"""Environment-side task factories and runners.

The public SDK task model lives in :mod:`hud.eval.task`. This module keeps the
server-side callable returned by ``@env.task`` private: it records the generator
function and builds public ``hud.eval.Task`` objects when called.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, cast

if TYPE_CHECKING:
    from hud.eval import Task as EvalTask

    from .env import Environment

TaskFn = Callable[..., AsyncGenerator[Any, Any]]

P = ParamSpec("P")


class _TaskFactory(Generic[P]):
    """Registered ``@env.task`` callable that creates concrete public tasks.

    ``TaskRunner`` drives its async-generator ``func`` (prompt → score) server-side;
    calling this object with args binds a runnable :class:`~hud.eval.Task`::

        task = fix_bug(difficulty=3)  # -> Task
        async with task as run:
            await agent(run)
    """

    def __init__(
        self,
        env: Environment,
        id: str,
        description: str,
        func: Callable[P, AsyncGenerator[Any, Any]],
        *,
        input: Any = None,
        returns: Any = None,
    ) -> None:
        self.env = env
        self.id = id
        self.description = description
        self.func: TaskFn = func
        #: Type(s) the agent is given as input (a model or union; ``None`` = text).
        self.input_type = input
        #: Type the agent must produce (``None`` = plain text). Drives answer
        #: deserialization into ``AgentAnswer[T]``.
        self.return_type = returns
        self._sig = inspect.signature(func)
        functools.update_wrapper(self, func)

    def manifest_entry(self) -> dict[str, Any]:
        from pydantic import TypeAdapter

        entry: dict[str, Any] = {"id": self.id, "description": self.description}
        for key, typ in (("input", self.input_type), ("returns", self.return_type)):
            if typ is not None:
                with contextlib.suppress(Exception):
                    entry[key] = TypeAdapter(typ).json_schema()
        return entry

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> EvalTask:
        from hud.eval.task import Task  # local import: avoid env<->eval cycle

        bound = self._sig.bind(*args, **kwargs)
        return Task(env=self.env, id=self.id, args=dict(bound.arguments))


def _jsonable(value: Any) -> Any:
    """Recursively convert a prompt payload into JSON-safe primitives.

    The prompt frame may carry rich objects — most importantly a list of
    ``PromptMessage`` (chat-style message prompts) — which must become plain
    dicts/lists before the JSON-RPC framing layer (``json.dumps``) ships them.
    """
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce_args(func: TaskFn, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce string wire args into the task fn's annotated param types.

    JSON-RPC sends args as JSON scalars/strings; a param annotated with a richer
    type (Pydantic model, list, etc.) is validated via a ``TypeAdapter``. Values
    that already match (or fail to coerce) are passed through unchanged.
    """
    from pydantic import TypeAdapter

    hints = inspect.signature(func).parameters
    coerced: dict[str, Any] = {}
    for name, value in args.items():
        param = hints.get(name)
        annotation = param.annotation if param is not None else inspect.Parameter.empty
        if annotation in (inspect.Parameter.empty, str, Any) or not isinstance(value, str):
            coerced[name] = value
            continue
        try:
            coerced[name] = TypeAdapter(annotation).validate_json(value)
        except Exception:
            coerced[name] = value
    return coerced


def _build_answer(return_type: Any, payload: dict[str, Any]) -> Any:
    """Build the value sent into the task gen for evaluation.

    Without a declared ``return_type`` the answer value is forwarded unchanged.
    With one, the agent's answer is parsed into an ``AgentAnswer[T]``
    (typed ``content`` + citations) — the structured-answer contract.
    """
    if return_type is None:
        return payload.get("answer") if isinstance(payload, dict) else payload
    from pydantic import TypeAdapter

    from hud.agents.types import AgentAnswer, Citation

    raw_text = payload.get("answer", "") if isinstance(payload, dict) else payload
    raw_citations = payload.get("citations", []) if isinstance(payload, dict) else []
    try:
        adapter = TypeAdapter(return_type)
        content = (
            adapter.validate_json(raw_text)
            if isinstance(raw_text, str)
            else (adapter.validate_python(raw_text))
        )
    except Exception:
        content = raw_text
    citations = [Citation(**c) for c in raw_citations if isinstance(c, dict)]
    return AgentAnswer(
        content=content,
        raw=raw_text if isinstance(raw_text, str) else str(raw_text),
        citations=citations,
    )


class TaskRunner:
    """Drives one task through prompt -> grade."""

    def __init__(self, task: _TaskFactory[Any], args: dict[str, Any] | None = None) -> None:
        self.task = task
        self._args = args or {}
        self._gen: AsyncGenerator[Any, Any] | None = None

        # Fail fast on bad args (TypeError before any side-effects run).
        try:
            inspect.signature(task.func).bind(**self._args)
        except TypeError as exc:
            raise TypeError(
                f"task {task.id!r}: bad args {sorted(self._args)}: {exc}",
            ) from exc

    async def start(self) -> dict[str, Any]:
        self._gen = self.task.func(**_coerce_args(self.task.func, self._args))
        prompt = await self._gen.__anext__()
        frame = prompt if isinstance(prompt, dict) and "prompt" in prompt else {"prompt": prompt}
        return cast("dict[str, Any]", _jsonable(frame))

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._gen is None:
            raise RuntimeError("task not started")
        try:
            evaluation = await self._gen.asend(_build_answer(self.task.return_type, payload))
        except StopAsyncIteration:
            evaluation = 0.0
        frame = (
            evaluation
            if isinstance(evaluation, dict) and "score" in evaluation
            else {"score": _score_value(evaluation)}
        )
        with contextlib.suppress(Exception):
            await self._gen.aclose()
        return frame

    async def cancel(self) -> None:
        if self._gen is not None:
            with contextlib.suppress(Exception):
                await self._gen.aclose()
            self._gen = None


def _score_value(result: Any) -> float:
    score = getattr(result, "reward", result)
    return float(score) if isinstance(score, (int, float)) else 0.0


__all__ = ["TaskFn", "TaskRunner"]
