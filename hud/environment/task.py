"""Task: async-generator that yields {"prompt": ...} then {"score": ...}.

A ``Task`` is the in-env challenge definition (formerly "scenario"): an async
generator that yields a prompt for the agent, then — once an answer is sent
back via ``asend`` — yields a score. ``TaskRunner`` drives one task through
its ``start -> evaluate`` lifecycle.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, cast

if TYPE_CHECKING:
    from hud.eval import Variant

    from .env import Environment

TaskFn = Callable[..., AsyncGenerator[dict[str, Any], dict[str, Any]]]

P = ParamSpec("P")


class Task(Generic[P]):
    """A registered challenge — and a typed factory for runnable variants.

    Returned by ``@env.task``. Holds the async-generator ``func`` (prompt -> score),
    identity (``id`` / ``description``), and the owning ``env``. ``TaskRunner`` drives
    ``func`` server-side; calling the ``Task`` with the task's args binds a runnable
    :class:`~hud.client.Variant`, type-checked against the signature via ``ParamSpec``::

        @env.task(id="fix_bug")
        async def fix_bug(difficulty: int = 1, hint: str | None = None): ...

        variant_1 = fix_bug(difficulty=3, hint="line 42")   # -> Variant (type-checked)
        async with variant_1 as run:
            await agent(run)
    """

    def __init__(
        self,
        env: Environment,
        id: str,
        description: str,
        func: Callable[P, AsyncGenerator[dict[str, Any], dict[str, Any]]],
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

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Variant:
        from hud.eval import Variant  # local import: avoid env<->eval cycle

        bound = self._sig.bind(*args, **kwargs)
        return Variant(env=self.env, task=self.id, args=dict(bound.arguments))


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

    Without a declared ``return_type`` the raw evaluate payload is forwarded
    unchanged. With one, the agent's answer is parsed into an ``AgentAnswer[T]``
    (typed ``content`` + citations) — the structured-answer contract.
    """
    if return_type is None:
        return payload
    from pydantic import TypeAdapter

    from hud.agents.types import AgentAnswer, Citation

    raw_text = payload.get("answer", "") if isinstance(payload, dict) else payload
    raw_citations = payload.get("citations", []) if isinstance(payload, dict) else []
    try:
        adapter = TypeAdapter(return_type)
        content = adapter.validate_json(raw_text) if isinstance(raw_text, str) else (
            adapter.validate_python(raw_text)
        )
    except Exception:
        content = raw_text
    citations = [Citation(**c) for c in raw_citations if isinstance(c, dict)]
    return AgentAnswer(
        content=content,
        raw=raw_text if isinstance(raw_text, str) else str(raw_text),
        citations=citations,
    )


def scenario_to_task_fn(scenario_fn: Any) -> Any:
    """Wrap a legacy-style scenario gen (``yield prompt`` then ``yield reward``) as
    a new task gen (``yield {"prompt": ...}`` then ``yield {"score": ...}``).

    Lets ``@env.scenario`` be a thin alias for ``@env.task``: the raw prompt is
    normalized to ``{"prompt": ...}``, the answer is unwrapped from the evaluate
    payload, and a float / ``EvaluationResult`` reward becomes ``{"score": ...}``.
    """

    async def task_fn(**args: Any) -> AsyncGenerator[dict[str, Any], dict[str, Any]]:
        gen = scenario_fn(**args)
        prompt = await gen.__anext__()
        # Pass the prompt through unchanged (str, dict, or a PromptMessage list for
        # chat-style scenarios); only wrap a bare value into the {"prompt": ...} frame.
        if isinstance(prompt, dict) and "prompt" in prompt:
            payload = yield prompt
        else:
            payload = yield {"prompt": prompt}
        answer = payload.get("answer") if isinstance(payload, dict) else payload
        try:
            result = await gen.asend(answer)
        except StopAsyncIteration:
            result = 0.0
        if isinstance(result, dict) and "score" in result:
            yield result
        else:
            score = getattr(result, "reward", result)
            yield {"score": float(score) if isinstance(score, (int, float)) else 0.0}
        with contextlib.suppress(Exception):
            await gen.aclose()

    functools.update_wrapper(task_fn, scenario_fn)
    return task_fn


class TaskRunner:
    """Drives one task through prompt -> evaluate."""

    def __init__(self, task: Task[Any], args: dict[str, Any] | None = None) -> None:
        self.task = task
        self._args = args or {}
        self._gen: AsyncGenerator[dict[str, Any], dict[str, Any]] | None = None

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
        if not isinstance(prompt, dict) or "prompt" not in prompt:
            raise RuntimeError(
                f"task {self.task.id!r}: first yield must be a dict with 'prompt'",
            )
        return cast("dict[str, Any]", _jsonable(prompt))

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._gen is None:
            raise RuntimeError("task not started")
        try:
            evaluation = await self._gen.asend(_build_answer(self.task.return_type, payload))
        except StopAsyncIteration as exc:
            raise RuntimeError(
                f"task {self.task.id!r}: ended without yielding an evaluation",
            ) from exc
        if not isinstance(evaluation, dict) or "score" not in evaluation:
            raise RuntimeError(
                f"task {self.task.id!r}: second yield must be a dict with 'score'",
            )
        with contextlib.suppress(Exception):
            await self._gen.aclose()
        return evaluation

    async def cancel(self) -> None:
        if self._gen is not None:
            with contextlib.suppress(Exception):
                await self._gen.aclose()
            self._gen = None


__all__ = ["Task", "TaskFn", "TaskRunner", "scenario_to_task_fn"]
