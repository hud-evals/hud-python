"""One session of an environment: the task it holds suspended between start and grade."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import secrets
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from hud.graders import EvaluationResult
from hud.graders.base import Grade, grade_problems

from .env import Answer, current_session_id

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Sequence

    from hud.capabilities.base import Capability
    from hud.capabilities.connection import Connection

    from .egress import WorkspaceRoute
    from .env import Environment, TaskTemplate
    from .file_tracker import FileTrackingOp


def validate_session_id(session_id: str) -> None:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("session id must be a single path component")


SESSION_STATE_ROOT = PurePosixPath("/runtime/sessions")


def session_state_dir(session_id: str) -> PurePosixPath:
    """Where ``session_id`` keeps, inside its host, the state it hands to a verifier
    placed elsewhere. Environments write it; providers snapshot and restore it."""
    validate_session_id(session_id)
    return SESSION_STATE_ROOT / session_id


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce_args(sig: inspect.Signature, args: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for name, value in args.items():
        param = sig.parameters.get(name)
        annotation = param.annotation if param is not None else inspect.Parameter.empty
        if annotation in (inspect.Parameter.empty, Any):
            coerced[name] = value
            continue
        adapter = TypeAdapter(annotation)
        try:
            coerced[name] = adapter.validate_python(value)
        except ValidationError:
            if not isinstance(value, str):
                coerced[name] = value
                continue
            try:
                coerced[name] = adapter.validate_json(value)
            except ValidationError:
                coerced[name] = value
    return coerced


def _build_answer(return_type: Any, payload: dict[str, Any]) -> Any:
    if return_type is None:
        return payload.get("answer")

    raw_text = payload.get("answer", "")
    adapter = TypeAdapter(return_type)
    try:
        content = (
            adapter.validate_json(raw_text)
            if isinstance(raw_text, str)
            else adapter.validate_python(raw_text)
        )
    except ValidationError:
        content = raw_text  # an unparseable answer still reaches the task to grade
    return Answer(
        content=content,
        raw=raw_text if isinstance(raw_text, str) else str(raw_text),
    )


class Session:
    """One session of an environment and the task it holds suspended between start and grade.

    Task code, teardown hooks included, runs with ``current_session_id`` set to the
    session's id: environment code keys per-session state, such as robot slot claims, on
    it. The session holds its task only while the task is suspended, so ``cancel`` never
    touches a task that is running.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.session_id = "sess-" + secrets.token_hex(4)
        self._held: tuple[TaskTemplate[Any], AsyncGenerator[Any, Any]] | None = None

    @property
    def capabilities(self) -> list[Capability]:
        """The environment's capabilities, at the addresses it published."""
        return list(self.env.capabilities)

    @property
    def has_task(self) -> bool:
        """Whether a started task is suspended here, waiting to be graded."""
        return self._held is not None

    async def bind(
        self,
        connections: Sequence[Connection],
        *,
        workspace_routes: Sequence[WorkspaceRoute] | None = None,
    ) -> None:
        """Bind ``connections`` in place of the ones the session holds, and its workspace
        routes when given."""
        await self.env.bind_connections(connections, session=self.session_id)
        if workspace_routes is not None:
            await self.env.bind_workspace_routes(workspace_routes, session=self.session_id)

    async def start_task(self, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """Start ``task_id``, replacing any task held here, and return its prompt frame.

        A start that fails tears its task down.
        """
        await self.cancel()
        task = self.env.tasks[task_id]
        try:
            task.sig.bind(**args)
        except TypeError as exc:
            raise TypeError(f"task {task.id!r}: bad args {sorted(args)}: {exc}") from exc
        gen: AsyncGenerator[Any, Any] | None = None
        with self._current():
            try:
                gen = task.func(**_coerce_args(task.sig, args))
                prompt = await gen.__anext__()
                is_frame = isinstance(prompt, dict) and "prompt" in prompt
                frame = _jsonable(prompt if is_frame else {"prompt": prompt})
            except BaseException:
                await self._teardown(gen)
                raise
        self._held = (task, gen)
        return cast("dict[str, Any]", frame)

    async def grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``payload``'s answer to the task held here and return its grade frame."""
        held, self._held = self._held, None
        if held is None:
            raise RuntimeError("no task in progress")
        task, gen = held
        with self._current():
            try:
                evaluation = await gen.asend(_build_answer(task.return_type, payload))
            except StopAsyncIteration:
                evaluation = 0.0
            finally:
                await self._teardown(gen)
        if isinstance(evaluation, (int, float)):
            evaluation = {"score": float(evaluation)}
        elif isinstance(evaluation, EvaluationResult):
            evaluation = {**evaluation.model_dump(exclude={"reward"}), "score": evaluation.reward}
        if isinstance(evaluation, dict) and not isinstance(evaluation.get("score"), (int, float)):
            raise TypeError(
                f"task {task.id!r} graded with a dict missing a numeric "
                f"'score' (keys: {sorted(evaluation)})"
            )
        try:
            grade = Grade.model_validate(_jsonable(evaluation), by_name=False)
        except ValidationError as exc:
            raise ValueError(f"task {task.id!r} graded with {grade_problems(exc)}") from exc
        return grade.model_dump(mode="json")

    async def cancel(self) -> None:
        """Tear down the task held here, if any."""
        held, self._held = self._held, None
        if held is not None:
            with self._current():
                await self._teardown(held[1])

    async def close(self) -> None:
        """End the session: tear down any task it holds and release what ``bind`` installed."""
        await self.cancel()
        self.env.release_session(self.session_id)

    async def files(self, op: FileTrackingOp) -> dict[str, Any] | None:
        tracker = self.env.file_tracker
        return None if tracker is None else await asyncio.to_thread(tracker.call, op)

    async def revoke(self) -> None:
        """Nothing to cut: in this process the agent reaches the capabilities directly."""

    @contextlib.contextmanager
    def _current(self) -> Iterator[None]:
        token = current_session_id.set(self.session_id)
        try:
            yield
        finally:
            current_session_id.reset(token)

    async def _teardown(self, gen: AsyncGenerator[Any, Any] | None) -> None:
        if gen is not None:
            with contextlib.suppress(Exception):
                await gen.aclose()
        await self.env.end_task()
