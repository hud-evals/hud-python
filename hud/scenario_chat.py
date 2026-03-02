"""High-level SDK helper for running a scenario as chat."""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

from hud.environment import Environment
from hud.eval import Task
from hud.eval.manager import run_eval
from hud.tools.types import EvaluationResult

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - import-time optional dependency
    AsyncOpenAI = Any  # type: ignore[misc,assignment]

ScenarioChatApi = Literal["chat_completions", "responses", "auto"]
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_trace_id(trace_id: str) -> str:
    return trace_id.replace("-", "")[:32].ljust(32, "0")


def _make_user_message_span(trace_id: str, message: str) -> dict[str, Any]:
    """Build a telemetry span for a user message."""
    now = _now_iso()
    return {
        "name": "user.message",
        "trace_id": _normalize_trace_id(trace_id),
        "span_id": uuid.uuid4().hex[:16],
        "parent_span_id": None,
        "start_time": now,
        "end_time": now,
        "status_code": "OK",
        "status_message": None,
        "internal_type": "user-message",
        "attributes": {
            "task_run_id": trace_id,
            "category": "agent",
            "type": "CLIENT",
            "request": {"method": "user/message", "params": {"content": message}},
            "start_timestamp": now,
            "end_timestamp": now,
        },
    }


@dataclass
class ScenarioChatResult:
    """Result returned by :func:`run_scenario_chat`."""

    answer: str
    reward: float | None
    evaluation_result: EvaluationResult | None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""


@dataclass
class ScenarioChatTurnResult:
    """Result of a single interactive turn."""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatEvent:
    """A single event yielded by :meth:`ScenarioChatSession.send_stream`.

    Event types:

    - ``text_delta``: Incremental text token from the model.
    - ``tool_call``: A tool invocation (name + args resolved after streaming).
    - ``tool_result``: The result returned by the tool.
    - ``turn_complete``: Signals the turn is done; ``content`` holds the full answer.
    """

    type: Literal["text_delta", "tool_call", "tool_result", "turn_complete"]
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_call_id: str | None = None


class ScenarioChatSession(AbstractAsyncContextManager["ScenarioChatSession"]):
    """Interactive scenario chat session.

    Use this with ``async with`` and call ``send()`` for each user turn.
    Call ``finish()`` when done to submit and evaluate the scenario.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        task: Task,
        api: ScenarioChatApi,
        max_steps: int,
        system_prompt: str | None,
        trace: bool,
        api_key: str | None,
        completion_kwargs: dict[str, Any] | None,
    ) -> None:
        self.client = client
        self.model = model
        self.task = task
        self.api: Literal["chat_completions", "responses"] = (
            "chat_completions" if api == "auto" else api
        )
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.trace = trace
        self.api_key = api_key
        self.completion_kwargs = completion_kwargs or {}

        self.messages: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.last_answer: str = ""
        self.trace_id: str = ""

        self._ctx: Any = None
        self._eval_cm: Any = None
        self._entered = False
        self._finished = False
        self._previous_response_id: str | None = None
        self._scenario_prompt: str = ""

    async def __aenter__(self) -> "ScenarioChatSession":
        self._eval_cm = run_eval(self.task, trace=self.trace, api_key=self.api_key)
        self._ctx = await self._eval_cm.__aenter__()
        self.trace_id = self._ctx.trace_id

        if not self._ctx.prompt:
            raise ValueError(
                f"Scenario '{self.task.scenario}' returned an empty prompt. "
                "Ensure setup yields a non-empty instruction."
            )
        self._scenario_prompt = cast("str", self._ctx.prompt)

        self.system_prompt = self.system_prompt or self._ctx.system_prompt
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.messages.append({"role": "user", "content": self._scenario_prompt})
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        submit_error: Exception | None = None
        if self._entered and not self._finished:
            try:
                await self._ctx.submit(self.last_answer or "")
                self._finished = True
            except Exception as e:
                submit_error = e
        if self._eval_cm is not None:
            await self._eval_cm.__aexit__(exc_type, exc, tb)
        if submit_error is not None:
            raise submit_error
        return False

    async def send(self, message: str) -> ScenarioChatTurnResult:
        """Send one user message and run tool loop until assistant replies."""
        if not self._entered:
            raise RuntimeError("Session is not active. Use 'async with' first.")
        if self._finished:
            raise RuntimeError("Session already finished.")

        if self.api == "responses":
            return await self._send_responses(message)
        return await self._send_chat_completions(message)

    async def finish(self, answer: str | None = None) -> ScenarioChatResult:
        """Submit final answer and evaluate scenario."""
        if not self._entered:
            raise RuntimeError("Session is not active. Use 'async with' first.")
        if self._finished:
            return ScenarioChatResult(
                answer=self.last_answer,
                reward=self._ctx.reward,
                evaluation_result=self._ctx.evaluation_result,
                messages=self.messages,
                tool_calls=self.tool_calls,
                trace_id=self.trace_id,
            )

        final_answer = answer if answer is not None else self.last_answer
        self.last_answer = final_answer
        submit_error: Exception | None = None
        try:
            await self._ctx.submit(final_answer)
            self._finished = True
        except Exception as e:
            submit_error = e
        finally:
            if self._eval_cm is not None:
                # Suppress ContextVar token errors from different async contexts (server sessions)
                with contextlib.suppress(ValueError):
                    await self._eval_cm.__aexit__(None, None, None)
        if submit_error is not None:
            raise submit_error

        return ScenarioChatResult(
            answer=final_answer,
            reward=self._ctx.reward,
            evaluation_result=self._ctx.evaluation_result,
            messages=self.messages,
            tool_calls=self.tool_calls,
            trace_id=self.trace_id,
        )

    def _merged_extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        extra_headers = self.completion_kwargs.get("extra_headers")
        if isinstance(extra_headers, dict):
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        if self.trace_id:
            headers["Trace-Id"] = self.trace_id
        return headers

    def _chat_request_kwargs(
        self, *, messages: list[dict[str, Any]], stream: bool = False
    ) -> dict[str, Any]:
        request_kwargs = dict(self.completion_kwargs)
        request_kwargs.pop("extra_headers", None)
        request_kwargs.update(
            {
                "model": self.model,
                "messages": messages,
                "tools": self._ctx.as_openai_chat_tools(),
            }
        )
        if stream:
            request_kwargs["stream"] = True
        merged_headers = self._merged_extra_headers()
        if merged_headers:
            request_kwargs["extra_headers"] = merged_headers
        return request_kwargs

    async def _send_chat_completions(self, message: str) -> ScenarioChatTurnResult:
        from hud.telemetry.exporter import queue_span

        full_messages = list(self.messages)
        user_msg = {"role": "user", "content": message}
        full_messages.append(user_msg)
        self.messages.append(user_msg)
        tool_calls_this_turn: list[dict[str, Any]] = []

        if self.trace and self.trace_id:
            queue_span(_make_user_message_span(self.trace_id, message))

        for _ in range(self.max_steps):
            response = await self.client.chat.completions.create(
                **self._chat_request_kwargs(messages=full_messages)
            )
            msg = response.choices[0].message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            full_messages.append(assistant_msg)
            self.messages.append(assistant_msg)

            if not msg.tool_calls:
                self.last_answer = msg.content or self.last_answer
                return ScenarioChatTurnResult(
                    answer=msg.content or "",
                    tool_calls=tool_calls_this_turn,
                )

            for tool_call in msg.tool_calls:
                record = {
                    "id": getattr(tool_call, "id", None),
                    "name": getattr(getattr(tool_call, "function", None), "name", None),
                    "arguments": _parse_tool_arguments(
                        getattr(getattr(tool_call, "function", None), "arguments", "{}")
                    ),
                }
                self.tool_calls.append(record)
                tool_calls_this_turn.append(record)
                tool_result = await self._ctx.call_tool(tool_call)
                normalized = _normalize_message(tool_result)
                full_messages.append(normalized)
                self.messages.append(normalized)

        return ScenarioChatTurnResult(answer=self.last_answer, tool_calls=tool_calls_this_turn)

    async def send_stream(self, message: str) -> AsyncIterator[ChatEvent]:
        """Send one user message and yield events as the model responds.

        Yields :class:`ChatEvent` objects:

        - ``text_delta`` for each streamed token
        - ``tool_call`` when a tool invocation is fully resolved
        - ``tool_result`` after tool execution completes
        - ``turn_complete`` when the model finishes its response

        For the ``responses`` API, streaming falls back to yielding the
        complete response as a single ``text_delta`` followed by ``turn_complete``.
        """
        if not self._entered:
            raise RuntimeError("Session is not active. Use 'async with' first.")
        if self._finished:
            raise RuntimeError("Session already finished.")

        if self.api == "responses":
            result = await self._send_responses(message)
            if result.answer:
                yield ChatEvent(type="text_delta", content=result.answer)
            for tc in result.tool_calls:
                yield ChatEvent(
                    type="tool_call",
                    tool_name=tc.get("name"),
                    tool_args=tc.get("arguments") if isinstance(tc.get("arguments"), dict)
                    else None,
                    tool_call_id=tc.get("id"),
                )
            yield ChatEvent(type="turn_complete", content=result.answer)
            return

        async for event in self._stream_chat_completions(message):
            yield event

    async def _stream_chat_completions(self, message: str) -> AsyncIterator[ChatEvent]:
        """Streaming implementation of the chat completions tool loop."""
        from hud.telemetry.exporter import queue_span

        full_messages = list(self.messages)
        user_msg = {"role": "user", "content": message}
        full_messages.append(user_msg)
        self.messages.append(user_msg)
        if self.trace and self.trace_id:
            queue_span(_make_user_message_span(self.trace_id, message))

        for _ in range(self.max_steps):
            response = await self.client.chat.completions.create(
                **self._chat_request_kwargs(messages=full_messages, stream=True)
            )

            content_parts: list[str] = []
            pending_tool_calls: dict[int, dict[str, str]] = {}

            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    content_parts.append(delta.content)
                    yield ChatEvent(type="text_delta", content=delta.content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = pending_tool_calls[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

            content = "".join(content_parts)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}

            if pending_tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": pending_tool_calls[idx]["id"],
                        "type": "function",
                        "function": {
                            "name": pending_tool_calls[idx]["name"],
                            "arguments": pending_tool_calls[idx]["arguments"],
                        },
                    }
                    for idx in sorted(pending_tool_calls)
                ]

            full_messages.append(assistant_msg)
            self.messages.append(assistant_msg)

            if not pending_tool_calls:
                self.last_answer = content or self.last_answer
                yield ChatEvent(type="turn_complete", content=content)
                return

            for idx in sorted(pending_tool_calls):
                tc = pending_tool_calls[idx]
                parsed_args = _parse_tool_arguments(tc["arguments"])
                record = {"id": tc["id"], "name": tc["name"], "arguments": parsed_args}
                self.tool_calls.append(record)

                yield ChatEvent(
                    type="tool_call",
                    tool_name=tc["name"],
                    tool_args=parsed_args if isinstance(parsed_args, dict) else None,
                    tool_call_id=tc["id"],
                )

                openai_tc = {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                tool_result = await self._ctx.call_tool(openai_tc)
                normalized = _normalize_message(tool_result)
                full_messages.append(normalized)
                self.messages.append(normalized)

                result_content = normalized.get("content", str(tool_result))
                yield ChatEvent(
                    type="tool_result",
                    content=result_content,
                    tool_name=tc["name"],
                    tool_call_id=tc["id"],
                )

        yield ChatEvent(type="turn_complete", content=self.last_answer)

    async def _send_responses(self, message: str) -> ScenarioChatTurnResult:
        from hud.telemetry.exporter import queue_span

        self.messages.append({"role": "user", "content": message})
        response_input: Any = message
        tool_calls_this_turn: list[dict[str, Any]] = []
        if not self._previous_response_id:
            response_input = [
                {"role": "user", "content": self._scenario_prompt},
                {"role": "user", "content": message},
            ]

        if self.trace and self.trace_id:
            queue_span(_make_user_message_span(self.trace_id, message))

        for _ in range(self.max_steps):
            request_kwargs: dict[str, Any] = dict(self.completion_kwargs)
            request_kwargs.pop("extra_headers", None)
            request_kwargs.update(
                {
                    "model": self.model,
                    "input": response_input,
                    "tools": self._ctx.as_openai_responses_tools(),
                }
            )
            if self.system_prompt:
                request_kwargs.setdefault("instructions", self.system_prompt)
            if self._previous_response_id:
                request_kwargs["previous_response_id"] = self._previous_response_id

            merged_headers = self._merged_extra_headers()
            if merged_headers:
                request_kwargs["extra_headers"] = merged_headers
            response = await self.client.responses.create(**request_kwargs)
            self._previous_response_id = cast("str | None", getattr(response, "id", None))
            output_text = cast("str", getattr(response, "output_text", "") or "")
            self.messages.append(
                {
                    "role": "assistant",
                    "content": output_text,
                    "response_id": self._previous_response_id,
                }
            )

            function_calls = [
                item
                for item in (getattr(response, "output", []) or [])
                if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                self.last_answer = output_text or self.last_answer
                return ScenarioChatTurnResult(answer=output_text, tool_calls=tool_calls_this_turn)

            tool_outputs: list[Any] = []
            for function_call in function_calls:
                record = {
                    "id": getattr(function_call, "id", None),
                    "name": getattr(function_call, "name", None),
                    "arguments": getattr(function_call, "arguments", {}),
                }
                self.tool_calls.append(record)
                tool_calls_this_turn.append(record)
                tool_output = await self._ctx.call_tool(function_call)
                normalized = _normalize_message(tool_output)
                tool_outputs.append(normalized)
                self.messages.append(normalized)

            response_input = tool_outputs

        return ScenarioChatTurnResult(answer=self.last_answer, tool_calls=tool_calls_this_turn)

    def to_state(self) -> dict[str, Any]:
        """Serialize session state for persistence across HTTP requests.

        Returns a JSON-serializable dict that can be stored and later
        passed to :meth:`from_state` to restore the session.
        """
        return {
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "last_answer": self.last_answer,
            "trace_id": self.trace_id,
            "model": self.model,
            "api": self.api,
            "max_steps": self.max_steps,
            "system_prompt": self.system_prompt,
            "completion_kwargs": self.completion_kwargs,
            "scenario_prompt": self._scenario_prompt,
            "previous_response_id": self._previous_response_id,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        client: AsyncOpenAI,
        ctx: Any,
    ) -> "ScenarioChatSession":
        """Restore a session from a previously serialized state.

        The caller must provide a live ``client`` and ``ctx``
        (an :class:`~hud.eval.context.EvalContext` for the same
        environment/scenario/trace).  Lifecycle of ``ctx`` is managed
        externally -- calling :meth:`finish` will submit via ``ctx``
        but will *not* exit an eval context manager.

        Args:
            state: Dict produced by :meth:`to_state`.
            client: An ``AsyncOpenAI`` (or compatible) client.
            ctx: A live EvalContext bound to the same trace.
        """
        session = object.__new__(cls)
        session.client = client
        session.model = state["model"]
        session.task = None  # type: ignore[assignment]
        session.api = state["api"]
        session.max_steps = state["max_steps"]
        session.system_prompt = state.get("system_prompt")
        session.trace = True
        session.api_key = None
        session.completion_kwargs = state.get("completion_kwargs", {})

        session.messages = list(state["messages"])
        session.tool_calls = list(state["tool_calls"])
        session.last_answer = state["last_answer"]
        session.trace_id = state["trace_id"]

        session._ctx = ctx
        session._eval_cm = None
        session._entered = True
        session._finished = False
        session._previous_response_id = state.get("previous_response_id")
        session._scenario_prompt = state.get("scenario_prompt", "")

        return session


def _resolve_task(
    *,
    task: Task | None,
    env: Environment | None,
    scenario: str | None,
    args: dict[str, Any] | None,
) -> Task:
    if task is not None and (env is not None or scenario is not None or args is not None):
        raise ValueError("Provide either task OR (env, scenario, args), not both")

    if task is None:
        if env is None or scenario is None:
            raise ValueError("When task is not provided, both env and scenario are required")
        task = env(scenario, **(args or {}))

    if not task.scenario:
        raise ValueError("Task must include a scenario to run scenario chat")

    return task


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any] | str:
    try:
        parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        return parsed if isinstance(parsed, dict) else raw_arguments
    except (json.JSONDecodeError, TypeError):
        return raw_arguments


def _normalize_message(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return vars(value)
    return {"content": str(value)}


async def run_scenario_chat(
    *,
    client: AsyncOpenAI,
    model: str,
    task: Task | None = None,
    env: Environment | None = None,
    scenario: str | None = None,
    args: dict[str, Any] | None = None,
    api: ScenarioChatApi = "auto",
    max_steps: int = 20,
    system_prompt: str | None = None,
    trace: bool = True,
    api_key: str | None = None,
    completion_kwargs: dict[str, Any] | None = None,
) -> ScenarioChatResult:
    """Run a scenario as a single chat session and return the evaluated result.

    This is the simplest way to export a scenario as a chat -- one call, done.

    Example:
        ```python
        result = await hud.run_scenario_chat(
            client=client, model="gpt-4o",
            env=env, scenario="my-scenario",
            args={"arg": "value"},
        )
        print(result.reward, result.answer)
        ```
    """
    eval_task = _resolve_task(task=task, env=env, scenario=scenario, args=args)

    async with run_scenario_chat_interactive(
        client=client,
        model=model,
        task=eval_task,
        api=api,
        max_steps=max_steps,
        system_prompt=system_prompt,
        trace=trace,
        api_key=api_key,
        completion_kwargs=completion_kwargs,
    ) as chat:
        turn = await chat.send("Begin.")
        return await chat.finish(turn.answer)


def run_scenario_chat_interactive(
    *,
    client: AsyncOpenAI,
    model: str,
    task: Task | None = None,
    env: Environment | None = None,
    scenario: str | None = None,
    args: dict[str, Any] | None = None,
    api: ScenarioChatApi = "auto",
    max_steps: int = 20,
    system_prompt: str | None = None,
    trace: bool = True,
    api_key: str | None = None,
    completion_kwargs: dict[str, Any] | None = None,
) -> ScenarioChatSession:
    """Create an interactive scenario chat session.

    Example:
        ```python
        async with hud.run_scenario_chat_interactive(...) as chat:
            await chat.send("first turn")
            await chat.send("follow-up")
            result = await chat.finish()
        ```
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be >= 1")
    if api not in ("auto", "chat_completions", "responses"):
        raise ValueError("api must be one of: auto, chat_completions, responses")

    eval_task = _resolve_task(task=task, env=env, scenario=scenario, args=args)
    return ScenarioChatSession(
        client=client,
        model=model,
        task=eval_task,
        api=api,
        max_steps=max_steps,
        system_prompt=system_prompt,
        trace=trace,
        api_key=api_key,
        completion_kwargs=completion_kwargs,
    )


