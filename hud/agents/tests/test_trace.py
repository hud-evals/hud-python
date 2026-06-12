"""Tool-agent family layer tests: the flat ``Step`` subclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from mcp import types as mcp_types

from hud.agents.types import (
    AgentStep,
    Citation,
    Sample,
    SubagentStep,
    ToolStep,
    Usage,
)
from hud.telemetry.context import set_trace_context
from hud.types import MCPToolCall, MCPToolResult, Step, Trace


def test_agent_step_raw_serializes_safely():
    """AgentStep captures raw provider payloads in JSON-safe dumps."""

    @dataclass
    class RawResponse:
        raw_data: str

    step = AgentStep(raw=RawResponse(raw_data="value"))
    data = step.model_dump(mode="json")

    assert step.raw == RawResponse(raw_data="value")
    assert data["raw"] == {"raw_data": "value"}


def test_agent_step_dump_uses_canonical_field_names():
    """AgentStep dumps use the normalized SDK field names, flat on the step."""
    step = AgentStep(raw={"raw_data": "value"})
    step.reasoning = "because"
    step.citations = [Citation(source="https://example.com")]

    data = step.model_dump(exclude_none=True, mode="json")

    assert data["source"] == "agent"
    assert data["reasoning"] == "because"
    assert data["citations"] == [{"type": "citation", "text": "", "source": "https://example.com"}]
    assert data["raw"] == {"raw_data": "value"}


def test_agent_step_citations_roundtrip():
    """Citations survive serialize/deserialize as typed Citations."""
    cit = Citation(type="url_citation", source="https://example.com", title="Example")
    step = AgentStep(content="hello", citations=[cit])
    data = step.model_dump(mode="json")
    restored = AgentStep(**data)
    assert restored.citations == [cit]
    assert restored.citations[0].source == "https://example.com"


def test_citation_defaults():
    """Citation defaults to a generic, empty-span citation."""
    c = Citation()
    assert c.type == "citation"
    assert c.text == ""
    assert c.start_index is None


def test_final_query_reads_the_final_agent_turns_citations():
    """The chat-surface read: trace.final() with family vocabulary at the call site."""

    def reply_citations(step: Step) -> list[Citation] | None:
        return step.citations if isinstance(step, AgentStep) else None

    cited = Citation(source="https://example.com")
    trace = Trace()
    trace.record(AgentStep(content="draft", citations=[cited]))
    trace.record(ToolStep())
    assert trace.final(reply_citations) == [cited]

    trace.record(AgentStep(content="final", done=True))
    assert trace.final(reply_citations) == []


def test_agent_step_timing_and_error_live_on_the_skeleton():
    """The turn uses the skeleton's channels: record() stamps ended_at, error is error."""
    trace = Trace()
    step = AgentStep(error="rate limited", done=True)
    trace.record(step)

    assert step.ended_at is not None
    assert trace.error == "rate limited"


def test_trace_dump_keeps_family_payloads():
    """Family subclass fields survive a whole-trace dump (SerializeAsAny)."""
    trace = Trace()
    trace.record(
        AgentStep(
            content="answer",
            usage=Usage(prompt_tokens=10, completion_tokens=3),
            sample=Sample(prompt_token_ids=[1], output_token_ids=[2], output_logprobs=[-0.5]),
        )
    )
    trace.record(SubagentStep(subagent=Trace(content="inner")))

    dump = trace.model_dump(mode="json", exclude_none=True)
    assert dump["steps"][0]["content"] == "answer"
    assert dump["steps"][0]["usage"]["prompt_tokens"] == 10
    assert dump["steps"][0]["sample"]["output_token_ids"] == [2]
    assert dump["steps"][1]["subagent"]["content"] == "inner"


def test_step_emit_carries_tool_call_and_result():
    captured: list[dict[str, Any]] = []
    call = MCPToolCall(name="bash", arguments={"command": "ls"})
    result = MCPToolResult(
        content=[mcp_types.TextContent(type="text", text="file.txt")],
        isError=False,
    )

    with (
        patch("hud.types.queue_span", side_effect=captured.append),
        set_trace_context("run-1"),
    ):
        ToolStep(call=call, result=result).emit()

    (span,) = captured
    assert span["name"] == "step.tool"
    assert span["attributes"]["hud.schema"] == "hud.step.v1"
    payload = span["attributes"]["hud.payload"]
    assert payload["call"]["name"] == "bash"
    assert payload["result"]["content"][0]["text"] == "file.txt"
    assert payload["result"]["isError"] is False
