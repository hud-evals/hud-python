"""Claude CLI stream translation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import mcp.types as mcp_types
from anthropic.types.beta import BetaMessage

from hud.agents.claude.agent import ClaudeAgent
from hud.agents.types import ToolStep
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.eval.run import Run


class ClaudeEvents:
    """Translate Claude CLI stream messages into canonical HUD steps."""

    def __init__(self, run: Run, *, started_at: str) -> None:
        self.run = run
        self.agent_started_at = started_at
        self.pending_calls: dict[str, tuple[MCPToolCall, str]] = {}
        self.saw_result = False
        self.error: str | None = None

    def consume(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("Claude stream event must be an object")
        received_at = now_iso()
        match message.get("type"):
            case "system" if message.get("subtype") == "init":
                self.agent_started_at = received_at
            case "assistant":
                step = ClaudeAgent.message_to_agent_step(
                    BetaMessage.model_validate(message["message"])
                )
                step.started_at = self.agent_started_at
                step.ended_at = received_at
                if step.content:
                    self.run.trace.content = step.content
                self.run.record(step)
                for call in step.tool_calls:
                    self.pending_calls[call.id] = (call, received_at)
            case "user":
                saw_result = False
                for block in message["message"]["content"]:
                    if block["type"] != "tool_result":
                        continue
                    call_id = block["tool_use_id"]
                    try:
                        call, started_at = self.pending_calls.pop(call_id)
                    except KeyError:
                        raise ValueError(
                            f"Claude returned a result for unknown tool call {call_id!r}"
                        ) from None

                    raw_result = block.get("content")
                    raw_items = raw_result if isinstance(raw_result, list) else [raw_result]
                    content: list[mcp_types.ContentBlock] = []
                    for item in raw_items:
                        if isinstance(item, str):
                            content.append(mcp_types.TextContent(type="text", text=item))
                        elif item["type"] == "text":
                            content.append(mcp_types.TextContent(type="text", text=item["text"]))
                        elif item["type"] == "image":
                            source = item["source"]
                            content.append(
                                mcp_types.ImageContent(
                                    type="image",
                                    data=source["data"],
                                    mimeType=source["media_type"],
                                )
                            )
                        else:
                            raise ValueError(f"unsupported Claude tool result block: {item!r}")

                    self.run.record(
                        ToolStep(
                            call=call,
                            result=MCPToolResult(
                                call_id=call_id,
                                content=content,
                                isError=block.get("is_error") is True,
                            ),
                            started_at=started_at,
                            ended_at=received_at,
                        )
                    )
                    saw_result = True
                if saw_result:
                    self.agent_started_at = received_at
            case "result":
                self.saw_result = True
                trace = self.run.trace
                result = message.get("result")
                if isinstance(result, str):
                    trace.content = result
                if message.get("is_error") is True:
                    self.error = trace.content or "claude CLI reported an error"
                for key in (
                    "subtype",
                    "session_id",
                    "duration_ms",
                    "duration_api_ms",
                    "stop_reason",
                    "num_turns",
                    "total_cost_usd",
                ):
                    if (value := message.get(key)) is not None:
                        trace.extra[key] = value

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self.run.trace
        error = self.error
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = stderr.strip() or f"claude CLI exited with return code {returncode}"
        elif not self.saw_result:
            error = "claude CLI exited without a result event"
        elif self.pending_calls:
            missing = ", ".join(sorted(self.pending_calls))
            error = f"claude CLI exited without results for tool calls: {missing}"

        if error is not None and stderr:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)
