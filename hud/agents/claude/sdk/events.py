"""Claude CLI stream translation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import mcp.types as mcp_types
from anthropic.types.beta import BetaMessage

from hud.agents.claude.agent import ClaudeAgent
from hud.agents.types import ToolStep
from hud.types import MCPToolCall, MCPToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.eval.run import Run

logger = logging.getLogger(__name__)


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
                    call, started_at = self.pending_calls.pop(call_id)

                    raw_result = block.get("content")
                    raw_items = raw_result if isinstance(raw_result, list) else [raw_result]
                    content: list[mcp_types.ContentBlock] = []
                    for index, item in enumerate(raw_items):
                        if isinstance(item, str):
                            content.append(mcp_types.TextContent(type="text", text=item))
                            continue
                        match item["type"]:
                            case "text":
                                content.append(
                                    mcp_types.TextContent(type="text", text=item["text"])
                                )
                            case "image":
                                source = item["source"]
                                content.append(
                                    mcp_types.ImageContent(
                                        type="image",
                                        data=source["data"],
                                        mimeType=source["media_type"],
                                    )
                                )
                            case "document" if item["source"]["type"] in ("base64", "text"):
                                # The CLI's Read on a PDF.
                                source = item["source"]
                                uri = f"document://{call_id}/{index}"
                                resource: mcp_types.ResourceContents = (
                                    mcp_types.BlobResourceContents(
                                        uri=uri,
                                        mimeType=source.get("media_type"),
                                        blob=source["data"],
                                    )
                                    if source["type"] == "base64"
                                    else mcp_types.TextResourceContents(
                                        uri=uri,
                                        mimeType=source.get("media_type"),
                                        text=source["data"],
                                    )
                                )
                                content.append(
                                    mcp_types.EmbeddedResource(type="resource", resource=resource)
                                )
                            case kind:
                                logger.warning("unsupported Claude tool result block: %s", kind)
                                content.append(
                                    mcp_types.TextContent(
                                        type="text", text=f"[unsupported {kind} block]"
                                    )
                                )

                    self.run.record(
                        ToolStep(
                            call=call,
                            result=MCPToolResult(
                                call_id=call_id,
                                content=content,
                                isError=bool(block.get("is_error")),
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
                if message.get("is_error"):
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
        """Fail the rollout unless the CLI delivered a complete terminal result.

        A nonzero exit after a result event is only recorded on the trace when
        stderr is empty; the CLI exits that way after successful runs.
        """
        trace = self.run.trace
        if returncode != 0:
            trace.extra["returncode"] = returncode

        reported = stderr.strip()
        if self.error is not None:
            error = self.error
        elif not self.saw_result:
            error = reported or (
                f"claude CLI exited with return code {returncode}"
                if returncode != 0
                else "claude CLI exited without a result event"
            )
        elif returncode != 0 and reported:
            error = reported
        elif self.pending_calls:
            missing = ", ".join(sorted(self.pending_calls))
            error = f"claude CLI exited without results for tool calls: {missing}"
        else:
            error = None

        if error is not None and stderr:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)
