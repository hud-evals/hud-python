"""How each provider agent hands a tool's result back to its model.

The environment serves an MCP server whose tools return each kind of MCP
content: text, nothing, an error, an image, a PDF and structured data. The
scripted model calls every one; the rows pin the result entries the provider
received on its next requests.
"""

from __future__ import annotations

import base64
import io
import json
import random
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
import pytest
from dirty_equals import IsStr
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from inline_snapshot import snapshot
from PIL import Image

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from tests.agents.support import image, mcp_server, run_task, wire, workspace_env
from tests.harness import call, say

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.harness import HudEnv, ModelRequest, Models

PDF = base64.b64encode(b"%PDF-1.4 scripted").decode()


def result_tools() -> FastMCP:
    server = FastMCP("results")

    @server.tool()
    def note() -> str:
        """Return a short note."""
        return "the note"

    @server.tool()
    def nothing() -> list[mcp_types.ContentBlock]:
        """Return no content."""
        return []

    @server.tool()
    def broken() -> str:
        """Always fail."""
        raise ToolError("broken tool")

    @server.tool()
    def picture() -> list[mcp_types.ContentBlock]:
        """Return a picture."""
        data = base64.b64encode(image((20, 10))).decode()
        return [mcp_types.ImageContent(type="image", data=data, mimeType="image/png")]

    @server.tool()
    def report() -> list[mcp_types.ContentBlock]:
        """Return a PDF report."""
        return [
            mcp_types.EmbeddedResource(
                type="resource",
                resource=mcp_types.BlobResourceContents(
                    uri="file:///report.pdf", mimeType="application/pdf", blob=PDF
                ),
            )
        ]

    @server.tool()
    def figures() -> dict[str, int]:
        """Return structured figures."""
        return {"total": 3}

    @server.tool()
    def long_output() -> str:
        """Return a long text."""
        return "h" * 500 + "t" * 500

    @server.tool()
    def link() -> list[mcp_types.ContentBlock]:
        """Return a resource link."""
        return [
            mcp_types.ResourceLink(type="resource_link", uri="file:///data.csv", name="data.csv")
        ]

    return server


ALL_TOOLS = ("note", "nothing", "broken", "picture", "report", "figures")


def conversation(requests: list[ModelRequest]) -> Any:
    """Everything after the prompt: the last request's history, or every Responses input."""
    last = requests[-1]
    if last.protocol == "responses":
        return [item for request in requests[1:] for item in request.body["input"]]
    if last.protocol == "gemini":
        return last.body["contents"][1:]
    messages = last.body["messages"]
    start = 2 if messages[0]["role"] == "system" else 1
    return messages[start:]


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    tools: tuple[str, ...]
    results: Any
    error: str | None = None


ROWS = {
    "claude": Row(
        agent=lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
        tools=ALL_TOOLS,
        results=snapshot(
            [
                {
                    "role": "assistant",
                    "content": [{"id": "toolu_1", "input": {}, "name": "note", "type": "tool_use"}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "the note"}],
                            "is_error": False,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_3", "input": {}, "name": "nothing", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_3",
                            "content": [],
                            "is_error": False,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_5", "input": {}, "name": "broken", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_5",
                            "content": [{"type": "text", "text": "Error: broken tool"}],
                            "is_error": True,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_7", "input": {}, "name": "picture", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_7",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "<image/png 20x10>",
                                    },
                                }
                            ],
                            "is_error": False,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_9", "input": {}, "name": "report", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_9",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": "JVBERi0xLjQgc2NyaXB0ZWQ=",
                                    },
                                }
                            ],
                            "is_error": False,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_11", "input": {}, "name": "figures", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_11",
                            "content": [{"type": "text", "text": '{"total":3}'}],
                            "is_error": False,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ]
        ),
    ),
    "claude-citations": Row(
        agent=lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6", citations_enabled=True)),
        tools=("note", "report", "broken"),
        results=snapshot(
            [
                {
                    "role": "assistant",
                    "content": [{"id": "toolu_1", "input": {}, "name": "note", "type": "tool_use"}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "the note"}],
                            "is_error": False,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "the note",
                            },
                            "title": "note",
                            "citations": {"enabled": True},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_3", "input": {}, "name": "report", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_3",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": "JVBERi0xLjQgc2NyaXB0ZWQ=",
                                    },
                                }
                            ],
                            "is_error": False,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQgc2NyaXB0ZWQ=",
                            },
                            "citations": {"enabled": True},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"id": "toolu_5", "input": {}, "name": "broken", "type": "tool_use"}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_5",
                            "content": [{"type": "text", "text": "Error: broken tool"}],
                            "is_error": True,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ]
        ),
    ),
    "claude-resource-link": Row(
        agent=lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
        tools=("link",),
        results=snapshot([]),
        error="Unknown content block type: <class 'mcp.types.ResourceLink'>",
    ),
    "openai": Row(
        agent=lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6")),
        tools=(*ALL_TOOLS, "link"),
        results=snapshot(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_3",
                    "output": [
                        {"type": "input_text", "text": '{"result": "the note"}'},
                        {"type": "input_text", "text": "the note"},
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_6",
                    "output": [{"type": "input_text", "text": ""}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_9",
                    "output": [
                        {"type": "input_text", "text": "[tool_error] true"},
                        {"type": "input_text", "text": "broken tool"},
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_12",
                    "output": [{"type": "input_image", "image_url": "<image/png 20x10>"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_15",
                    "output": [{"type": "input_file", "file_data": "JVBERi0xLjQgc2NyaXB0ZWQ="}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_18",
                    "output": [
                        {"type": "input_text", "text": '{"total": 3}'},
                        {"type": "input_text", "text": '{"total":3}'},
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_21",
                    "output": [{"type": "input_file", "file_url": "file:///data.csv"}],
                },
            ]
        ),
    ),
    "gemini": Row(
        agent=lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview")),
        tools=ALL_TOOLS,
        results=snapshot(
            [
                {"parts": [{"functionCall": {"args": {}, "name": "note"}}], "role": "model"},
                {
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "note",
                                "response": {"success": True, "output": "the note"},
                            }
                        }
                    ],
                    "role": "user",
                },
                {"parts": [{"functionCall": {"args": {}, "name": "nothing"}}], "role": "model"},
                {
                    "parts": [
                        {"functionResponse": {"name": "nothing", "response": {"success": True}}}
                    ],
                    "role": "user",
                },
                {"parts": [{"functionCall": {"args": {}, "name": "broken"}}], "role": "model"},
                {
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "broken",
                                "response": {"error": "broken tool"},
                            }
                        }
                    ],
                    "role": "user",
                },
                {"parts": [{"functionCall": {"args": {}, "name": "picture"}}], "role": "model"},
                {
                    "parts": [
                        {
                            "functionResponse": {
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": "<image/png 20x10>",
                                        }
                                    }
                                ],
                                "name": "picture",
                                "response": {"success": True},
                            }
                        }
                    ],
                    "role": "user",
                },
                {"parts": [{"functionCall": {"args": {}, "name": "report"}}], "role": "model"},
                {
                    "parts": [
                        {"functionResponse": {"name": "report", "response": {"success": True}}}
                    ],
                    "role": "user",
                },
                {"parts": [{"functionCall": {"args": {}, "name": "figures"}}], "role": "model"},
                {
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "figures",
                                "response": {"success": True, "output": '{"total":3}'},
                            }
                        }
                    ],
                    "role": "user",
                },
            ]
        ),
    ),
    "chat": Row(
        agent=lambda: OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")),
        tools=ALL_TOOLS,
        results=snapshot(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "note", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "the note"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {"name": "nothing", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_3", "content": "Tool executed successfully"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_5",
                            "type": "function",
                            "function": {"name": "broken", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_5", "content": "broken tool"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_7",
                            "type": "function",
                            "function": {"name": "picture", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_7", "content": "Tool executed successfully"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tool returned the following:"},
                        {"type": "image_url", "image_url": {"url": "<image/png 20x10>"}},
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": "report", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_9", "content": "Tool executed successfully"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_11",
                            "type": "function",
                            "function": {"name": "figures", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_11", "content": '{"total":3}'},
            ]
        ),
    ),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test_each_kind_of_tool_result_reaches_the_model_in_its_provider_shape(
    row: Row, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([*(call(name) for name in row.tools), say("done")])

    async with mcp_server(result_tools()) as tools:
        run = await run_task(workspace_env(tmp_path / "ws", capabilities=(tools,)), row.agent())

    assert (run.trace.status, run.trace.error) == (
        "completed" if row.error is None else "error",
        row.error,
    )
    assert wire(conversation(models.requests())) == row.results


async def test_gemini_bounds_a_long_tool_output_to_its_head_and_tail(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("long_output"), say("done")])
    agent = GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", max_tool_result_chars=100))

    async with mcp_server(result_tools()) as tools:
        await run_task(workspace_env(tmp_path / "ws", capabilities=(tools,)), agent)

    (result,) = models.requests()[-1].tool_results
    assert json.loads(result) == {
        "success": True,
        "output": (
            "Output too large: showing the first 20 and last 80 characters, 900 omitted. "
            "Re-run scoped to the part you need.\n"
            + "h" * 20
            + "\n\n... [900 characters omitted] ...\n\n"
            + "t" * 80
        ),
    }


def noisy_rgba(size: tuple[int, int]) -> bytes:
    noise = Image.frombytes("RGBA", size, random.Random(0).randbytes(size[0] * size[1] * 4))
    buffer = io.BytesIO()
    noise.save(buffer, format="PNG")
    return buffer.getvalue()


def png_claiming(size: tuple[int, int]) -> bytes:
    """A 1x1 PNG whose header claims ``size``."""
    data = bytearray(image((1, 1)))
    data[16:24] = struct.pack(">II", *size)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    return bytes(data)


def rotated_jpeg() -> bytes:
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    return image((40, 20), format="JPEG", color="red", exif=exif)


def picture(mime: str, size: str) -> list[dict[str, Any]]:
    data = f"<{mime} {size}>"
    return [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}]


def text(value: str | IsStr) -> list[dict[str, Any]]:
    return [{"type": "text", "text": value}]


CLAUDE_VIEWS = {
    "small-png": (image((40, 30)), (False, picture("image/png", "40x30"))),
    "wide-png": (image((3000, 2000)), (False, picture("image/png", "1312x875"))),
    "tall-png": (image((2000, 4000)), (False, picture("image/png", "758x1516"))),
    "exif-jpeg": (rotated_jpeg(), (False, picture("image/png", "20x40"))),
    "gif": (image((32, 24), format="GIF"), (False, picture("image/png", "32x24"))),
    "webp": (image((32, 24), format="WEBP"), (False, picture("image/png", "32x24"))),
    "transparent-png": (
        image((32, 24), mode="RGBA", color=(255, 0, 0, 128)),
        (False, picture("image/png", "32x24")),
    ),
    "noisy-rgba": (noisy_rgba((1072, 1072)), (False, picture("image/jpeg", "1072x1072"))),
    "oversized-header": (
        png_claiming((4001, 4000)),
        (
            True,
            text(
                "Cannot view image: Image exceeds the 16,000,000-pixel source limit. "
                "Resize or crop it before viewing."
            ),
        ),
    ),
    "corrupt-png": (
        b"\x89PNG\r\n\x1a\n" + b"garbage",
        (True, text(IsStr(regex=r"Cannot view image: cannot identify image file .*"))),
    ),
    "binary": (
        b"\x00binary",
        (True, text("Error: Cannot view this file: expected UTF-8 text or an image.")),
    ),
    "invalid-utf8": (
        b"\xff\xfeinvalid",
        (True, text("Error: Cannot view this file: expected UTF-8 text or an image.")),
    ),
    "unicode-text": ("Hello, 世界\n".encode(), (False, text("Hello, 世界\n"))),
}


@pytest.mark.filterwarnings("ignore:Image size.*:PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize(("data", "expected"), CLAUDE_VIEWS.values(), ids=CLAUDE_VIEWS.keys())
async def test_claude_sees_a_viewed_file_within_its_image_limits(
    data: bytes, expected: tuple[bool, Any], models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(
        [call("str_replace_based_edit_tool", command="view", path=f"{root}/file"), say("done")]
    )

    run = await run_task(
        workspace_env(root, files={"file": data}),
        ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
    )

    assert run.trace.status == "completed"
    (_prompt, _call, (result,)) = (m["content"] for m in models.requests()[-1].body["messages"])
    assert (result["is_error"], wire(result["content"])) == expected


CHAT_VIEWS = {
    "png": (image((40, 30)), "<image/png 40x30>"),
    "jpeg": (image((40, 30), format="JPEG"), "<image/jpeg 40x30>"),
    "gif": (image((40, 30), format="GIF"), "<image/gif 40x30>"),
    "webp": (image((40, 30), format="WEBP"), "<image/webp 40x30>"),
    "large-png": (image((3000, 2000)), "<image/png 3000x2000>"),
}


@pytest.mark.parametrize(("data", "expected"), CHAT_VIEWS.values(), ids=CHAT_VIEWS.keys())
async def test_chat_models_see_a_read_image_as_it_is_on_disk(
    data: bytes, expected: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script([call("read", filePath=f"{root}/file"), say("done")])

    await run_task(
        workspace_env(root, files={"file": data}),
        OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")),
    )

    *_, tool, follow_up = models.requests()[-1].body["messages"]
    assert tool["content"] == "Tool executed successfully"
    assert wire(follow_up["content"]) == [
        {"type": "text", "text": "Tool returned the following:"},
        {"type": "image_url", "image_url": {"url": expected}},
    ]
