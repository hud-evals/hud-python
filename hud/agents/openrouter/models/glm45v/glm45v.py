"""glm-4.5v computer-use agent backed by litellm + openrouter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar
from pathlib import Path

import litellm
import mcp.types as types
from litellm.types.utils import ModelResponse

from hud.agents.base import MCPAgent
from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud import instrument
from hud.agents.openrouter import (
    _convert_json_action_to_items,
    _decode_image_dimensions,
    _extract_user_instruction,
    _make_failed_tool_call_items,
    _make_output_text_item,
    _make_reasoning_item,
    _make_screenshot_item,
    _parse_json_action_string,
    _random_id,
    get_last_image_from_messages,
)

logger = logging.getLogger(__name__)

def _load_text_resource(path: str | Path) -> str | None:
    try:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

_BASE_DIR = Path(__file__).resolve().parent
_ACTION_SPACE_PATH = _BASE_DIR / "action_space.txt"

GLM_ACTION_SPACE = _load_text_resource(_ACTION_SPACE_PATH) or ""
if not GLM_ACTION_SPACE.strip():
    raise RuntimeError(f"Missing action space file at {_ACTION_SPACE_PATH}")

def convert_responses_items_to_glm45v_pc_prompt(
    messages: list[dict[str, Any]],
    task: str,
    memory: str = "[]",
) -> list[dict[str, Any]]:
    head_text = (
        "You are a GUI Agent, and your primary task is to respond accurately to user"
        " requests or questions. In addition to directly answering the user's queries,"
        " you can also use tools or perform GUI operations directly until you fulfill"
        " the user's request or provide a correct answer. You should carefully read and"
        " understand the images and questions provided by the user, and engage in"
        " thinking and reflection when appropriate. The coordinates involved are all"
        " represented in thousandths (0-999)."
        "\n\n# Task:\n"
        f"{task}\n\n# Task Platform\nUbuntu\n\n# Action Space\n{GLM_ACTION_SPACE}\n\n"
        "# Historical Actions and Current Memory\nHistory:"
    )

    tail_text = (
        "\nMemory:\n"
        f"{memory}\n"
        "# Output Format\nPlain text explanation with action(param='...')\n"
        "Memory:\n[{\"key\": \"value\"}, ...]\n\n# Some Additional Notes\n"
        "- I'll give you the most recent history screenshots(shrunked to 50%*50%) along with the historical action steps.\n"
        "- You should put the key information you *have to remember* in a seperated memory part and I'll give it to you in the next round."
        " The content in this part should be a dict list. If you no longer need some given information, you should remove it from the memory."
        " Even if you don't need to remember anything, you should also output an empty list.\n"
        "- If elevated privileges are needed, credentials are referenced as <OS_PASSWORD>.\n"
        "- For any mail account interactions, credentials are referenced as <MAIL_PASSWORD>.\n\n"
        "Current Screenshot:\n"
    )

    history: list[dict[str, Any]] = []
    history_images: list[str] = []
    current_step: list[dict[str, Any]] = []
    step_num = 0

    # Optimization: Limit history to last 10 messages to improve performance
    for message in messages[-10:]:
        if not isinstance(message, dict):
            continue
        msg_type = message.get("type")

        if msg_type in {"reasoning", "message", "computer_call", "computer_call_output"}:
            current_step.append(message)

        if msg_type == "computer_call_output" and current_step:
            step_num += 1

            bot_thought = ""
            action_text = ""
            for item in current_step:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content") or []
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "output_text":
                                bot_thought = block.get("text", "")
                                break
                if item.get("type") == "computer_call":
                    action_text = json.dumps(item.get("action", {}))

            history.append({
                "step_num": step_num,
                "bot_thought": bot_thought,
                "action_text": action_text,
            })

            output = message.get("output") or {}
            if isinstance(output, dict) and output.get("type") == "input_image":
                url = output.get("image_url")
                if isinstance(url, str):
                    history_images.append(url)

            current_step = []

    content: list[dict[str, Any]] = []
    current_text = head_text

    total_steps = len(history)
    image_tail = min(2, len(history_images))

    for idx, step in enumerate(history):
        step_no = step["step_num"]
        bot_thought = step["bot_thought"]
        action_text = step["action_text"]

        if idx < total_steps - image_tail:
            current_text += (
                f"\nstep {step_no}: Screenshot:(Omitted in context.)"
                f" Thought: {bot_thought}\nAction: {action_text}"
            )
        else:
            current_text += f"\nstep {step_no}: Screenshot:"
            content.append({"type": "text", "text": current_text})
            image_idx = idx - (total_steps - image_tail)
            if 0 <= image_idx < len(history_images):
                content.append({"type": "image_url", "image_url": {"url": history_images[image_idx]}})
            current_text = f" Thought: {bot_thought}\nAction: {action_text}"

    current_text += tail_text
    content.append({"type": "text", "text": current_text})
    return content

def _parse_string_action_to_dict(action: str) -> dict[str, Any]:
    """Converts GLM's string-based action output to a structured dictionary."""
    if action.startswith("left_click"):
        match = re.search(r"start_box='?\[(\d+),\s*(\d+)\]'?", action)
        if match: return {"type": "click", "button": "left", "start_box": [match.group(1), match.group(2)]}
    elif action.startswith("right_click"):
        match = re.search(r"start_box='?\[(\d+),\s*(\d+)\]'?", action)
        if match: return {"type": "click", "button": "right", "start_box": [match.group(1), match.group(2)]}
    elif action.startswith("left_double_click"):
        match = re.search(r"start_box='?\[(\d+),\s*(\d+)\]'?", action)
        if match: return {"type": "double_click", "start_box": [match.group(1), match.group(2)]}
    elif action.startswith("left_drag"):
        start_match = re.search(r"start_box='?\[(\d+),\s*(\d+)\]'?", action)
        end_match = re.search(r"end_box='?\[(\d+),\s*(\d+)\]'?", action)
        if start_match and end_match:
            return {
                "type": "drag",
                "start_box": [start_match.group(1), start_match.group(2)],
                "end_box": [end_match.group(1), end_match.group(2)],
            }
    elif action.startswith("key"):
        key_match = re.search(r"keys='([^']+)'", action)
        if key_match:
            keys = key_match.group(1)
            key_list = keys.split("+") if "+" in keys else [keys]
            return {"type": "keypress", "keys": key_list}
    elif action.startswith("type"):
        content_match = re.search(r"content='([^']*)'", action)
        if content_match: return {"type": "type", "content": content_match.group(1)}
    elif action.startswith("scroll"):
        coord_match = re.search(r"start_box='?\[(\d+),\s*(\d+)\]'?", action)
        direction_match = re.search(r"direction='([^']+)'", action)
        if coord_match and direction_match:
            return {
                "type": "scroll",
                "start_box": [coord_match.group(1), coord_match.group(2)],
                "direction": direction_match.group(1),
            }
    elif action == "WAIT()":
        return {"type": "wait"}
    return {}


def convert_glm_completion_to_responses_items(
    response: ModelResponse,
    image_width: int,
    image_height: int,
    parsed_response: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if not getattr(response, "choices", None):
        return items

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if not message:
        return items

    content = getattr(message, "content", "") or ""
    reasoning_content = getattr(message, "reasoning_content", None)

    if reasoning_content:
        items.append(_make_reasoning_item(str(reasoning_content)))

    parsed = parsed_response or parse_glm_response(content)
    action = parsed.get("action", "")
    action_text = parsed.get("action_text", "")

    if action_text:
        clean_text = action_text
        if action:
            clean_text = clean_text.replace(action, "").strip()
        clean_text = re.sub(r"Memory:\s*\[.*?\]\s*$", "", clean_text, flags=re.DOTALL).strip()
        if clean_text:
            items.append(_make_output_text_item(clean_text))

    if action:
        call_id = _random_id()

        json_action = _parse_json_action_string(action)
        if not json_action:
            json_action = _parse_string_action_to_dict(action)

        if json_action:
            json_entries = _convert_json_action_to_items(
                json_action,
                call_id=call_id,
                image_width=image_width,
                image_height=image_height,
            )
            if json_entries:
                items.extend(json_entries)

    return items


def parse_glm_response(response: str) -> dict[str, str]:
    json_match = re.search(r'(\{.*\})', response)
    if json_match:
        action = json_match.group(1).strip()
    else:
        box_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", response)
        if box_match:
            action = box_match.group(1).strip()
        else:
            action_pattern = r"[\w_]+\([^)]*\)"
            matches = re.findall(action_pattern, response)
            action = matches[0] if matches else ""

    memory_pattern = r"Memory:(.*?)$"
    memory_match = re.search(memory_pattern, response, re.DOTALL)
    memory = memory_match.group(1).strip() if memory_match else "[]"

    action_text_pattern = r"^(.*?)Memory:"
    action_text_match = re.search(action_text_pattern, response, re.DOTALL)
    action_text = action_text_match.group(1).strip() if action_text_match else response
    if action_text:
        action_text = action_text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")

    return {
        "action": action or "",
        "action_text": action_text,
        "memory": memory,
    }

class Glm45vAgent(MCPAgent):
    """LiteLLM-backed GLM-4.5V agent that speaks MCP."""

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.OPENAI_COMPUTER_WIDTH,
        "display_height": computer_settings.OPENAI_COMPUTER_HEIGHT,
    }

    required_tools: ClassVar[list[str]] = ["openai_computer"]

    def __init__(
        self,
        *,
        model_name: str = "z-ai/glm-4.5v",
        completion_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(**agent_kwargs)
        self.model_name = model_name
        self.completion_kwargs = completion_kwargs or {}
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = ""
        self._memory = "[]"
        self._last_instruction = ""
        self._task_description = ""

    async def get_system_messages(self) -> list[Any]:
        return []

    @instrument(span_type="agent", record_args=False)
    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[dict[str, Any]]:
        content_items: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                text_parts.append(block.text)
            elif isinstance(block, types.ImageContent):
                content_items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{getattr(block, 'mimeType', 'image/png')};base64,{block.data}",
                                },
                            }
                        ],
                    }
                )

        if text_parts:
            content_items.insert(
                0,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "\n".join(text_parts)}],
                },
            )

        return content_items

    def _glm_tool_call_to_mcp(self, item: dict[str, Any]) -> MCPToolCall:
        call_id = item.get("call_id") or _random_id()
        action = item.get("action") or {}
        action_type = action.get("type", "")

        arguments: dict[str, Any] = {"type": action_type}
        for key in ("x", "y", "scroll_x", "scroll_y"):
            if key in action:
                arguments[key] = action[key]
        if "button" in action:
            arguments["button"] = action["button"]
        if "keys" in action:
            arguments["keys"] = action["keys"]
        if "text" in action:
            arguments["text"] = action["text"]
        if "path" in action:
            arguments["path"] = action["path"]

        return MCPToolCall(id=call_id, name="openai_computer", arguments=arguments)

    @instrument(span_type="agent", record_args=False)
    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        instruction = _extract_user_instruction(messages)
        if instruction:
            self._last_instruction = instruction  # type: ignore[attr-defined]
            self._task_description = instruction
        task_instruction = self._task_description or getattr(self, "_last_instruction", "")

        screenshot_b64 = get_last_image_from_messages(messages)
        if not screenshot_b64:
            call_id = _random_id()
            screenshot_call = _make_screenshot_item(call_id)
            messages.append(screenshot_call)
            logger.debug("glm45v requesting initial screenshot")
            tool_call = MCPToolCall(
                id=call_id,
                name="openai_computer",
                arguments={"type": "screenshot"},
            )
            return AgentResponse(
                content="capturing initial screenshot",
                tool_calls=[tool_call],
                done=False,
            )

        self.console.debug(f"glm45v task instruction: {task_instruction}")
        self.console.debug(f"glm45v memory (pre-step): {self._memory}")

        prompt_content = convert_responses_items_to_glm45v_pc_prompt(
            messages=messages,
            task=task_instruction,
            memory=self._memory,
        )
        prompt_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
        )

        litellm_messages: list[dict[str, Any]] = []
        if getattr(self, "system_prompt", None):
            litellm_messages.append({"role": "system", "content": self.system_prompt})
        litellm_messages.append({"role": "user", "content": prompt_content})

        api_kwargs = {"model": self.model_name, "messages": litellm_messages}
        if settings.openrouter_api_key:
            api_kwargs["api_key"] = settings.openrouter_api_key
        api_kwargs.update(self.completion_kwargs)

        try:
            response = await litellm.acompletion(**api_kwargs)
        except Exception as exc:  # pragma: no cover - network errors
            logger.exception("glm45v completion failed: %s", exc)
            return AgentResponse(
                content=f"GLM-4.5V request failed: {exc}",
                tool_calls=[],
                done=True,
                isError=True,
            )

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        response_content = getattr(message, "content", "") if message else ""
        parsed = parse_glm_response(response_content or "") if response_content else {
            "memory": self._memory,
        }
        if parsed.get("memory"):
            self._memory = parsed["memory"]
        logger.debug("glm45v model content: %s", response_content)
        trimmed = response_content[:400] if response_content else ""
        self.console.debug(f"glm45v model content: {trimmed}")
        self.console.debug(f"glm45v parsed response: {parsed}")

        image_width, image_height = _decode_image_dimensions(screenshot_b64)
        response_items = convert_glm_completion_to_responses_items(
            response,
            image_width=image_width,
            image_height=image_height,
            parsed_response=parsed,
        )

        messages.extend(response_items)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[MCPToolCall] = []

        for item in response_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
            elif item.get("type") == "reasoning":
                summary = item.get("summary", [])
                for block in summary:
                    if isinstance(block, dict) and block.get("text"):
                        reasoning_parts.append(block["text"])
            elif item.get("type") == "computer_call":
                tool_calls.append(self._glm_tool_call_to_mcp(item))

        content_text = "\n".join(text_parts).strip()
        reasoning_text = "\n".join(reasoning_parts).strip()

        if not tool_calls:
            self.console.info_log(
                f"glm45v returned no tool calls. content='{content_text}' reasoning='{reasoning_text}'"
            )
            self.console.info_log(f"glm45v parsed response: {parsed}")

        return AgentResponse(
            content=content_text or None,
            reasoning=reasoning_text or None,
            tool_calls=tool_calls,
            done=not tool_calls,
            raw=response,
        )

    @instrument(span_type="agent", record_args=False)
    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []

        for call, result in zip(tool_calls, tool_results, strict=False):
            call_args = call.arguments or {}
            if result.isError:
                error_text = "".join(
                    content.text
                    for content in result.content
                    if isinstance(content, types.TextContent)
                )
                rendered.extend(
                    _make_failed_tool_call_items(
                        tool_name=call_args.get("type", call.name),
                        tool_kwargs=call_args,
                        error_message=error_text or "Unknown error",
                        call_id=call.id,
                    )
                )
                continue

            screenshot_found = False
            for content in result.content:
                if isinstance(content, types.ImageContent):
                    rendered.append(
                        {
                            "type": "computer_call_output",
                            "call_id": call.id,
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:{content.mimeType};base64,{content.data}",
                            },
                        }
                    )
                    screenshot_found = True
                    break

            text_parts = [
                content.text
                for content in result.content
                if isinstance(content, types.TextContent) and content.text
            ]
            if text_parts:
                rendered.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "\n".join(text_parts)}],
                    }
                )

            if not screenshot_found and not text_parts:
                rendered.append(
                    {
                        "type": "computer_call_output",
                        "call_id": call.id,
                        "output": {"type": "input_text", "text": "Tool executed"},
                    }
                )

        return rendered


__all__ = ["Glm45vAgent"]
