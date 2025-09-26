"""LiteLLM MCP Agent implementation.

This module defines :class:`LiteAgent`, a thin wrapper around
``GenericOpenAIChatAgent`` which swaps out the underlying transport for
`litellm` and adds support for LiteLLM-specific tooling features such as
prompt caching.  The base class is kept intentionally stateless to make
composition easy, so the only differences in this subclass are the
return type of :meth:`get_tool_schemas`, the format of system messages and
tool result messages, and the mechanism used to invoke the chat
completion endpoint.

Several subtle issues existed in the original implementation which
prevented the agent from operating reliably:

* ``litellm`` is an optional dependency.  If it is not available at
  runtime we must fall back to the generic OpenAI semantics rather
  than attempting to use a missing module.  The original code would
  attempt to call ``litellm.acompletion`` unconditionally, which
  raises an ``ImportError``.
* ``transform_mcp_tool_to_openai_tool`` from the ``litellm`` package
  returns a ``ChatCompletionToolParam`` object.  The OpenAI and
  LiteLLM APIs expect a plain dictionary for tools, so we must
  convert this Pydantic model to a native ``dict``.  Without this
  conversion the underlying JSON encoder cannot serialise the
  ``ChatCompletionToolParam`` instance.
* The original code used the ``strict`` keyword argument when
  invoking ``zip``.  This argument was only introduced in Python
  3.11.  To maintain compatibility with Python 3.8–3.10 we avoid
  passing it.

The changes below address these issues by deferring import of
``litellm`` until runtime, converting tool schemas into plain
dictionaries, and simplifying ``zip`` calls.  See individual comments
inline for further details.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Any, ClassVar, Dict, List

try:
    import litellm  # type: ignore[import]
except Exception:
    # If litellm isn't installed the module will remain None.  The
    # fallback transport will be provided by the base Generic agent.
    litellm = None  # type: ignore[assignment]

try:
    # Prefer LiteLLM's built-in MCP -> OpenAI tool transformer (handles
    # Bedrock nuances).  If this import fails we fall back to the
    # GenericOpenAIChatAgent implementation which sanitises schemas.
    from litellm.experimental_mcp_client.tools import (
        transform_mcp_tool_to_openai_tool,  # type: ignore[import]
    )
except Exception:  # pragma: no cover - optional dependency
    transform_mcp_tool_to_openai_tool = None  # type: ignore[assignment]

import mcp.types as types

from hud.types import MCPToolCall, MCPToolResult

from .openai_chat_generic import GenericOpenAIChatAgent

logger = logging.getLogger(__name__)


class LiteAgent(GenericOpenAIChatAgent):
    """
    Same OpenAI chat-completions shape + MCP tool plumbing, but the
    transport is provided by ``litellm`` when available.  When
    LiteLLM's tool transformer is present we use it to shape our tool
    schemas, otherwise we fall back to the generic OpenAI
    transformation defined in the base class.
    """

    metadata: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        completion_kwargs: Dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        # We don't need an OpenAI client; pass None.  All other
        # arguments are forwarded to GenericOpenAIChatAgent.
        super().__init__(
            openai_client=None,
            model_name=model_name,
            completion_kwargs=completion_kwargs,
            **agent_kwargs,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool schemas.

        When ``transform_mcp_tool_to_openai_tool`` is available from
        ``litellm``, it returns a ``ChatCompletionToolParam`` instance
        that must be converted to a native ``dict``.  Otherwise we
        delegate to the base class to handle schema sanitisation.
        """
        if transform_mcp_tool_to_openai_tool is not None:
            tools: List[Dict[str, Any]] = []
            for t in self.get_available_tools():
                openai_tool_obj = transform_mcp_tool_to_openai_tool(t)
                # ``transform_mcp_tool_to_openai_tool`` returns a
                # ``ChatCompletionToolParam`` (a Pydantic model).  To
                # ensure compatibility with both LiteLLM and OpenAI
                # clients we convert it into a plain dict.  Pydantic v2
                # uses ``model_dump`` while v1 uses ``dict``.
                if hasattr(openai_tool_obj, "model_dump"):
                    openai_tool: Dict[str, Any] = openai_tool_obj.model_dump(exclude_none=True)  # type: ignore[assignment]
                elif hasattr(openai_tool_obj, "dict"):
                    openai_tool = openai_tool_obj.dict(exclude_none=True)  # type: ignore[assignment]
                else:
                    openai_tool = openai_tool_obj  # type: ignore[assignment]
                # Sanitize the function parameters to ensure OpenAI-compliant
                # schemas (e.g. add ``items`` to array types, select appropriate
                # variants from anyOf).  The transformed tool will always
                # include a ``function`` entry with ``parameters``; if not,
                # leave it unchanged.
                try:
                    fn = openai_tool.get("function")
                    if fn and isinstance(fn, dict):
                        params = fn.get("parameters")
                        if params and isinstance(params, dict):
                            sanitized = GenericOpenAIChatAgent._sanitize_schema_for_openai(self, params)  # type: ignore[attr-defined]
                            fn["parameters"] = sanitized  # type: ignore[index]
                except Exception:
                    # If sanitization fails, leave schema unchanged
                    pass
                tools.append(openai_tool)  # type: ignore[arg-type]
            return tools
        # Fallback to the generic OpenAI sanitiser
        return GenericOpenAIChatAgent.get_tool_schemas(self)

    async def get_system_messages(self) -> List[Any]:  # type: ignore[override]
        """Get system messages with caching support.

        The system prompt is returned as a single text block marked
        ``ephemeral`` so that LiteLLM's prompt caching middleware can
        recognise it as not being persisted between requests.
        """
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

    async def format_blocks(self, blocks: List[types.ContentBlock]) -> List[Any]:  # type: ignore[override]
        """Format user-provided content blocks with caching support."""
        content: List[Dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append(
                    {
                        "type": "text",
                        "text": block.text,
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            elif isinstance(block, types.ImageContent):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                        "cache_control": {"type": "ephemeral"},
                    }
                )
        return [{"role": "user", "content": content}]

    async def format_tool_results(
        self,
        tool_calls: List[MCPToolCall],
        tool_results: List[MCPToolResult],
    ) -> List[Any]:  # type: ignore[override]
        """Render MCP tool results with caching support.

        The OpenAI and LiteLLM APIs only support string content for tool
        messages.  When images are present, they are returned as a
        separate user message.  For compatibility with Python
        versions prior to 3.11 we avoid passing ``strict=False`` to
        ``zip``; instead we simply iterate over the shortest list.
        """
        rendered: List[Dict[str, Any]] = []
        image_parts: List[Dict[str, Any]] = []
        # iterate over the overlapping portion of tool_calls and tool_results
        for call, res in zip(tool_calls, tool_results):
            text_parts: List[str] = []
            items = res.content
            # Use structuredContent.result if available, otherwise use content
            if not res.content and res.structuredContent:
                items = [res.structuredContent.get("result", res.content)]
            for item in items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        mime_type = item.get("mimeType", "image/png")
                        data = item.get("data", "")
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{data}"},
                                "cache_control": {"type": "ephemeral"},
                            }
                        )
                elif isinstance(item, types.TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, types.ImageContent):
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
            text_content = "".join(text_parts) if text_parts else "Tool executed successfully"
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": text_content,
                }
            )
        # If there are images, add the last image as a separate user message.
        # LiteLLM/OpenAI currently support only one image per message; we
        # therefore take the most recent image.  We prefix it with a
        # short text so that the client knows why the image is being
        # shown.
        if image_parts:
            content_with_images: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": "Tool returned the following:",
                    "cache_control": {"type": "ephemeral"},
                },
                image_parts[-1],
            ]
            rendered.append(
                {
                    "role": "user",
                    "content": content_with_images,
                }
            )
        return rendered

    async def _invoke_chat_completion(
        self,
        *,
        messages: List[Any],
        tools: List[Dict[str, Any]] | None,
        extra: Dict[str, Any],
    ) -> Any:  # type: ignore[override]
        """Invoke the underlying LLM via LiteLLM when available.

        If ``litellm`` is installed we delegate to ``litellm.acompletion``.
        Otherwise we defer to the base class implementation which uses
        the OpenAI client.  This allows the agent to function even
        when LiteLLM isn't installed.
        """
        if litellm is None:
            # Defer to the generic implementation which will raise if
            # ``openai_client`` is missing.  This mirrors the original
            # behaviour when LiteLLM is unavailable.
            return await super()._invoke_chat_completion(messages=messages, tools=tools, extra=extra)

        # On LiteLLM we tolerate ``None`` for tools and drop the
        # ``tool_choice`` parameter when no tools are provided.  Set
        # ``tool_choice='auto'`` explicitly when tools are present.
        # Additionally pull retry-related parameters from the caller
        # configuration.  ``num_retries`` controls the total number of
        # attempts (default 3).  ``retry_backoff`` sets the base delay
        # in seconds for exponential backoff (default 1).
        # Pop these keys from ``extra`` so they are not forwarded to
        # ``litellm.acompletion`` where they may not be supported.
        max_attempts = extra.pop("num_retries", None)
        base_delay = extra.pop("retry_backoff", None)
        try:
            # ``completion_kwargs`` on the agent can also specify retry
            # configuration.  Only set if not already provided via
            # ``extra``.
            if max_attempts is None:
                max_attempts = self.completion_kwargs.get("num_retries")  # type: ignore[index]
            if base_delay is None:
                base_delay = self.completion_kwargs.get("retry_backoff")  # type: ignore[index]
        except Exception:
            # completion_kwargs may be None or not indexable
            pass
        # Fallback defaults
        if not isinstance(max_attempts, int) or max_attempts <= 0:
            max_attempts = 3
        if not isinstance(base_delay, (int, float)) or base_delay <= 0:
            base_delay = 1.0

        params: Dict[str, Any] = {**extra, "model": self.model_name, "messages": messages}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        for attempt in range(max_attempts):
            try:
                # ``litellm.acompletion`` is asynchronous.  See
                # https://docs.litellm.ai/docs/completion/function_call for
                # details on available parameters.
                return await litellm.acompletion(**params)
            except Exception as e:
                # Identify rate limit errors.  ``litellm`` exposes
                # ``RateLimitError`` on the module; if not present,
                # fallback to checking the class name.  Only retry
                # rate-limit errors with backoff; re-raise others.
                rate_limit_error = False
                if litellm is not None and hasattr(litellm, "RateLimitError"):
                    rate_limit_error = isinstance(e, litellm.RateLimitError)  # type: ignore[attr-defined]
                # Fallback: check the exception's class name for RateLimit
                if not rate_limit_error and "RateLimit" in e.__class__.__name__:
                    rate_limit_error = True
                if rate_limit_error and attempt < max_attempts - 1:
                    delay = float(base_delay) * (2 ** attempt)
                    logger.warning(
                        f"Rate limit encountered on attempt {attempt + 1} of {max_attempts}; "
                        f"sleeping {delay:.1f}s before retrying: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                # For all other exceptions or exhausted retries, re-raise
                raise