"""Hosted tools that are executed by the provider, not the client.

These tools are declared in the environment but executed server-side by the LLM provider.
The client only declares them and processes the response metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType

if TYPE_CHECKING:
    from mcp.types import ContentBlock


class HostedTool(BaseTool):
    """Base class for tools executed by the provider, not the client.

    Hosted tools are declared in the environment and registered with the provider's
    native API, but the actual execution happens on the provider's infrastructure.
    The client receives results through the response metadata.

    Subclasses should:
    1. Define `native_specs` with `hosted=True`
    2. Optionally override `process_response` to extract provider-specific metadata

    Example:
        class GoogleSearchTool(HostedTool):
            native_specs = {
                AgentType.GEMINI: NativeToolSpec(api_type="google_search", hosted=True),
            }
    """

    async def __call__(self, **kwargs: Any) -> list[ContentBlock]:
        """Hosted tools cannot be called directly - they are executed by the provider.

        Raises:
            NotImplementedError: Always, as hosted tools are provider-executed
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is executed by the provider. "
            "Results are returned in the response metadata, not via tool calls."
        )

    @staticmethod
    def process_response(response: Any) -> dict[str, Any]:
        """Extract provider-specific metadata from the response.

        Override this method in subclasses to parse provider-specific response formats.

        Args:
            response: The raw response from the provider

        Returns:
            Dictionary with extracted metadata
        """
        return {}


class GoogleSearchTool(HostedTool):
    """Gemini's native Google Search grounding tool.

    When enabled, Gemini will ground its responses in real-time Google Search results.
    The search happens server-side and results are included in the response metadata.

    See: https://ai.google.dev/gemini-api/docs/google-search
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(api_type="google_search", hosted=True),
        AgentType.GEMINI_CUA: NativeToolSpec(api_type="google_search", hosted=True),
    }

    def __init__(self, dynamic_threshold: float | None = None) -> None:
        """Initialize GoogleSearchTool.

        Args:
            dynamic_threshold: Optional threshold for dynamic retrieval.
                Controls when grounding is triggered (0.0-1.0).
                Lower values mean more grounding, higher means less.
        """
        extra: dict[str, Any] = {}
        if dynamic_threshold is not None:
            extra["dynamic_threshold"] = dynamic_threshold

        # Build instance-level specs with extra params if provided
        instance_specs: NativeToolSpecs | None = None
        if extra:
            instance_specs = {
                AgentType.GEMINI: NativeToolSpec(
                    api_type="google_search",
                    hosted=True,
                    extra=extra,
                ),
                AgentType.GEMINI_CUA: NativeToolSpec(
                    api_type="google_search",
                    hosted=True,
                    extra=extra,
                ),
            }

        super().__init__(
            name="google_search",
            title="Google Search",
            description="Ground responses in real-time Google Search results",
            native_specs=instance_specs,
        )

    @staticmethod
    def process_response(response: Any) -> dict[str, Any]:
        """Extract grounding metadata from Gemini response.

        Args:
            response: Gemini GenerateContentResponse

        Returns:
            Dictionary with search_queries, sources, and citations
        """
        try:
            if not response.candidates:
                return {}

            candidate = response.candidates[0]
            metadata = getattr(candidate, "grounding_metadata", None)

            if not metadata:
                return {}

            result: dict[str, Any] = {}

            # Extract search queries
            if hasattr(metadata, "web_search_queries"):
                result["search_queries"] = list(metadata.web_search_queries or [])

            # Extract grounding chunks (sources)
            if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                result["sources"] = [
                    {"uri": chunk.web.uri, "title": chunk.web.title}
                    for chunk in metadata.grounding_chunks
                    if hasattr(chunk, "web") and chunk.web
                ]

            # Extract grounding supports (citations)
            if hasattr(metadata, "grounding_supports") and metadata.grounding_supports:
                result["citations"] = [
                    {
                        "text": support.segment.text if support.segment else "",
                        "source_indices": list(support.grounding_chunk_indices or []),
                    }
                    for support in metadata.grounding_supports
                ]

            return result
        except Exception:
            return {}


class CodeExecutionTool(HostedTool):
    """Provider-executed code execution tool.

    When enabled, the model can generate and execute code in a sandboxed environment.
    Supported by Gemini (code_execution) and OpenAI (code_interpreter).

    Note: OpenAI's code_interpreter requires additional configuration and may have
    usage costs. Gemini's code_execution is included in standard API access.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(api_type="code_execution", hosted=True),
        AgentType.GEMINI_CUA: NativeToolSpec(api_type="code_execution", hosted=True),
        AgentType.OPENAI: NativeToolSpec(api_type="code_interpreter", hosted=True),
    }

    def __init__(self) -> None:
        """Initialize CodeExecutionTool."""
        super().__init__(
            name="code_execution",
            title="Code Execution",
            description="Execute code in a sandboxed environment",
        )

    @staticmethod
    def process_response(response: Any) -> dict[str, Any]:
        """Extract code execution results from the response.

        Args:
            response: Provider response containing code execution results

        Returns:
            Dictionary with code and output fields
        """
        # Gemini includes executable_code and code_execution_result in parts
        try:
            results: list[dict[str, Any]] = []

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts or []:
                        if hasattr(part, "executable_code") and part.executable_code:
                            results.append({
                                "type": "code",
                                "language": getattr(
                                    part.executable_code, "language", "python"
                                ),
                                "code": part.executable_code.code,
                            })
                        if hasattr(part, "code_execution_result") and part.code_execution_result:
                            results.append({
                                "type": "result",
                                "outcome": getattr(
                                    part.code_execution_result, "outcome", "unknown"
                                ),
                                "output": part.code_execution_result.output,
                            })

            return {"executions": results} if results else {}
        except Exception:
            return {}


class UrlContextTool(HostedTool):
    """Gemini's URL context tool for fetching and including web content.

    When enabled, allows the model to fetch and include content from URLs
    in its context. The fetching happens server-side.

    See: https://ai.google.dev/gemini-api/docs/url-context
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(api_type="url_context", hosted=True),
        AgentType.GEMINI_CUA: NativeToolSpec(api_type="url_context", hosted=True),
    }

    def __init__(self) -> None:
        """Initialize UrlContextTool."""
        super().__init__(
            name="url_context",
            title="URL Context",
            description="Fetch and include web content from URLs",
        )


__all__ = [
    "HostedTool",
    "GoogleSearchTool",
    "CodeExecutionTool",
    "UrlContextTool",
]
