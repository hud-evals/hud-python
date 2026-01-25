"""Gemini's native Google Search grounding tool."""

from __future__ import annotations

from typing import Any, ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


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
