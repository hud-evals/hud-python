"""Base class for hosted tools executed by the provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.tools.base import BaseTool

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
