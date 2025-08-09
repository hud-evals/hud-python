"""Client-side patches for handling known server issues gracefully."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_mcp_client_call_tool():
    """Patch the MCP client's call_tool to handle output schema validation errors gracefully."""
    try:
        from mcp import types

        from hud.clients.fastmcp import FastMCPHUDClient
        from hud.types import MCPToolResult

        # Store original call_tool method
        original_call_tool = FastMCPHUDClient.call_tool

        async def patched_call_tool(self, name, arguments=None):
            """Patched call_tool that converts certain errors to warnings."""
            try:
                # Call the original method
                result = await original_call_tool(self, name, arguments)

                # Check if it's an error related to output schema validation
                if result.isError and result.content:
                    error_text = ""
                    for content in result.content:
                        if isinstance(content, types.TextContent):
                            error_text += content.text

                    # Check for the specific error
                    if (
                        "has an output schema but did not return structured content" in error_text
                        or "Output validation error: outputSchema defined but no structured output returned"
                        in error_text
                    ):
                        logger.warning(
                            f"Tool '{name}' returned output schema validation error. "
                            f"Converting to warning and continuing. Original error: {error_text}"
                        )

                        # Convert to a successful result with warning
                        return MCPToolResult(
                            content=[
                                types.TextContent(
                                    text=f"Warning: {error_text}\n"
                                    f"The tool executed but didn't return structured content as expected. "
                                    f"This has been converted to a warning.",
                                    type="text",
                                )
                            ],
                            isError=False,
                            structuredContent=None,
                        )

                return result

            except Exception as e:
                # Check if the exception message contains our specific error
                error_msg = str(e)
                if (
                    "has an output schema but did not return structured content" in error_msg
                    or "Output validation error: outputSchema defined but no structured output returned"
                    in error_msg
                ):
                    logger.warning(
                        f"Tool '{name}' raised output schema validation error. "
                        f"Converting to warning and continuing. Original error: {error_msg}"
                    )

                    # Convert to a successful result with warning
                    return MCPToolResult(
                        content=[
                            types.TextContent(
                                text=f"Warning: {error_msg}\n"
                                f"The tool executed but didn't return structured content as expected. "
                                f"This has been converted to a warning.",
                                type="text",
                            )
                        ],
                        isError=False,
                        structuredContent=None,
                    )
                else:
                    # Re-raise other exceptions
                    raise

        # Apply the patch
        FastMCPHUDClient.call_tool = patched_call_tool
        logger.info("Successfully patched FastMCPHUDClient.call_tool for graceful error handling")

    except Exception as e:
        logger.error(f"Failed to patch FastMCPHUDClient.call_tool: {e}")


def apply_all_patches():
    """Apply all client-side patches."""
    patch_mcp_client_call_tool()
