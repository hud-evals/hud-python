"""Mock functionality for Environment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types

from hud.types import MCPToolResult

if TYPE_CHECKING:
    from hud.environment.environment import Environment

__all__ = ["MockMixin", "generate_mock_value"]

logger = logging.getLogger(__name__)


def generate_mock_value(schema: dict[str, Any], depth: int = 0) -> Any:
    """Generate a reasonable mock value from a JSON schema.

    Args:
        schema: JSON schema dict with 'type', 'properties', etc.
        depth: Current recursion depth (to prevent infinite loops).

    Returns:
        A mock value that matches the schema.
    """
    if depth > 10:  # Prevent infinite recursion
        return None

    # Handle $ref - we don't resolve refs, just return placeholder
    if "$ref" in schema:
        return {}

    # Handle anyOf/oneOf/allOf - pick first option
    if "anyOf" in schema:
        return generate_mock_value(schema["anyOf"][0], depth + 1)
    if "oneOf" in schema:
        return generate_mock_value(schema["oneOf"][0], depth + 1)
    if "allOf" in schema:
        # Merge all schemas
        merged: dict[str, Any] = {}
        for sub_schema in schema["allOf"]:
            result = generate_mock_value(sub_schema, depth + 1)
            if isinstance(result, dict):
                merged.update(result)
        return merged

    # Check for const or enum first
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0] if schema["enum"] else None

    # Check for default value
    if "default" in schema:
        return schema["default"]

    # Handle by type
    schema_type = schema.get("type")

    if schema_type == "string":
        # Check for format hints
        fmt = schema.get("format", "")
        if fmt == "uri" or fmt == "url":
            return "https://example.com"
        if fmt == "email":
            return "user@example.com"
        if fmt == "date":
            return "2024-01-01"
        if fmt == "date-time":
            return "2024-01-01T00:00:00Z"
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        # Use title/description hint if available
        title = schema.get("title", "").lower()
        if "url" in title or "link" in title:
            return "https://example.com"
        if "name" in title:
            return "mock_name"
        if "id" in title:
            return "mock_id"
        return "mock_string"

    if schema_type == "number" or schema_type == "integer":
        # Check for bounds
        minimum = schema.get("minimum", 0)
        maximum = schema.get("maximum", 100)
        if schema_type == "integer":
            return int((minimum + maximum) / 2) if maximum != float("inf") else minimum
        return float((minimum + maximum) / 2) if maximum != float("inf") else float(minimum)

    if schema_type == "boolean":
        return True

    if schema_type == "null":
        return None

    if schema_type == "array":
        items_schema = schema.get("items", {})
        if items_schema:
            # Generate one item
            return [generate_mock_value(items_schema, depth + 1)]
        return []

    if schema_type == "object" or "properties" in schema:
        result: dict[str, Any] = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            # Only include required properties or first few optional ones
            if prop_name in required or len(result) < 3:
                result[prop_name] = generate_mock_value(prop_schema, depth + 1)

        return result

    # Handle list of types
    if isinstance(schema_type, list):
        # Pick first non-null type
        for t in schema_type:
            if t != "null":
                return generate_mock_value({"type": t}, depth + 1)
        return None

    # Fallback for unknown schema
    return None


def generate_mock_tool_result(tool: mcp_types.Tool) -> MCPToolResult:
    """Generate a mock result for a tool based on its output schema.

    Args:
        tool: MCP Tool with inputSchema and optionally outputSchema.

    Returns:
        MCPToolResult with mock content.
    """
    # Check if tool has an output schema
    output_schema = getattr(tool, "outputSchema", None)

    if output_schema:
        mock_value = generate_mock_value(output_schema)
        content_text = str(mock_value) if mock_value is not None else "mock_result"
    else:
        # Generate a sensible default based on tool name
        tool_name = tool.name
        if "screenshot" in tool_name.lower() or "image" in tool_name.lower():
            content_text = "[mock image data]"
        elif "get" in tool_name.lower() or "list" in tool_name.lower():
            content_text = "[]"
        elif "check" in tool_name.lower() or "verify" in tool_name.lower():
            content_text = "true"
        elif "count" in tool_name.lower():
            content_text = "0"
        else:
            content_text = "mock_success"

    return MCPToolResult(
        content=[mcp_types.TextContent(type="text", text=content_text)],
        isError=False,
    )


class MockMixin:
    """Mixin that adds mock functionality to Environment.

    When mock mode is enabled:
    - All tool calls return mock values instead of executing
    - Specific tools can have custom mock outputs via mock_tool()
    - Tools are automatically mocked with reasonable defaults based on their schemas

    Usage:
        env = Environment("test").connect_hub("browser")
        env.mock()  # Enable mock mode

        # Set specific mock outputs
        env.mock_tool("navigate", "Navigation successful")
        env.mock_tool("screenshot", {"image": "base64data..."})

        async with env:
            result = await env.call_tool("navigate", url="https://example.com")
            # Returns: MCPToolResult with "Navigation successful"
    """

    _mock_mode: bool
    _mock_outputs: dict[str, Any]
    _mock_tool_schemas: dict[str, mcp_types.Tool]

    def _init_mock(self) -> None:
        """Initialize mock state. Called from Environment.__init__."""
        self._mock_mode = False
        self._mock_outputs = {}
        self._mock_tool_schemas = {}

    def mock(self) -> Environment:
        """Enable mock mode - all tool calls will return mock values.

        Returns:
            self for chaining.

        Example:
            env = Environment("test").connect_hub("browser").mock()
        """
        self._mock_mode = True
        logger.info("Mock mode enabled for environment %s", getattr(self, "name", "unknown"))
        return self  # type: ignore[return-value]

    def unmock(self) -> Environment:
        """Disable mock mode - tool calls will execute normally.

        Returns:
            self for chaining.
        """
        self._mock_mode = False
        logger.info("Mock mode disabled for environment %s", getattr(self, "name", "unknown"))
        return self  # type: ignore[return-value]

    @property
    def is_mock(self) -> bool:
        """Check if mock mode is enabled."""
        return self._mock_mode

    def mock_tool(self, name: str, output: Any) -> Environment:
        """Set a specific mock output for a tool.

        Args:
            name: Tool name (with prefix if applicable).
            output: The value to return when this tool is called.
                   Can be a string, dict, or any JSON-serializable value.

        Returns:
            self for chaining.

        Example:
            env.mock_tool("navigate", "Success")
            env.mock_tool("screenshot", {"type": "image", "data": "..."})
            env.mock_tool("get_elements", [{"id": "1", "text": "Button"}])
        """
        self._mock_outputs[name] = output
        logger.debug("Mock output set for tool %s", name)
        return self  # type: ignore[return-value]

    def _get_mock_result(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Get mock result for a tool call.

        Priority:
        1. Custom mock output set via mock_tool()
        2. Auto-generated mock based on tool's output schema
        3. Default mock value

        Args:
            name: Tool name.
            arguments: Tool arguments (for potential future use).

        Returns:
            MCPToolResult with mock content.
        """
        # Check for custom mock output
        if name in self._mock_outputs:
            output = self._mock_outputs[name]
            # Convert to string if not already
            if isinstance(output, str):
                content_text = output
            else:
                import json

                try:
                    content_text = json.dumps(output)
                except (TypeError, ValueError):
                    content_text = str(output)

            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=content_text)],
                isError=False,
            )

        # Try to find tool schema for auto-generation
        if name in self._mock_tool_schemas:
            return generate_mock_tool_result(self._mock_tool_schemas[name])

        # Check router for tool schema
        router = getattr(self, "_router", None)
        if router:
            for tool in router.tools:
                if tool.name == name:
                    self._mock_tool_schemas[name] = tool
                    return generate_mock_tool_result(tool)

        # Default fallback
        return MCPToolResult(
            content=[mcp_types.TextContent(type="text", text="mock_success")],
            isError=False,
        )

    def _populate_mock_schemas(self) -> None:
        """Populate mock tool schemas from router after connection.

        Called after _build_routing to cache tool schemas for mock generation.
        """
        router = getattr(self, "_router", None)
        if router:
            for tool in router.tools:
                self._mock_tool_schemas[tool.name] = tool
