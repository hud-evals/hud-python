"""PDF manipulation tools."""

import logging
from typing import Any

from mcp.types import TextContent, ContentBlock
from hud.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ListFieldsTool(BaseTool):
    """Tool for listing PDF form fields."""

    def __init__(self, env: Any = None):
        super().__init__(
            env=env,
            name="list_fields",
            title="List Fields",
            description="List all form fields in the loaded PDF",
        )

    async def __call__(self, page: int | None = None) -> list[ContentBlock]:
        """List all form fields.

        Args:
            page: Optional page number to filter (0-indexed)
        """
        if self.env is None:
            return [TextContent(text="ERROR: No PDF loaded. Call setup first.", type="text")]

        result = self.env.list_fields(page)

        if "error" in result:
            return [TextContent(text=f"ERROR: {result['error']}", type="text")]

        fields = result["fields"]
        if not fields:
            return [TextContent(text="No form fields found.", type="text")]

        lines = [f"Found {len(fields)} fields:"]
        for f in fields:
            value = f.get("value", "")
            value_str = f" = '{value}'" if value else ""
            lines.append(f"  - {f['name']} ({f['type']}){value_str} [bbox: {f['bbox']}]")

        return [TextContent(text="\n".join(lines), type="text")]


class FillFieldTool(BaseTool):
    """Tool for filling PDF form fields."""

    def __init__(self, env: Any = None):
        super().__init__(
            env=env,
            name="fill_field",
            title="Fill Field",
            description="Fill a form field by name or bounding box",
        )

    async def __call__(
        self,
        value: str,
        field_name: str | None = None,
        bbox: str | None = None,
    ) -> list[ContentBlock]:
        """Fill a form field.

        Args:
            value: Value to fill in the field
            field_name: Name of the field to fill
            bbox: Bounding box key (format: "page,x0,y0,x1,y1") as alternative
        """
        if self.env is None:
            return [TextContent(text="ERROR: No PDF loaded. Call setup first.", type="text")]

        result = self.env.fill_field(field_name=field_name, bbox=bbox, value=value)

        if "error" in result:
            return [TextContent(text=f"ERROR: {result['error']}", type="text")]

        return [TextContent(
            text=f"Filled '{result['field_name']}' with '{result['value']}'",
            type="text"
        )]


class GetFieldTool(BaseTool):
    """Tool for getting a field's current value."""

    def __init__(self, env: Any = None):
        super().__init__(
            env=env,
            name="get_field",
            title="Get Field",
            description="Get the current value of a form field",
        )

    async def __call__(
        self,
        field_name: str | None = None,
        bbox: str | None = None,
    ) -> list[ContentBlock]:
        """Get a field's value.

        Args:
            field_name: Name of the field
            bbox: Bounding box key as alternative
        """
        if self.env is None:
            return [TextContent(text="ERROR: No PDF loaded. Call setup first.", type="text")]

        result = self.env.get_field(field_name=field_name, bbox=bbox)

        if "error" in result:
            return [TextContent(text=f"ERROR: {result['error']}", type="text")]

        return [TextContent(
            text=f"Field: {result['field_name']}\nType: {result['type']}\nValue: '{result['value']}'",
            type="text"
        )]


class SavePdfTool(BaseTool):
    """Tool for saving the filled PDF."""

    def __init__(self, env: Any = None):
        super().__init__(
            env=env,
            name="save_pdf",
            title="Save PDF",
            description="Save the filled PDF to a file",
        )

    async def __call__(self, output_path: str | None = None) -> list[ContentBlock]:
        """Save the filled PDF.

        Args:
            output_path: Path to save the PDF (default: path from setup)
        """
        if self.env is None:
            return [TextContent(text="ERROR: No PDF loaded. Call setup first.", type="text")]

        result = self.env.save_pdf(output_path)

        if "error" in result:
            return [TextContent(text=f"ERROR: {result['error']}", type="text")]

        return [TextContent(text=f"PDF saved to: {result['path']}", type="text")]
