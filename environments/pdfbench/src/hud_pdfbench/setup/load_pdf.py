"""Load PDF setup function."""

from mcp.types import TextContent, ContentBlock
from . import setup


@setup.tool("load_pdf")
async def setup_load_pdf(
    pdf_path: str,
    output_path: str | None = None,
    solution_path: str | None = None,
) -> list[ContentBlock]:
    """Initialize the environment with a blank PDF form.

    Args:
        pdf_path: Path to the blank PDF form
        output_path: Path where filled PDF will be saved
        solution_path: Path to solution.json for evaluation
    """
    ctx = setup.env
    result = ctx.load_pdf(pdf_path, output_path, solution_path)

    if "error" in result:
        return [TextContent(text=f"ERROR: {result['error']}", type="text")]

    # Format field list for display
    fields_text = "\n".join(
        f"  - {f['name']} ({f['type']}) [bbox: {f['bbox']}]"
        for f in result.get("fields", [])
    )

    text = f"""PDF loaded successfully!
- Path: {result['pdf_path']}
- Output: {result['output_path']}
- Total fields: {result['total_fields']}

First {len(result.get('fields', []))} fields:
{fields_text}

Use list_fields to see all fields, fill_field to fill them, and save_pdf when done."""

    return [TextContent(text=text, type="text")]
