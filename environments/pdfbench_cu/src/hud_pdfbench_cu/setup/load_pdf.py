"""Load PDF setup function for computer use."""

from mcp.types import TextContent, ImageContent, ContentBlock
from . import setup


@setup.tool("load_pdf")
async def setup_load_pdf(
    pdf_path: str,
    solution_path: str | None = None,
) -> list[ContentBlock]:
    """Load a PDF form in the browser for visual interaction.

    Args:
        pdf_path: Path to the PDF file
        solution_path: Path to solution.json for evaluation
    """
    from ..browser import pdf_browser

    # Start browser if needed
    if not pdf_browser.page:
        await pdf_browser.start(headless=True)

    result = await pdf_browser.load_pdf(pdf_path, solution_path)

    if "error" in result:
        return [TextContent(text=f"ERROR: {result['error']}", type="text")]

    content = [
        TextContent(
            text=f"""PDF loaded in browser!
- Path: {result['pdf_path']}
- Viewport: {result['viewport']['width']}x{result['viewport']['height']}

The PDF form is now displayed in the browser. You can interact with it using the computer tool:
- Use 'screenshot' action to see the current state
- Use 'left_click' with coordinates to click on form fields
- Use 'type' to enter text into the focused field
- Use 'key' to press Tab, Enter, or other keys
- Use 'scroll' to navigate the PDF

Fill out all the form fields as instructed, then the evaluation will verify your work.""",
            type="text"
        )
    ]

    # Include initial screenshot
    if result.get("screenshot"):
        content.append(ImageContent(
            type="image",
            data=result["screenshot"],
            mimeType="image/png"
        ))

    return content
