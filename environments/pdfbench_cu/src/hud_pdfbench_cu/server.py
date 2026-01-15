"""MCP server for PDF form filling with Playwright browser automation."""

import logging
import os
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

from hud.server import MCPServer
from hud.tools import PlaywrightTool

from .browser import pdf_browser
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Create main server
mcp = MCPServer(name="hud-pdfbench-cu")

# Global playwright tool
playwright_tool: PlaywrightTool | None = None


class PDFPlaywrightTool(PlaywrightTool):
    """PlaywrightTool that uses the shared pdf_browser page."""

    async def _ensure_browser(self) -> None:
        """Use the pdf_browser's page instead of launching new browser."""
        if pdf_browser.page is not None:
            self.env = pdf_browser.page
            self._browser = pdf_browser.browser
            self._browser_context = pdf_browser.page.context if pdf_browser.page else None
            logger.info("Using pdf_browser's existing page")
        else:
            # Fallback to parent implementation
            await super()._ensure_browser()


@mcp.initialize
async def initialize_environment(ctx):
    """Initialize the PDF Playwright environment."""
    global playwright_tool
    logger.info("Initializing PDFBench Playwright environment...")

    # Start browser (headed for PDF viewing via Xvfb)
    await pdf_browser.start(headless=False)

    # Auto-load PDF if path provided
    pdf_path = os.getenv("PDF_PATH")
    if pdf_path:
        logger.info(f"Auto-loading PDF: {pdf_path}")
        solution_path = os.getenv("SOLUTION_PATH")
        result = await pdf_browser.load_pdf(pdf_path, solution_path)
        if "error" in result:
            logger.error(f"Failed to load PDF: {result['error']}")
        else:
            logger.info("PDF loaded successfully")

    # Create PlaywrightTool with the browser's page
    playwright_tool = PDFPlaywrightTool(page=pdf_browser.page)

    # Mount hubs and tools
    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)
    mcp.add_tool(playwright_tool)

    logger.info("PDFBench Playwright environment ready")


@mcp.shutdown
async def shutdown_environment():
    """Cleanup on shutdown."""
    logger.info("Shutting down PDFBench Playwright...")
    await pdf_browser.stop()


if __name__ == "__main__":
    mcp.run()
