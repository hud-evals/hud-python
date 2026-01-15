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
from hud.tools import AnthropicComputerTool

from .browser import pdf_browser
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub
from .executor import PDFBrowserExecutor

# Create main server
mcp = MCPServer(name="hud-pdfbench-cu")

# Global computer tool
computer_tool: AnthropicComputerTool | None = None


@mcp.initialize
async def initialize_environment(ctx):
    """Initialize the PDF Computer Use environment."""
    global computer_tool
    logger.info("Initializing PDFBench Computer Use environment...")

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

    # Create executor with the browser's page
    executor = PDFBrowserExecutor(pdf_browser)

    # Create AnthropicComputerTool with our executor
    # Use name="computer" to match what ClaudeAgent expects
    computer_tool = AnthropicComputerTool(executor=executor, name="computer")

    # Mount hubs and tools
    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)
    mcp.add_tool(computer_tool)

    logger.info("PDFBench Computer Use environment ready")


@mcp.shutdown
async def shutdown_environment():
    """Cleanup on shutdown."""
    logger.info("Shutting down PDFBench Computer Use...")
    await pdf_browser.stop()


if __name__ == "__main__":
    mcp.run()
