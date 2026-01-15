"""MCP server for PDF form filling environment."""

import logging
import os
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress MCP server logs
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

from hud.server import MCPServer

from .context import pdf_context
from .tools import ListFieldsTool, FillFieldTool, GetFieldTool, SavePdfTool
from .setup import setup as setup_hub
from .evaluate import evaluate as evaluate_hub

# Create main server
mcp = MCPServer(name="hud-pdfbench")


@mcp.initialize
async def initialize_environment(ctx):
    """Initialize the PDF environment."""
    logger.info("Initializing PDFBench environment...")

    # Set up context on hubs
    setup_hub.env = pdf_context
    evaluate_hub.env = pdf_context

    # Mount hubs
    mcp.mount(setup_hub)
    mcp.mount(evaluate_hub)

    # Register action tools
    mcp.add_tool(ListFieldsTool(env=pdf_context))
    mcp.add_tool(FillFieldTool(env=pdf_context))
    mcp.add_tool(GetFieldTool(env=pdf_context))
    mcp.add_tool(SavePdfTool(env=pdf_context))

    # Auto-load PDF if path provided via environment
    pdf_path = os.getenv("PDF_PATH")
    if pdf_path:
        logger.info(f"Auto-loading PDF from PDF_PATH: {pdf_path}")
        result = pdf_context.load_pdf(
            pdf_path=pdf_path,
            output_path=os.getenv("OUTPUT_PATH"),
            solution_path=os.getenv("SOLUTION_PATH"),
        )
        if "error" in result:
            logger.error(f"Failed to auto-load PDF: {result['error']}")
        else:
            logger.info(f"Loaded PDF with {result['field_count']} fields")

    logger.info("PDFBench environment ready")


if __name__ == "__main__":
    mcp.run()
