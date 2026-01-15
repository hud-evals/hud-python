"""Playwright browser management for PDF viewing with headed Chrome."""

import asyncio
import base64
import logging
import os
import subprocess
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Browser, Page, Playwright

logger = logging.getLogger(__name__)


class PDFHTTPServer:
    """Simple HTTP server to serve PDF files."""

    def __init__(self, directory: str, port: int = 8765):
        self.directory = directory
        self.port = port
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self):
        """Start the HTTP server in a background thread."""
        if self.server:
            return

        handler = partial(SimpleHTTPRequestHandler, directory=self.directory)
        self.server = HTTPServer(("127.0.0.1", self.port), handler)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"HTTP server started on port {self.port} serving {self.directory}")

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            self.thread = None
            logger.info("HTTP server stopped")


class PDFBrowser:
    """Manages a Playwright browser for PDF viewing and interaction."""

    def __init__(self):
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.pdf_path: str | None = None
        self.solution_path: str | None = None
        self._width: int = 1280
        self._height: int = 800
        self._http_server: PDFHTTPServer | None = None
        self._xvfb_proc: subprocess.Popen | None = None

    async def _ensure_display(self):
        """Ensure Xvfb display is running."""
        display = os.environ.get("DISPLAY", ":1")
        x11_socket = f"/tmp/.X11-unix/X{display.replace(':', '')}"

        if Path(x11_socket).exists():
            logger.info(f"X11 display {display} already running")
            return

        # Start Xvfb
        self._xvfb_proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", f"{self._width}x{self._height}x24"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Started Xvfb on display {display}")

        # Wait for X11 to be ready
        for _ in range(50):  # 5 seconds max
            if Path(x11_socket).exists():
                logger.info(f"X11 display {display} is ready")
                return
            await asyncio.sleep(0.1)

        raise TimeoutError("Xvfb failed to start")

    async def start(self, headless: bool = False):
        """Start the browser (headed by default for PDF viewing)."""
        if self.playwright is None:
            # Ensure display is available for headed mode
            if not headless:
                await self._ensure_display()

            self.playwright = await async_playwright().start()

            display = os.environ.get("DISPLAY", ":1")

            self.browser = await self.playwright.firefox.launch(
                headless=headless,
                env={**os.environ, "DISPLAY": display},
                firefox_user_prefs={
                    # Use Firefox's built-in PDF.js viewer
                    "pdfjs.disabled": False,
                    # Don't ask to download PDFs
                    "browser.download.folderList": 2,
                    "browser.helperApps.neverAsk.saveToDisk": "",
                    "browser.download.manager.showWhenStarting": False,
                },
            )
            self.page = await self.browser.new_page(
                viewport={"width": self._width, "height": self._height}
            )
            logger.info(f"Browser started (headless={headless}, display={display})")

    async def stop(self):
        """Stop the browser."""
        if self._http_server:
            self._http_server.stop()
            self._http_server = None
        if self.page:
            await self.page.close()
            self.page = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        if self._xvfb_proc and self._xvfb_proc.poll() is None:
            self._xvfb_proc.terminate()
            await asyncio.sleep(0.5)
            if self._xvfb_proc.poll() is None:
                self._xvfb_proc.kill()
            self._xvfb_proc = None
            logger.info("Xvfb stopped")
        logger.info("Browser stopped")

    async def load_pdf(self, pdf_path: str, solution_path: str | None = None) -> dict[str, Any]:
        """Load a PDF file in the browser.

        Uses an HTTP server to serve the PDF so Chrome displays it inline.
        """
        if not self.page:
            await self.start()

        if not os.path.exists(pdf_path):
            return {"error": f"PDF not found: {pdf_path}"}

        self.pdf_path = pdf_path
        self.solution_path = solution_path

        # Start HTTP server to serve the PDF directory
        pdf_dir = str(Path(pdf_path).parent)
        pdf_name = Path(pdf_path).name

        if self._http_server:
            self._http_server.stop()

        self._http_server = PDFHTTPServer(directory=pdf_dir, port=8765)
        self._http_server.start()
        await asyncio.sleep(0.5)  # Give server time to start

        # Navigate to PDF via HTTP URL
        http_url = f"http://127.0.0.1:8765/{pdf_name}"

        try:
            await self.page.goto(http_url, wait_until="load", timeout=30000)
            await asyncio.sleep(2)  # Wait for PDF to render

            logger.info(f"Loaded PDF via HTTP: {http_url}")

            # Take initial screenshot
            screenshot = await self.screenshot()

            return {
                "success": True,
                "pdf_path": pdf_path,
                "url": http_url,
                "viewport": {"width": self._width, "height": self._height},
                "screenshot": screenshot,
            }
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            return {"error": str(e)}

    async def screenshot(self) -> str | None:
        """Take a screenshot and return base64 encoded image."""
        if not self.page:
            return None

        try:
            screenshot_bytes = await self.page.screenshot(full_page=False)
            return base64.b64encode(screenshot_bytes).decode()
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    async def click(self, x: int, y: int) -> dict[str, Any]:
        """Click at coordinates."""
        if not self.page:
            return {"error": "Browser not started"}

        try:
            await self.page.mouse.click(x, y)
            await asyncio.sleep(0.3)  # Wait for any UI updates
            screenshot = await self.screenshot()
            return {"success": True, "x": x, "y": y, "screenshot": screenshot}
        except Exception as e:
            return {"error": str(e)}

    async def type_text(self, text: str) -> dict[str, Any]:
        """Type text."""
        if not self.page:
            return {"error": "Browser not started"}

        try:
            await self.page.keyboard.type(text)
            await asyncio.sleep(0.2)
            screenshot = await self.screenshot()
            return {"success": True, "text": text, "screenshot": screenshot}
        except Exception as e:
            return {"error": str(e)}

    async def press_key(self, key: str) -> dict[str, Any]:
        """Press a key."""
        if not self.page:
            return {"error": "Browser not started"}

        try:
            await self.page.keyboard.press(key)
            await asyncio.sleep(0.2)
            screenshot = await self.screenshot()
            return {"success": True, "key": key, "screenshot": screenshot}
        except Exception as e:
            return {"error": str(e)}

    async def scroll(self, delta_x: int = 0, delta_y: int = 0) -> dict[str, Any]:
        """Scroll the page."""
        if not self.page:
            return {"error": "Browser not started"}

        try:
            await self.page.mouse.wheel(delta_x, delta_y)
            await asyncio.sleep(0.3)
            screenshot = await self.screenshot()
            return {"success": True, "delta_x": delta_x, "delta_y": delta_y, "screenshot": screenshot}
        except Exception as e:
            return {"error": str(e)}

    async def save_pdf(self, output_path: str) -> dict[str, Any]:
        """Save/print the PDF.

        Note: In browser context, we use Ctrl+S or print to PDF.
        For this implementation, we'll use PyMuPDF to save the actual PDF state.
        """
        if not self.page:
            return {"error": "Browser not started"}

        try:
            # Use keyboard shortcut to trigger save
            await self.page.keyboard.press("Control+s")
            await asyncio.sleep(1)

            # Note: Browser save dialogs are tricky to handle
            # In practice, we may need to use PyMuPDF to extract filled values
            return {"success": True, "note": "Save dialog triggered. Output path: " + output_path}
        except Exception as e:
            return {"error": str(e)}


# Global browser instance
pdf_browser = PDFBrowser()
