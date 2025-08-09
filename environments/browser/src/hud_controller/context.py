"""Global context for browser environment."""

from typing import Dict, Optional, Any
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)


class BrowserEnvironmentContext:
    """Context object providing access to browser and app state for all environment operations.

    This context can be used for setup, evaluation, and general environment interactions.
    """

    def __init__(self, service_manager, playwright_tool=None):
        """Initialize the environment context.

        Args:
            service_manager: ServiceManager instance for app discovery
            playwright_tool: Optional PlaywrightTool for browser interactions
        """
        self.service_manager = service_manager
        self.playwright = playwright_tool
        self._app_ports: Dict[str, int] = {}
        self._http_client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Clean up resources."""
        await self._http_client.aclose()

    # === App Discovery and Communication ===

    def get_app_port(self, app_name: str) -> int:
        """Get the port for a running app.

        Args:
            app_name: Name of the app (e.g., 'todo', 'gmail')

        Returns:
            Port number where the app is running

        Raises:
            ValueError: If app is not running or not found
        """
        if app_name not in self._app_ports:
            try:
                port = self.service_manager.get_app_port(app_name)
                self._app_ports[app_name] = port
                logger.info(f"Discovered app '{app_name}' running on port {port}")
            except Exception as e:
                raise ValueError(f"Could not find running app '{app_name}': {e}")

        return self._app_ports[app_name]

    def get_app_url(self, app_name: str) -> str:
        """Get the base URL for an app."""
        port = self.get_app_port(app_name)
        return f"http://localhost:{port}"

    def list_running_apps(self) -> list[str]:
        """Get list of currently running apps."""
        try:
            return self.service_manager.list_running_apps()
        except Exception as e:
            logger.error(f"Failed to list running apps: {e}")
            return []

    async def call_app_api(
        self, app_name: str, endpoint: str, method: str = "GET", **kwargs
    ) -> dict:
        """Make an API call to an app's backend.

        Args:
            app_name: Name of the app
            endpoint: API endpoint (e.g., '/api/eval/stats')
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for httpx request

        Returns:
            JSON response from the app

        Raises:
            Exception: If the API call fails after retries
        """
        app_url = self.get_app_url(app_name)
        full_url = f"{app_url}{endpoint}"

        logger.debug(f"Making {method} request to {full_url}")

        # Retry logic to handle app startup timing
        max_retries = 3
        retry_delay = 2.0  # seconds

        for attempt in range(max_retries):
            try:
                response = await self._http_client.request(method, full_url, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"HTTP error calling {full_url} (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"HTTP error calling {full_url} after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Error calling {full_url} (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Error calling {full_url} after {max_retries} attempts: {e}")
                    raise

    # === Browser Interactions ===

    async def get_page_content(self) -> str:
        """Get the current page content."""
        if not self.playwright:
            raise ValueError("PlaywrightTool not available")

        try:
            return await self.playwright.get_page_content()
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return ""

    async def get_current_url(self) -> str:
        """Get the current page URL."""
        if not self.playwright:
            raise ValueError("PlaywrightTool not available")

        try:
            return await self.playwright.get_current_url()
        except Exception as e:
            logger.error(f"Failed to get current URL: {e}")
            return ""

    async def get_screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        if not self.playwright:
            raise ValueError("PlaywrightTool not available")

        try:
            return await self.playwright.screenshot()
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return b""

    async def count_elements(self, selector: str) -> int:
        """Count elements matching a CSS selector."""
        if not self.playwright:
            raise ValueError("PlaywrightTool not available")

        try:
            return await self.playwright.count_elements(selector)
        except Exception as e:
            logger.error(f"Failed to count elements '{selector}': {e}")
            return 0

    # === Environment Operations ===

    async def execute_setup(self, setup_spec: dict) -> dict:
        """Execute a setup operation using the setup tool.

        Args:
            setup_spec: Setup specification with 'function' and 'args'

        Returns:
            Setup result dictionary
        """
        from .setup import setup_tool

        function = setup_spec.get("function")
        args = setup_spec.get("args", {})

        # Call the setup tool directly
        return await setup_tool(function, args, None, None, self)

    async def execute_evaluation(self, eval_spec: dict) -> dict:
        """Execute an evaluation operation using the evaluate tool.

        Args:
            eval_spec: Evaluation specification with 'function' and 'args'

        Returns:
            Evaluation result dictionary
        """
        from .evaluators import evaluate_tool

        function = eval_spec.get("function")
        args = eval_spec.get("args", {})

        # Call the evaluate tool directly
        return await evaluate_tool(function, args, None, None, self)

    # === Utility Methods ===

    def get_metadata(self) -> dict:
        """Get context metadata for debugging."""
        return {
            "running_apps": self.list_running_apps(),
            "app_ports": self._app_ports,
            "has_playwright": self.playwright is not None,
        }


# Keep backward compatibility alias
BrowserEvaluationContext = BrowserEnvironmentContext


# Global context instance that will be shared by setup and evaluate tools
_global_context: Optional[BrowserEnvironmentContext] = None


def get_global_context() -> Optional[BrowserEnvironmentContext]:
    """Get the global browser environment context."""
    return _global_context


def set_global_context(context: BrowserEnvironmentContext) -> None:
    """Set the global browser environment context."""
    global _global_context
    _global_context = context
    logger.info("Global context initialized")


def initialize_context(service_manager, playwright_tool=None) -> BrowserEnvironmentContext:
    """Initialize and set the global context."""
    context = BrowserEnvironmentContext(service_manager, playwright_tool)
    set_global_context(context)
    return context
