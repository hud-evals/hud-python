"""Task connection connector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hud.environment.connectors.mcp_config import MCPConfigConnectorMixin

if TYPE_CHECKING:
    from hud.types import Task

__all__ = ["TaskConnectorMixin"]

logger = logging.getLogger(__name__)


class TaskConnectorMixin(MCPConfigConnectorMixin):
    """Mixin providing connect_task() method.
    
    Inherits from MCPConfigConnectorMixin for connect_mcp_config().
    """

    def setup_tool(self, call: Any, /, **kwargs: Any) -> Any:
        raise NotImplementedError

    def evaluate_tool(self, call: Any, /, **kwargs: Any) -> Any:
        raise NotImplementedError

    def connect_task(self, slug: str) -> Any:
        """Connect to a task from the HUD platform.
        
        Fetches the task from api.hud.so immediately and applies configuration
        (mcp_config, setup_tool, evaluate_tool).
        
        Args:
            slug: Task slug in format "evalset/task_name" or "evalset/task_name@version".
        
        Returns:
            self for chaining.
        
        Example:
            ```python
            env = Environment("my-env").connect_task("my-org/browser-task")
            
            async with env:
                # Task's mcp_config is connected
                # Task's setup_tool runs automatically
                result = await env.call_tool("navigate", url="...")
                # Task's evaluate_tool runs on exit
            ```
        """
        import httpx
        
        from hud.settings import settings
        from hud.types import Task
        
        # Fetch task synchronously
        logger.info("Loading task from platform: %s", slug)
        
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        
        with httpx.Client() as client:
            response = client.get(
                f"{settings.hud_api_url}/tasks/{slug}",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        
        task = Task(**data)
        self._apply_task(task)
        logger.info("Task loaded and applied: %s", slug)
        return self

    def _apply_task(self, task: Task) -> None:
        """Apply a Task definition to this environment.
        
        Sets up:
            - MCP connections from task.mcp_config
            - Setup tool calls from task.setup_tool
            - Evaluate tool calls from task.evaluate_tool
        """
        # Connect MCP servers
        if task.mcp_config:
            self.connect_mcp_config(task.mcp_config)
        
        # Configure setup tool calls
        if task.setup_tool:
            setup_calls = task.setup_tool
            if not isinstance(setup_calls, list):
                setup_calls = [setup_calls]
            for call in setup_calls:
                self.setup_tool(call.name, **(call.arguments or {}))
        
        # Configure evaluate tool calls
        if task.evaluate_tool:
            eval_calls = task.evaluate_tool
            if not isinstance(eval_calls, list):
                eval_calls = [eval_calls]
            for call in eval_calls:
                self.evaluate_tool(call.name, **(call.arguments or {}))
