"""Jupyter execution tool following HUD BaseTool pattern."""

from __future__ import annotations

import os, logging
from typing import TYPE_CHECKING, Optional

from hud.tools.base import BaseTool
from hud.tools.types import ContentResult, ToolError

from ..config import SOLUTIONS_PATH

if TYPE_CHECKING:
    from mcp.types import ContentBlock
    from .jupyter import JupyterKernel

logger = logging.getLogger(__name__)


class JupyterTool(BaseTool):
    """
    A tool that executes Python code in a Jupyter kernel.
    Follows the original SpreadsheetBench architecture: one kernel per container.

    Provides methods for spreadsheet manipulation:
    - execute_code: Direct Python code execution
    """

    def __init__(self, kernel: JupyterKernel | None = None) -> None:
        """Initialize JupyterTool with a kernel.

        Args:
            kernel: The Jupyter kernel instance to execute code on
        """
        super().__init__(
            env=kernel,
            name="jupyter",
            title="Jupyter Spreadsheet Tools",
            description="Execute Python code and manipulate XLSX files in a Jupyter kernel",
        )
        # Track successfully executed code for generalization testing
        self.execution_history: list[str] = []

    @property
    def kernel(self) -> JupyterKernel | None:
        """Get the current kernel (alias for env)."""
        return self.env

    @kernel.setter
    def kernel(self, value: JupyterKernel | None) -> None:
        """Set the kernel (alias for env)."""
        self.env = value

    async def __call__(self, code: str, timeout: int = 15) -> list[ContentBlock]:
        """Execute Python code in the Jupyter kernel (default method).

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default: 15)

        Returns:
            List of ContentBlock with execution results
        """
        return await self.execute_code(code, timeout)

    async def execute_code(self, code: str, timeout: int = 15) -> list[ContentBlock]:
        """Execute Python code in the Jupyter kernel.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default: 15)

        Returns:
            List of ContentBlock with execution results
        """
        if self.kernel is None:
            raise ToolError("Jupyter kernel is not initialized")

        try:
            # Execute code on the kernel
            result = await self.kernel.execute(code, timeout)

            # Check for timeout
            if result.startswith("[Execution timed out"):
                return ContentResult(
                    output="",
                    error=result,
                ).to_content_blocks()

            # Record successfully executed code
            is_error = "-----" in result or "Error" in result or "Traceback" in result
            if not is_error:
                self.execution_history.append(code)

                # Append code to solution file
                with open(os.path.join(SOLUTIONS_PATH, "1_solution.py"), "a") as f:
                    f.write(code)
                    f.write("\n\n")

            # Successful execution
            return ContentResult(
                output=result if result.strip() else "Code executed successfully (no output)",
            ).to_content_blocks()

        except Exception as e:
            logger.error(f"Jupyter execution error: {e}")
            raise ToolError(f"Execution failed: {str(e)}") from e

