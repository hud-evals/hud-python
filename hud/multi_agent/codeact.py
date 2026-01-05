"""CodeAct: Agent writes Python code instead of JSON tool calls.

This module implements the CodeAct pattern:
- Agent generates and executes Python code in a sandbox
- Three-layer action space: L1 Atomic, L2 Sandbox CLI, L3 Packages
- Stack traces are preserved for agent debugging
- Counter-intuitive: Keep errors visible, don't erase them
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.tools.jupyter import JupyterTool
    from hud.tools.shell import ShellTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str
    error: str | None = None
    return_value: Any = None
    duration_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Metadata
    code: str = ""
    layer: str = "L1"  # L1, L2, or L3

    def to_context(self) -> str:
        """Format result for inclusion in context.

        Key principle: Keep errors visible so agent can debug.
        """
        if self.success:
            result = f"[Execution Success]\n{self.output}"
            if self.return_value is not None:
                result += f"\nReturn value: {self.return_value}"
        else:
            # IMPORTANT: Keep full stack trace for agent debugging
            result = f"[Execution Error]\n{self.error}"
            if self.output:
                result += f"\n\nOutput before error:\n{self.output}"

        if self.duration_ms:
            result += f"\n\n(Executed in {self.duration_ms:.1f}ms)"

        return result


class CodeActExecutor:
    """Execute Python code in sandbox, return result + errors.

    Three-Layer Action Space:
    - L1 Atomic: read_file, write_file, shell (~10 basic functions)
    - L2 Sandbox: ffmpeg, grep, curl (CLI tools via shell)
    - L3 Packages: requests, pandas, PIL (PyPI libraries)

    Example:
        executor = CodeActExecutor(jupyter_tool)

        # Agent generates code
        code = '''
        import requests
        response = requests.get("https://api.example.com/data")
        data = response.json()
        '''

        result = await executor.execute(code)
        if not result.success:
            # Stack trace is preserved for agent to debug
            print(result.error)
    """

    # L1 Atomic operations - always available
    L1_BUILTINS = {
        "read_file": """
def read_file(path: str) -> str:
    '''Read file contents.'''
    with open(path) as f:
        return f.read()
""",
        "write_file": """
def write_file(path: str, content: str) -> str:
    '''Write content to file.'''
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"
""",
        "append_file": """
def append_file(path: str, content: str) -> str:
    '''Append content to file.'''
    with open(path, 'a') as f:
        f.write(content)
    return f"Appended {len(content)} bytes to {path}"
""",
        "list_dir": """
def list_dir(path: str = '.') -> list[str]:
    '''List directory contents.'''
    import os
    return os.listdir(path)
""",
        "shell": """
import subprocess
def shell(cmd: str, timeout: int = 60) -> str:
    '''Execute shell command.'''
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    output = result.stdout
    if result.stderr:
        output += f"\\nSTDERR: {result.stderr}"
    if result.returncode != 0:
        output += f"\\nExit code: {result.returncode}"
    return output
""",
        "glob_files": """
def glob_files(pattern: str) -> list[str]:
    '''Find files matching glob pattern.'''
    import glob
    return glob.glob(pattern, recursive=True)
""",
        "file_exists": """
def file_exists(path: str) -> bool:
    '''Check if file exists.'''
    import os
    return os.path.exists(path)
""",
        "get_cwd": """
def get_cwd() -> str:
    '''Get current working directory.'''
    import os
    return os.getcwd()
""",
        "make_dir": """
def make_dir(path: str) -> str:
    '''Create directory (including parents).'''
    import os
    os.makedirs(path, exist_ok=True)
    return f"Created directory: {path}"
""",
        "delete_file": """
def delete_file(path: str) -> str:
    '''Delete a file.'''
    import os
    os.remove(path)
    return f"Deleted: {path}"
""",
    }

    def __init__(
        self,
        jupyter_tool: JupyterTool | None = None,
        shell_tool: ShellTool | None = None,
        timeout: int = 60,
        keep_errors: bool = True,
        workspace: str | None = None,
    ) -> None:
        """Initialize the CodeAct executor.

        Args:
            jupyter_tool: Optional JupyterTool for Python execution
            shell_tool: Optional ShellTool for shell execution
            timeout: Default timeout in seconds
            keep_errors: Whether to preserve stack traces (recommended: True)
            workspace: Working directory for code execution
        """
        self.jupyter = jupyter_tool
        self.shell = shell_tool
        self.timeout = timeout
        self.keep_errors = keep_errors
        self.workspace = workspace

        # Track execution history for debugging
        self._history: list[ExecutionResult] = []

    async def _ensure_jupyter(self) -> JupyterTool:
        """Ensure JupyterTool is available and connected."""
        if self.jupyter is None:
            from hud.tools.jupyter import JupyterTool

            self.jupyter = JupyterTool()
            # JupyterTool auto-connects on first call via _ensure_kernel()
            await self.jupyter._ensure_kernel()
        return self.jupyter

    def _inject_builtins(self, code: str) -> str:
        """Inject L1 atomic functions into code if not already defined."""
        lines_to_add = []

        for name, func_code in self.L1_BUILTINS.items():
            # Only inject if the function is used but not defined
            if name in code and f"def {name}" not in code:
                lines_to_add.append(func_code)

        if lines_to_add:
            return "\n".join(lines_to_add) + "\n\n" + code
        return code

    async def execute(
        self,
        code: str,
        timeout: int | None = None,
        inject_builtins: bool = True,
    ) -> ExecutionResult:
        """Execute Python code in sandbox.

        Args:
            code: Python code to execute
            timeout: Optional timeout override
            inject_builtins: Whether to inject L1 atomic functions

        Returns:
            ExecutionResult with output, errors, and metadata
        """
        import time

        start_time = time.time()
        timeout = timeout or self.timeout

        # Inject L1 builtins if needed
        if inject_builtins:
            code = self._inject_builtins(code)

        try:
            jupyter = await self._ensure_jupyter()

            # Execute via Jupyter
            result_blocks = await jupyter(code=code)

            # Parse result from content blocks
            output_parts = []
            return_value = None

            for block in result_blocks:
                if hasattr(block, "text"):
                    text = getattr(block, "text", "")
                    if text.startswith("Error:"):
                        # Error occurred
                        duration = (time.time() - start_time) * 1000
                        result = ExecutionResult(
                            success=False,
                            output="",
                            error=text,
                            code=code,
                            duration_ms=duration,
                            layer=self._detect_layer(code),
                        )
                        self._history.append(result)
                        return result
                    output_parts.append(text)
                elif hasattr(block, "data"):
                    # Image or other data
                    mime_type = getattr(block, "mimeType", "unknown")
                    output_parts.append(f"[Data: {mime_type}]")

            output = "\n".join(output_parts)
            duration = (time.time() - start_time) * 1000

            result = ExecutionResult(
                success=True,
                output=output,
                return_value=return_value,
                code=code,
                duration_ms=duration,
                layer=self._detect_layer(code),
            )
            self._history.append(result)
            return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000

            # IMPORTANT: Keep full stack trace for agent debugging
            error_msg = traceback.format_exc() if self.keep_errors else str(e)

            result = ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                code=code,
                duration_ms=duration,
                layer=self._detect_layer(code),
            )
            self._history.append(result)
            return result

    async def execute_shell(
        self,
        command: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a shell command (L2 layer).

        Args:
            command: Shell command to execute
            timeout: Optional timeout

        Returns:
            ExecutionResult
        """
        import time

        start_time = time.time()
        timeout = timeout or self.timeout

        try:
            if self.shell is not None:
                shell_result = await self.shell(commands=[command], timeout_ms=timeout * 1000)
                # ShellResult has .output which is a list of ShellCommandOutput
                output_parts = []
                for cmd_output in shell_result.output:
                    if cmd_output.stdout:
                        output_parts.append(cmd_output.stdout)
                    if cmd_output.stderr:
                        output_parts.append(f"STDERR: {cmd_output.stderr}")
                output = "\n".join(output_parts)
            else:
                # Fallback to subprocess
                import asyncio

                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode()
                if stderr:
                    output += f"\nSTDERR: {stderr.decode()}"
                if proc.returncode != 0:
                    output += f"\nExit code: {proc.returncode}"

            duration = (time.time() - start_time) * 1000
            result = ExecutionResult(
                success=True,
                output=output,
                code=command,
                duration_ms=duration,
                layer="L2",
            )
            self._history.append(result)
            return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            error_msg = traceback.format_exc() if self.keep_errors else str(e)

            result = ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                code=command,
                duration_ms=duration,
                layer="L2",
            )
            self._history.append(result)
            return result

    def _detect_layer(self, code: str) -> str:
        """Detect which layer the code primarily uses."""
        # L3: External packages
        l3_imports = ["requests", "pandas", "numpy", "PIL", "sklearn", "torch"]
        for pkg in l3_imports:
            if f"import {pkg}" in code or f"from {pkg}" in code:
                return "L3"

        # L2: CLI tools
        if "subprocess" in code or "shell(" in code:
            return "L2"

        # L1: Atomic operations
        return "L1"

    def get_history(self) -> list[ExecutionResult]:
        """Get execution history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()

    async def shutdown(self) -> None:
        """Shutdown the executor and release resources."""
        if self.jupyter is not None:
            await self.jupyter.shutdown()


class SandboxExecutor:
    """Simplified sandbox for code execution without Jupyter.

    Uses subprocess for isolation. Useful for testing or when Jupyter
    is not available.
    """

    def __init__(
        self,
        timeout: int = 60,
        workspace: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.workspace = workspace

    async def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute code in a subprocess sandbox."""
        import asyncio
        import tempfile
        import time

        start_time = time.time()
        timeout = timeout or self.timeout

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Execute in subprocess
            proc = await asyncio.create_subprocess_exec(
                "python",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            duration = (time.time() - start_time) * 1000
            output = stdout.decode()

            if proc.returncode != 0:
                error = stderr.decode() if stderr else f"Exit code: {proc.returncode}"
                return ExecutionResult(
                    success=False,
                    output=output,
                    error=error,
                    code=code,
                    duration_ms=duration,
                )

            return ExecutionResult(
                success=True,
                output=output,
                code=code,
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout}s",
                code=code,
                duration_ms=timeout * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=traceback.format_exc(),
                code=code,
            )
        finally:
            # Cleanup temp file
            import os

            if "temp_path" in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


__all__ = ["CodeActExecutor", "ExecutionResult", "SandboxExecutor"]

