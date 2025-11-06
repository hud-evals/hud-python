"""Jupyter execution tool."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import tornado
from tornado.escape import json_decode, json_encode, url_escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.ioloop import PeriodicCallback
from tornado.websocket import websocket_connect

from hud.tools.base import BaseTool
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from mcp.types import ContentBlock

logger = logging.getLogger(__name__)


def strip_ansi(output: str) -> str:
    """Remove ANSI escape sequences from string output."""
    pattern = re.compile(r"\x1B\[\d+(;\d+){0,2}m")
    return pattern.sub("", output)


class JupyterTool(BaseTool):
    """
    Execute Python code in a Jupyter kernel.
    """

    # Class-level kernel registry for sharing kernels
    _kernel_registry: ClassVar[dict[str, str]] = {}

    @classmethod
    def register_shared_kernel(cls, registry_name: str, kernel_id: str) -> None:
        """Register a kernel_id with a name for reuse.

        Args:
            registry_name: Name to register the kernel under
            kernel_id: The kernel ID to register
        """
        cls._kernel_registry[registry_name] = kernel_id
        logger.info("Registered kernel '%s': %s", registry_name, kernel_id)

    @classmethod
    def from_shared_kernel(cls, registry_name: str, **kwargs: Any) -> JupyterTool:
        """Connect to a kernel using its registry name.

        Args:
            registry_name: Name of the registered kernel
            **kwargs: Additional parameters for JupyterTool (url_suffix, kernel_name)

        Returns:
            JupyterTool instance connected to the registered kernel
        """
        kernel_id = cls._kernel_registry.get(registry_name)
        if not kernel_id:
            raise ValueError(f"No kernel registered with name '{registry_name}'")

        logger.info("Connecting to registered kernel '%s': %s", registry_name, kernel_id)
        return cls(kernel_id=kernel_id, **kwargs)

    def __init__(
        self,
        url_suffix: str = "localhost:8888",
        kernel_name: str = "python3",
        kernel_id: str = "",
    ) -> None:
        """Initialize JupyterTool with connection parameters.

        Args:
            url_suffix: (Optional) Kernel gateway host:port (default: localhost:8888)
            kernel_name: (Optional) Kernel name to use (default: python3)
            kernel_id: (Optional) If set, connect to the existed kernel with kernel_id.
                If empty, create new kernel
        """
        super().__init__(
            env=None,
            name="jupyter",
            title="Jupyter Code Execution",
            description="Execute Python code in a Jupyter kernel",
        )

        # Connection parameters
        self._base_url = f"http://{url_suffix}"
        self._base_ws_url = f"ws://{url_suffix}"
        self._kernel_name = kernel_name

        # Kernel state (reuse existing or create new)
        self._kernel_id = kernel_id
        self._ws = None
        self._initialized = False

        # WebSocket heartbeat
        self._heartbeat_interval = 10000  # 10 seconds
        self._heartbeat_callback = None

    async def __call__(self, code: str, execution_timeout: int = 15) -> list[ContentBlock]:
        """Execute Python code in the Jupyter kernel.

        Args:
            code: Python code to execute
            execution_timeout: Execution timeout in seconds (default: 15)

        Returns:
            List of ContentBlock with execution results
        """
        try:
            # Ensure kernel is ready (lazy initialization)
            await self._ensure_kernel()

            # Execute code
            result = await self._execute(code, execution_timeout)

            # Check for timeout
            if result.startswith("[Execution timed out"):
                return ContentResult(error=result).to_content_blocks()

            # Return result
            output = result if result.strip() else "Code executed successfully (no output)"
            return ContentResult(output=output).to_content_blocks()

        except Exception as e:
            logger.error("Jupyter execution error: %s", e)
            raise ToolError(f"Execution failed: {e!s}") from e

    async def _ensure_kernel(self) -> None:
        """Ensure kernel is initialized and connected."""
        if not self._initialized:
            logger.info("Initializing Jupyter kernel connection")
            await self._connect()
            self._initialized = True
            logger.info("Jupyter kernel connected successfully")

    async def _connect(self) -> None:
        """Connect to Jupyter kernel via WebSocket."""
        if self._ws:
            self._ws.close()
            self._ws = None

        client = AsyncHTTPClient()
        if not self._kernel_id:
            # Start a new kernel
            n_tries = 5
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        f"{self._base_url}/api/kernels",
                        method="POST",
                        body=json_encode({"name": self._kernel_name}),
                    )
                    kernel = json_decode(response.body)
                    self._kernel_id = kernel["id"]
                    logger.info("Kernel started with ID: %s", self._kernel_id)
                    break
                except Exception as e:
                    logger.warning("Kernel connection attempt failed: %s", e)
                    n_tries -= 1
                    await asyncio.sleep(1)

            if n_tries == 0:
                raise ConnectionRefusedError("Failed to connect to kernel gateway")

        # Connect WebSocket to kernel
        ws_req = HTTPRequest(
            url=f"{self._base_ws_url}/api/kernels/{url_escape(self._kernel_id)}/channels"
        )
        self._ws = await websocket_connect(ws_req)
        logger.info("WebSocket connected to kernel")

        # Setup heartbeat to keep connection alive
        if self._heartbeat_callback:
            self._heartbeat_callback.stop()
        self._heartbeat_callback = PeriodicCallback(self._send_heartbeat, self._heartbeat_interval)
        self._heartbeat_callback.start()

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to maintain WebSocket connection."""
        if not self._ws:
            return
        try:
            self._ws.ping()
        except tornado.iostream.StreamClosedError:
            try:
                await self._connect()
            except ConnectionRefusedError:
                logger.warning(
                    "Failed to reconnect to kernel websocket - Is the kernel still running?"
                )

    async def _execute(self, code: str, execution_timeout: int = 15) -> str:
        """Execute code in Jupyter kernel and return output.

        Args:
            code: Python code to execute
            execution_timeout: Execution timeout in seconds

        Returns:
            String output from the kernel
        """
        if not self._ws:
            await self._connect()

        msg_id = uuid4().hex
        self._ws.write_message(  # type: ignore
            json_encode(
                {
                    "header": {
                        "username": "",
                        "version": "5.0",
                        "session": "",
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                    },
                    "parent_header": {},
                    "channel": "shell",
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                    },
                    "metadata": {},
                    "buffers": {},
                }
            )
        )

        outputs = []

        async def wait_for_messages() -> bool:
            execution_done = False
            while not execution_done:
                msg = await self._ws.read_message()  # type: ignore
                msg = json_decode(msg)  # type: ignore
                msg_type = msg["msg_type"]
                parent_msg_id = msg["parent_header"].get("msg_id", None)

                if parent_msg_id != msg_id:
                    continue

                if msg_type == "error":
                    traceback = "\n\n\n\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                elif msg_type in ["execute_result", "display_data"]:
                    outputs.append(msg["content"]["data"]["text/plain"])
                    # Handle image outputs
                    if "image/png" in msg["content"]["data"]:
                        outputs.append(
                            f"![image](data:image/png;base64,{msg['content']['data']['image/png']})"
                        )
                elif msg_type == "execute_reply":
                    execution_done = True
            return execution_done

        async def interrupt_kernel() -> None:
            client = AsyncHTTPClient()
            interrupt_response = await client.fetch(
                f"{self._base_url}/api/kernels/{self._kernel_id}/interrupt",
                method="POST",
                body=json_encode({"kernel_id": self._kernel_id}),
            )
            logger.info("Kernel interrupted: %s", interrupt_response)

        try:
            await asyncio.wait_for(wait_for_messages(), execution_timeout)
        except TimeoutError:
            await interrupt_kernel()
            return f"[Execution timed out ({execution_timeout} seconds).]"

        ret = "".join(outputs)

        # Remove ANSI escape sequences
        return strip_ansi(ret)

    async def shutdown(self) -> None:
        """Shutdown the kernel connection."""
        if self._kernel_id:
            client = AsyncHTTPClient()
            try:
                await client.fetch(
                    f"{self._base_url}/api/kernels/{self._kernel_id}",
                    method="DELETE",
                )
                logger.info("Kernel %s shut down", self._kernel_id)
            except Exception as e:
                logger.warning("Error shutting down kernel: %s", e)

            self._kernel_id = ""

            if self._heartbeat_callback:
                self._heartbeat_callback.stop()
                self._heartbeat_callback = None

            if self._ws:
                self._ws.close()
                self._ws = None

        self._initialized = False

    def get_kernel_id(self) -> str:
        """Get the jupyter kernel id."""
        return self._kernel_id
