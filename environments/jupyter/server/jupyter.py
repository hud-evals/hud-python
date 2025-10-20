"""
SpreadsheetBench native kernel execution - From GitHub RUCKBReasoning/SpreadsheetBench
SpreadsheetBench Execution Environment.
"""

import os
import re
import time
import json
import asyncio
import tornado
import logging
from tornado.escape import json_encode, json_decode, url_escape
from tornado.websocket import websocket_connect, WebSocketHandler
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from uuid import uuid4

logger = logging.getLogger(__name__)


def strip_ansi(o: str) -> str:
    """
    Removes ANSI escape sequences from string output.
    Adapted from SpreadsheetBench implementation.
    """
    pattern = re.compile(r"\x1B\[\d+(;\d+){0,2}m")
    stripped = pattern.sub("", o)
    return stripped


class JupyterKernel:
    """
    SpreadsheetBench-compatible Jupyter kernel for code execution.
    Adapted from SpreadsheetBench's JupyterKernel class.
    """

    def __init__(self, url_suffix, convid, lang="python"):
        self.base_url = f"http://{url_suffix}"
        self.base_ws_url = f"ws://{url_suffix}"
        self.lang = lang
        self.kernel_id : str | None = None
        self.ws = None
        self.convid = convid
        logger.info(f"SpreadsheetBench kernel created for conversation {convid} at {url_suffix}")

        self.heartbeat_interval = 10000  # 10 seconds
        self.heartbeat_callback = None

    async def initialize(self):
        """Initialize kernel with SpreadsheetBench settings."""
        await self.execute(r"%colors nocolor")

        # SpreadsheetBench-compatible initialization
        self.tools_to_run = [
            # Pre-defined tools for SpreadsheetBench compatibility
            "import pandas as pd",
            "import numpy as np",
            "import openpyxl",
            "import os",
            "from pathlib import Path",
            "import warnings",
            "warnings.filterwarnings('ignore')",
            # Set up data directories
            "os.makedirs('/app/data', exist_ok=True)",
            "os.makedirs('/app/shared_data', exist_ok=True)",
            "os.makedirs('/app/workspace', exist_ok=True)",
        ]

        for tool in self.tools_to_run:
            await self.execute(tool)

        logger.info(f"Jupyter kernel initialized for {self.convid}")

    async def _send_heartbeat(self):
        """Send heartbeat to maintain WebSocket connection."""
        if not self.ws:
            return
        try:
            self.ws.ping()
        except tornado.iostream.StreamClosedError:
            try:
                await self._connect()
            except ConnectionRefusedError:
                logger.warning(
                    "Failed to reconnect to kernel websocket - Is the kernel still running?"
                )

    async def _connect(self):
        """Connect to Jupyter kernel via WebSocket."""
        if self.ws:
            self.ws.close()
            self.ws = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            n_tries = 5
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        f"{self.base_url}/api/kernels",
                        method="POST",
                        body=json_encode({"name": self.lang}),
                    )
                    kernel = json_decode(response.body)
                    self.kernel_id = kernel["id"]
                    break
                except Exception as e:
                    logger.warning(f"Kernel connection attempt failed: {e}")
                    n_tries -= 1
                    await asyncio.sleep(1)

            if n_tries == 0:
                raise ConnectionRefusedError("Failed to connect to kernel")

        ws_req = HTTPRequest(
            url=f"{self.base_ws_url}/api/kernels/{url_escape(self.kernel_id)}/channels"
        )
        self.ws = await websocket_connect(ws_req)
        logger.info("Connected to SpreadsheetBench kernel websocket")

        # Setup heartbeat
        if self.heartbeat_callback:
            self.heartbeat_callback.stop()
        self.heartbeat_callback = PeriodicCallback(self._send_heartbeat, self.heartbeat_interval)
        self.heartbeat_callback.start()

    async def execute(self, code, timeout=60):
        """
        Execute code in SpreadsheetBench kernel.
        Returns string output exactly as SpreadsheetBench would.
        """
        if not self.ws:
            await self._connect()

        msg_id = uuid4().hex
        self.ws.write_message(
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

        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                msg = await self.ws.read_message()
                msg = json_decode(msg)
                msg_type = msg["msg_type"]
                parent_msg_id = msg["parent_header"].get("msg_id", None)

                if parent_msg_id != msg_id:
                    continue

                if os.environ.get("DEBUG", False):
                    logger.debug(
                        f"MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {msg['content']}"
                    )

                if msg_type == "error":
                    traceback = "\n\n\n\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                elif msg_type in ["execute_result", "display_data"]:
                    outputs.append(msg["content"]["data"]["text/plain"])
                    if "image/png" in msg["content"]["data"]:
                        # SpreadsheetBench image handling
                        outputs.append(
                            f"![image](data:image/png;base64,{msg['content']['data']['image/png']})"
                        )
                elif msg_type == "execute_reply":
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            client = AsyncHTTPClient()
            interrupt_response = await client.fetch(
                f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt",
                method="POST",
                body=json_encode({"kernel_id": self.kernel_id}),
            )
            logger.info(f"Kernel interrupted: {interrupt_response}")

        try:
            execution_done = await asyncio.wait_for(wait_for_messages(), timeout)
        except asyncio.TimeoutError:
            await interrupt_kernel()
            return f"[Execution timed out ({timeout} seconds).]"

        if not outputs and execution_done:
            ret = "[Code executed successfully with no output]"
        else:
            ret = "".join(outputs)

        # Remove ANSI escape sequences
        ret = strip_ansi(ret)

        if os.environ.get("DEBUG", False):
            logger.debug(f"OUTPUT:\n{ret}")

        return ret

    async def shutdown_async(self):
        """Shutdown the kernel connection."""
        if self.kernel_id:
            client = AsyncHTTPClient()
            try:
                await client.fetch(
                    f"{self.base_url}/api/kernels/{self.kernel_id}",
                    method="DELETE",
                )
                logger.info(f"Kernel {self.kernel_id} shut down")
            except Exception as e:
                logger.warning(f"Error shutting down kernel: {e}")

            self.kernel_id = None

            if self.heartbeat_callback:
                self.heartbeat_callback.stop()
                self.heartbeat_callback = None

            if self.ws:
                self.ws.close()
                self.ws = None


class JupyterKernelWrapper:
    """
    Simplified kernel wrapper for our integrated environment.
    Returns localhost connection since kernel gateway runs in the same container.
    """

    def __init__(self, name: str):
        self.name = name
        self.url_suffix = "localhost:8888"  # Our kernel gateway runs on port 8888

    def __enter__(self):
        """Return URL suffix for kernel connection."""
        return self.url_suffix

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup - nothing needed for integrated setup."""
        pass
