"""Test JupyterTool"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from hud.tools.jupyter import JupyterTool, strip_ansi


class TestStripAnsi:
    """Test strip_ansi utility function."""

    def test_strip_ansi(self):
        """Test stripping ANSI color codes."""
        input_text = "\x1b[31mRed text\x1b[0m"
        assert strip_ansi(input_text) == "Red text"


class TestJupyterTool:
    """Test class for JupyterTool"""

    def test_jupyter_tool_init(self):
        """Test JupyterTool initialization with defaults."""
        tool = JupyterTool()
        assert tool.name == "jupyter"
        assert tool.title == "Jupyter Code Execution"
        assert tool.description == "Execute Python code in a Jupyter kernel"
        assert tool._base_url == "http://localhost:8888"
        assert tool._base_ws_url == "ws://localhost:8888"
        assert tool._kernel_name == "python3"
        assert tool._kernel_id == ""
        assert tool._ws is None
        assert tool._initialized is False

    def test_shared_kernel(self):
        """Test reregister_shared_kernel and from_shared_kernel."""
        # Succeed on `reregister_shared_kernel` and `from_shared_kernel`
        JupyterTool._kernel_registry.clear()
        JupyterTool.register_shared_kernel("shared_kernel", "kernel-456")
        tool = JupyterTool.from_shared_kernel("shared_kernel", url_suffix="localhost:8888")

        assert tool._kernel_id == "kernel-456"
        assert tool._base_url == "http://localhost:8888"

        # Failure on `from_shared_kernel`
        JupyterTool._kernel_registry.clear()
        with pytest.raises(ValueError) as exc_info:
            JupyterTool.from_shared_kernel("nonexistent_kernel")

        assert "No kernel registered with name 'nonexistent_kernel'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call(self):
        """Test public API integration with successful execution."""
        tool = JupyterTool()

        with (
            patch.object(tool, "_ensure_kernel", new_callable=AsyncMock),
            patch.object(tool, "_execute", new_callable=AsyncMock) as mock_execute,
        ):
            mock_execute.return_value = "Hello, World!"
            result = await tool(code="print('Hello, World!')")
            assert isinstance(result[0], TextContent)
            assert result[0].text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_ensure_kernel(self):
        """Test kernel initialization on first call."""
        tool = JupyterTool()
        with patch.object(tool, "_connect", new_callable=AsyncMock):
            await tool._ensure_kernel()
            assert tool._initialized is True

    @pytest.mark.asyncio
    async def test_connect_new_kernel(self):
        """Test connecting and starting a new kernel."""
        tool = JupyterTool()
        mock_response = MagicMock(body=b'{"id": "new-kernel-123"}')
        mock_client = MagicMock(fetch=AsyncMock(return_value=mock_response))

        with (
            patch("hud.tools.jupyter.AsyncHTTPClient", return_value=mock_client),
            patch("hud.tools.jupyter.websocket_connect", new_callable=AsyncMock),
            patch("hud.tools.jupyter.PeriodicCallback"),
        ):
            await tool._connect()
            assert tool._kernel_id == "new-kernel-123"

    @pytest.mark.asyncio
    async def test_connect_existing_kernel(self):
        """Test connecting to an existing kernel."""
        tool = JupyterTool(kernel_id="existing-kernel-456")
        with (
            patch("hud.tools.jupyter.AsyncHTTPClient"),
            patch("hud.tools.jupyter.websocket_connect", new_callable=AsyncMock),
            patch("hud.tools.jupyter.PeriodicCallback"),
        ):
            await tool._connect()
            assert tool._kernel_id == "existing-kernel-456"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful code execution via Jupyter protocol."""
        tool = JupyterTool(kernel_id="test-kernel")
        stream_msg = (
            '{"msg_type": "stream", "parent_header": {"msg_id": "test-msg"}, '
            '"content": {"text": "Output"}}'
        )
        reply_msg = (
            '{"msg_type": "execute_reply", "parent_header": {"msg_id": "test-msg"}, "content": {}}'
        )
        tool._ws = MagicMock(read_message=AsyncMock(side_effect=[stream_msg, reply_msg]))

        with patch("hud.tools.jupyter.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "test-msg"
            result = await tool._execute("print('Output')")
            assert result == "Output"

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test code execution with error via Jupyter protocol."""
        tool = JupyterTool(kernel_id="test-kernel")
        error_msg = (
            '{"msg_type": "error", "parent_header": {"msg_id": "test-msg"}, '
            '"content": {"traceback": ["Traceback", "Error"]}}'
        )
        tool._ws = MagicMock(read_message=AsyncMock(side_effect=[error_msg]))

        with patch("hud.tools.jupyter.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "test-msg"
            result = await tool._execute("1/0")
            assert "Traceback" in result and "Error" in result

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test code execution timeout with kernel interrupt."""
        import asyncio

        tool = JupyterTool(kernel_id="test-kernel")

        # Mock websocket to hang indefinitely
        async def hang_forever():
            await asyncio.sleep(9999)

        tool._ws = MagicMock(read_message=hang_forever)
        mock_client = MagicMock(fetch=AsyncMock())

        with (
            patch("hud.tools.jupyter.uuid4") as mock_uuid,
            patch("hud.tools.jupyter.AsyncHTTPClient", return_value=mock_client),
        ):
            mock_uuid.return_value.hex = "test-msg"
            result = await tool._execute("while True: pass", execution_timeout=1)
            assert "[Execution timed out" in result

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutdown cleans up kernel state."""
        tool = JupyterTool(kernel_id="shutdown-kernel")
        tool._initialized = True
        tool._ws = MagicMock()
        tool._heartbeat_callback = MagicMock()

        with patch("hud.tools.jupyter.AsyncHTTPClient"):
            await tool.shutdown()
            assert tool._kernel_id == ""
            assert tool._ws is None
            assert not tool._initialized

    def test_get_kernel_id(self):
        """Test getting kernel ID."""
        tool = JupyterTool(kernel_id="test-kernel-789")
        assert tool.get_kernel_id() == "test-kernel-789"
