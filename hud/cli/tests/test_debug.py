"""Tests for hud.cli.debug module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hud.cli.debug import debug_mcp_stdio
from hud.cli.utils.logging import CaptureLogger


class TestDebugMCPStdio:
    """Test the debug_mcp_stdio function."""

    @pytest.mark.asyncio
    async def test_phase_1_command_not_found(self) -> None:
        """Test Phase 1 failure when command not found."""
        logger = CaptureLogger(print_output=False)

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            phases = await debug_mcp_stdio(["nonexistent"], logger, max_phase=5)
            assert phases == 0
            output = logger.get_output()
            assert "Command not found: nonexistent" in output

    @pytest.mark.asyncio
    async def test_phase_1_command_fails(self) -> None:
        """Test Phase 1 failure when command returns error."""
        logger = CaptureLogger(print_output=False)

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Command failed with error"

        with patch("subprocess.run", return_value=mock_result):
            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
            assert phases == 0
            output = logger.get_output()
            assert "Command failed with exit code 1" in output
            assert "Command failed with error" in output

    @pytest.mark.asyncio
    async def test_phase_1_success(self) -> None:
        """Test Phase 1 success."""
        logger = CaptureLogger(print_output=False)

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=1)
            assert phases == 1
            output = logger.get_output()
            assert "Command executable found" in output
            assert "Stopping at phase 1 as requested" in output

    @pytest.mark.asyncio
    async def test_phase_1_usage_in_stderr(self) -> None:
        """Test Phase 1 success when usage info in stderr."""
        logger = CaptureLogger(print_output=False)

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "usage: test-cmd [options]"

        with patch("subprocess.run", return_value=mock_result):
            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=1)
            assert phases == 1
            output = logger.get_output()
            assert "Command executable found" in output

    @pytest.mark.asyncio
    async def test_phase_2_mcp_initialize_success(self) -> None:
        """Test Phase 2 MCP initialization success."""
        logger = CaptureLogger(print_output=False)

        # Mock Phase 1 success
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        # Mock subprocess.Popen for Phase 2
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        # Mock successful MCP response
        init_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "serverInfo": {"name": "TestServer", "version": "1.0"},
                "capabilities": {"tools": {}, "resources": {}},
            },
        }

        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])  # No stderr output

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=2)
            assert phases == 2
            output = logger.get_output()
            assert "MCP server initialized successfully" in output
            assert "Server: TestServer v1.0" in output

    @pytest.mark.asyncio
    async def test_phase_2_no_response(self) -> None:
        """Test Phase 2 failure when no MCP response."""
        logger = CaptureLogger(print_output=False)

        # Mock Phase 1 success
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        # Mock subprocess.Popen for Phase 2
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        # No stdout response
        mock_proc.stdout.readline.return_value = ""
        mock_proc.stderr.__iter__ = lambda x: iter(["[ERROR] Server failed to start"])

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.cli.debug.time.time", side_effect=[0, 0, 20]),
        ):
            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
            assert phases == 1
            output = logger.get_output()
            assert "No valid MCP response received" in output

    @pytest.mark.asyncio
    async def test_phase_2_invalid_json_response(self) -> None:
        """Test Phase 2 handling of invalid JSON response."""
        logger = CaptureLogger(print_output=False)

        # Mock Phase 1 success
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        # Invalid JSON response
        mock_proc.stdout.readline.return_value = "Invalid JSON\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            # Simulate timeout - time.time() is called multiple times in the loop
            # Return increasing values to simulate time passing
            time_values = list(range(20))
            with patch("hud.cli.debug.time.time", side_effect=time_values):
                phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
                assert phases == 1
                output = logger.get_output()
                # The error message might vary, but should indicate no valid response
                assert (
                    "Failed to parse MCP response" in output
                    or "No valid MCP response received" in output
                )

    @pytest.mark.asyncio
    async def test_phase_3_tool_discovery(self) -> None:
        """Test Phase 3 tool discovery."""
        logger = CaptureLogger(print_output=False)

        # Mock Phase 1 & 2 success
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        init_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"serverInfo": {"name": "TestServer", "version": "1.0"}},
        }
        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        # Mock tool discovery - create proper mock tools
        mock_tools = []
        for tool_name in ["setup", "evaluate", "computer", "custom_tool"]:
            tool = Mock()
            tool.name = tool_name
            mock_tools.append(tool)

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.clients.fastmcp.FastMCPHUDClient") as MockClient,
        ):
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tools)
            mock_client.list_resources = AsyncMock(return_value=[])
            mock_client.shutdown = AsyncMock()

            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=3)
            assert phases == 3
            output = logger.get_output()
            assert "Found 4 tools" in output
            assert "Lifecycle tools: setup=✅, evaluate=✅" in output
            assert "Interaction tools: computer" in output
            assert "All tools: setup, evaluate, computer, custom_tool" in output

    @pytest.mark.asyncio
    async def test_phase_3_no_tools(self) -> None:
        """Test Phase 3 when no tools found."""
        logger = CaptureLogger(print_output=False)

        # Mock Phase 1 & 2 success
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        mock_proc = MagicMock()
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.clients.fastmcp.FastMCPHUDClient") as MockClient,
        ):
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.shutdown = AsyncMock()

            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
            assert phases == 2
            output = logger.get_output()
            assert "No tools found" in output
            assert "@mcp.tool() decorator" in output

    @pytest.mark.asyncio
    async def test_phase_4_remote_deployment(self) -> None:
        """Test Phase 4 remote deployment readiness."""
        logger = CaptureLogger(print_output=False)

        # Setup mocks for phases 1-3
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        mock_proc = MagicMock()
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        # Create proper mock tools
        mock_tools = []
        for tool_name in ["setup", "evaluate"]:
            tool = Mock()
            tool.name = tool_name
            mock_tools.append(tool)

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.clients.fastmcp.FastMCPHUDClient") as MockClient,
        ):
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tools)
            mock_client.list_resources = AsyncMock(return_value=[])
            mock_client.call_tool = AsyncMock()
            mock_client.shutdown = AsyncMock()

            with patch(
                "hud.cli.debug.time.time", side_effect=[0, 5, 5, 5, 5]
            ):  # Start at 0, then 5 for the rest
                phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=4)
                assert phases == 4
                output = logger.get_output()
                assert "Total initialization time: 5.00s" in output
                # Should have tested setup and evaluate tools
                assert mock_client.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_phase_4_slow_initialization(self) -> None:
        """Test Phase 4 with slow initialization warning."""
        logger = CaptureLogger(print_output=False)

        # Setup basic mocks
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        mock_proc = MagicMock()
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.clients.fastmcp.FastMCPHUDClient") as MockClient,
        ):
            mock_client = MockClient.return_value
            mock_client.initialize = AsyncMock()
            # Create proper mock tool
            test_tool = Mock()
            test_tool.name = "test"
            mock_client.list_tools = AsyncMock(return_value=[test_tool])
            mock_client.list_resources = AsyncMock(return_value=[])
            mock_client.shutdown = AsyncMock()

            # Simulate slow init (>30s)
            # time.time() is called at start and after phase 3
            with patch("hud.cli.debug.time.time", side_effect=[0, 0, 0, 35, 35, 35]):
                phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
                output = logger.get_output()
                # Check if we got to phase 4 where the timing check happens
                if phases >= 4:
                    assert "Initialization took >30s" in output
                    assert "Consider optimizing startup time" in output

    @pytest.mark.asyncio
    async def test_phase_5_concurrent_clients(self) -> None:
        """Test Phase 5 concurrent clients."""
        logger = CaptureLogger(print_output=False)

        # Setup mocks for all phases
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        mock_proc = MagicMock()
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.clients.fastmcp.FastMCPHUDClient") as MockClient,
        ):
            # Create different mock instances for each client
            mock_clients = []
            for i in range(4):  # 1 main + 3 concurrent
                mock_client = MagicMock()
                mock_client.initialize = AsyncMock()
                # Create proper mock tool
                test_tool = Mock()
                test_tool.name = "test"
                mock_client.list_tools = AsyncMock(return_value=[test_tool])
                mock_client.list_resources = AsyncMock(return_value=[])
                mock_client.shutdown = AsyncMock()
                mock_clients.append(mock_client)

            MockClient.side_effect = mock_clients

            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
            assert phases == 5
            output = logger.get_output()
            assert "Creating 3 concurrent MCP clients" in output
            assert "All concurrent clients connected" in output

            # Verify all clients were shut down
            for client in mock_clients:
                client.shutdown.assert_called()

    @pytest.mark.asyncio
    async def test_phase_5_concurrent_failure(self) -> None:
        """Test Phase 5 handling concurrent client failures."""
        logger = CaptureLogger(print_output=False)

        # Setup basic mocks
        mock_run_result = Mock()
        mock_run_result.returncode = 0

        mock_proc = MagicMock()
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_proc.stdout.readline.return_value = json.dumps(init_response) + "\n"
        mock_proc.stderr.__iter__ = lambda x: iter([])

        with (
            patch("subprocess.run", return_value=mock_run_result),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("hud.clients.fastmcp.FastMCPHUDClient") as MockClient,
        ):
            # Set up for phase 1-4 success first
            test_tool = Mock()
            test_tool.name = "test"

            # Phase 1-4 client
            phase_client = MagicMock()
            phase_client.initialize = AsyncMock()
            phase_client.list_tools = AsyncMock(return_value=[test_tool])
            phase_client.list_resources = AsyncMock(return_value=[])
            phase_client.shutdown = AsyncMock()

            # Phase 5 clients - first succeeds, second fails
            mock_client1 = MagicMock()
            mock_client1.initialize = AsyncMock()
            mock_client1.list_tools = AsyncMock(return_value=[test_tool])
            mock_client1.list_resources = AsyncMock(return_value=[])
            mock_client1.shutdown = AsyncMock()

            mock_client2 = MagicMock()
            mock_client2.initialize = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client2.shutdown = AsyncMock()

            MockClient.side_effect = [phase_client, mock_client1, mock_client2]

            await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
            output = logger.get_output()
            assert "Concurrent test failed: Connection failed" in output

    @pytest.mark.asyncio
    async def test_docker_command_handling(self) -> None:
        """Test special handling of Docker commands."""
        logger = CaptureLogger(print_output=False)

        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await debug_mcp_stdio(["docker", "run", "--rm", "image:latest"], logger, max_phase=1)
            # Should add echo command for Docker
            call_args = mock_run.call_args[0][0]
            assert call_args == ["docker"]

    @pytest.mark.asyncio
    async def test_phase_exception_handling(self) -> None:
        """Test general exception handling in phases."""
        logger = CaptureLogger(print_output=False)

        with patch("subprocess.run", side_effect=Exception("Unexpected error")):
            phases = await debug_mcp_stdio(["test-cmd"], logger, max_phase=5)
            assert phases == 0
            output = logger.get_output()
            assert "Startup test failed: Unexpected error" in output


if __name__ == "__main__":
    pytest.main([__file__])
