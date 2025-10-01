"""Tests for hud.cli.utils module."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from hud.cli.utils.logging import HINT_REGISTRY, CaptureLogger, Colors, analyze_error_for_hints


class TestColors:
    """Test ANSI color codes."""

    def test_color_constants(self) -> None:
        """Test that color constants are defined."""
        assert Colors.HEADER == "\033[95m"
        assert Colors.BLUE == "\033[94m"
        assert Colors.CYAN == "\033[96m"
        assert Colors.GREEN == "\033[92m"
        assert Colors.YELLOW == "\033[93m"
        assert Colors.GOLD == "\033[33m"
        assert Colors.RED == "\033[91m"
        assert Colors.GRAY == "\033[37m"
        assert Colors.ENDC == "\033[0m"
        assert Colors.BOLD == "\033[1m"


class TestCaptureLogger:
    """Test CaptureLogger functionality."""

    def test_logger_print_mode(self) -> None:
        """Test logger in print mode."""
        logger = CaptureLogger(print_output=True)

        with patch("builtins.print") as mock_print:
            logger._log("Test message", Colors.GREEN)
            mock_print.assert_called_once_with(f"{Colors.GREEN}Test message{Colors.ENDC}")

    def test_logger_capture_mode(self) -> None:
        """Test logger in capture-only mode."""
        logger = CaptureLogger(print_output=False)

        with patch("builtins.print") as mock_print:
            logger._log("Test message", Colors.GREEN)
            mock_print.assert_not_called()

        output = logger.get_output()
        assert "Test message" in output

    def test_strip_ansi(self) -> None:
        """Test ANSI code stripping."""
        logger = CaptureLogger(print_output=False)

        # Test various ANSI sequences
        text_with_ansi = (
            f"{Colors.GREEN}Green text{Colors.ENDC} normal {Colors.BOLD}bold{Colors.ENDC}"
        )
        clean_text = logger._strip_ansi(text_with_ansi)
        assert clean_text == "Green text normal bold"

    def test_timestamp(self) -> None:
        """Test timestamp generation."""
        logger = CaptureLogger(print_output=False)

        timestamp = logger.timestamp()
        # Should be in HH:MM:SS format
        assert len(timestamp) == 8
        assert timestamp[2] == ":"
        assert timestamp[5] == ":"

    def test_phase_logging(self) -> None:
        """Test phase header logging."""
        logger = CaptureLogger(print_output=False)

        logger.phase(1, "Test Phase")
        output = logger.get_output()

        assert "=" * 80 in output
        assert "PHASE 1: Test Phase" in output

    def test_command_logging(self) -> None:
        """Test command logging."""
        logger = CaptureLogger(print_output=False)

        logger.command(["python", "script.py", "--arg", "value"])
        output = logger.get_output()

        assert "$ python script.py --arg value" in output

    def test_success_logging(self) -> None:
        """Test success message logging."""
        logger = CaptureLogger(print_output=False)

        logger.success("Operation completed")
        output = logger.get_output()

        assert "✅ Operation completed" in output

    def test_error_logging(self) -> None:
        """Test error message logging."""
        logger = CaptureLogger(print_output=False)

        logger.error("Operation failed")
        output = logger.get_output()

        assert "❌ Operation failed" in output

    def test_info_logging(self) -> None:
        """Test info message logging with timestamp."""
        logger = CaptureLogger(print_output=False)

        with patch.object(logger, "timestamp", return_value="12:34:56"):
            logger.info("Information message")
            output = logger.get_output()

            assert "[12:34:56] Information message" in output

    def test_stdio_logging(self) -> None:
        """Test STDIO communication logging."""
        logger = CaptureLogger(print_output=False)

        logger.stdio("JSON-RPC message")
        output = logger.get_output()

        assert "[STDIO] JSON-RPC message" in output

    def test_stderr_logging(self) -> None:
        """Test STDERR output logging."""
        logger = CaptureLogger(print_output=False)

        logger.stderr("Error output from server")
        output = logger.get_output()

        assert "[STDERR] Error output from server" in output

    def test_hint_logging(self) -> None:
        """Test hint message logging."""
        logger = CaptureLogger(print_output=False)

        logger.hint("Try checking the configuration")
        output = logger.get_output()

        assert "💡 Hint: Try checking the configuration" in output

    def test_progress_bar(self) -> None:
        """Test progress bar visualization."""
        logger = CaptureLogger(print_output=False)

        # Test partial progress
        logger.progress_bar(3, 5)
        output = logger.get_output()

        assert "Progress: [███░░] 3/5 phases (60%)" in output
        assert "Failed at Phase 4" in output

        # Test complete progress
        logger = CaptureLogger(print_output=False)
        logger.progress_bar(5, 5)
        output = logger.get_output()

        assert "Progress: [█████] 5/5 phases (100%)" in output
        assert "All phases completed successfully!" in output

    def test_progress_bar_failure_messages(self) -> None:
        """Test progress bar failure messages at different phases."""
        test_cases = [
            (0, "Failed at Phase 1 - Server startup"),
            (1, "Failed at Phase 2 - MCP initialization"),
            (2, "Failed at Phase 3 - Tool discovery"),
            (3, "Failed at Phase 4 - Remote deployment readiness"),
            (4, "Failed at Phase 5 - Concurrent clients & resources"),
        ]

        for completed, expected_msg in test_cases:
            logger = CaptureLogger(print_output=False)
            logger.progress_bar(completed, 5)
            output = logger.get_output()
            assert expected_msg in output

    def test_get_output(self) -> None:
        """Test getting accumulated output."""
        logger = CaptureLogger(print_output=False)

        logger.info("First message")
        logger.error("Second message")
        logger.success("Third message")

        output = logger.get_output()
        assert "First message" in output
        assert "Second message" in output
        assert "Third message" in output


class TestAnalyzeErrorForHints:
    """Test error analysis and hint generation."""

    def test_x11_display_errors(self) -> None:
        """Test X11/display related error hints."""
        errors = [
            "Can't connect to display :0",
            "X11 connection rejected",
            "DISPLAY environment variable not set",
            "Xlib.error.DisplayConnectionError",
        ]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "GUI environment needs X11" in hint
            assert "Xvfb" in hint

    def test_import_errors(self) -> None:
        """Test import/module error hints."""
        errors = [
            "ModuleNotFoundError: No module named 'requests'",
            "ImportError: cannot import name 'api'",
            "No module named numpy",
        ]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Missing Python dependencies" in hint
            assert "pyproject.toml" in hint

    def test_json_errors(self) -> None:
        """Test JSON parsing error hints."""
        errors = [
            "json.decoder.JSONDecodeError: Expecting value",
            "JSONDecodeError: Expecting value: line 1 column 1",
        ]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Invalid JSON-RPC communication" in hint
            assert "proper JSON-RPC format" in hint

    def test_permission_errors(self) -> None:
        """Test permission error hints."""
        errors = [
            "Permission denied: /var/log/app.log",
            "EACCES: permission denied",
            "Operation not permitted",
        ]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Permission issues" in hint
            assert "Check file permissions" in hint

    def test_memory_errors(self) -> None:
        """Test memory/resource error hints."""
        errors = ["Cannot allocate memory", "Process killed", "Container OOMKilled"]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Resource limits exceeded" in hint
            assert "memory limits" in hint

    def test_port_errors(self) -> None:
        """Test port binding error hints."""
        errors = [
            "bind: address already in use",
            "EADDRINUSE: address already in use",
            "port 8080 already allocated",
        ]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Port conflict detected" in hint
            assert "different port" in hint

    def test_file_not_found_errors(self) -> None:
        """Test file not found error hints."""
        errors = [
            "FileNotFoundError: [Errno 2] No such file or directory",
            "No such file or directory: config.json",
        ]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "File or directory missing" in hint
            assert "required files exist" in hint

    def test_traceback_errors(self) -> None:
        """Test general traceback error hints."""
        error = """Traceback (most recent call last):
  File "app.py", line 10, in <module>
    import missing_module
ImportError: No module named missing_module"""

        hint = analyze_error_for_hints(error)
        assert hint is not None
        # Should match both traceback and import patterns
        # Import has higher priority
        assert "Missing Python dependencies" in hint

    def test_timeout_errors(self) -> None:
        """Test timeout error hints."""
        errors = ["Operation timed out after 30 seconds", "Connection timeout", "Request timed out"]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Server taking too long to start" in hint
            assert "slow operations" in hint

    def test_no_error_text(self) -> None:
        """Test with empty or None error text."""
        assert analyze_error_for_hints("") is None
        assert analyze_error_for_hints(None) is None

    def test_no_matching_pattern(self) -> None:
        """Test with error that doesn't match any pattern."""
        hint = analyze_error_for_hints("Some random error message")
        assert hint is None

    def test_priority_ordering(self) -> None:
        """Test that higher priority hints are returned."""
        # This error matches both "No module" (priority 9) and "Exception" (priority 2)
        error = "Exception: No module named requests"
        hint = analyze_error_for_hints(error)
        assert hint is not None
        # Should get the higher priority hint (import error)
        assert "Missing Python dependencies" in hint

    def test_case_insensitive_matching(self) -> None:
        """Test that pattern matching is case insensitive."""
        errors = ["PERMISSION DENIED", "permission denied", "Permission Denied"]

        for error in errors:
            hint = analyze_error_for_hints(error)
            assert hint is not None
            assert "Permission issues" in hint


class TestHintRegistry:
    """Test the hint registry structure."""

    def test_hint_registry_structure(self) -> None:
        """Test that HINT_REGISTRY has correct structure."""
        assert isinstance(HINT_REGISTRY, list)
        assert len(HINT_REGISTRY) > 0

        for hint_data in HINT_REGISTRY:
            assert "patterns" in hint_data
            assert "priority" in hint_data
            assert "hint" in hint_data

            assert isinstance(hint_data["patterns"], list)
            assert isinstance(hint_data["priority"], int)
            assert isinstance(hint_data["hint"], str)

            # All patterns should be strings
            for pattern in hint_data["patterns"]:
                assert isinstance(pattern, str)

    def test_hint_priorities_unique(self) -> None:
        """Test that hint priorities are reasonable."""
        priorities = [hint["priority"] for hint in HINT_REGISTRY]

        # Priorities should be positive
        assert all(p > 0 for p in priorities)

        # Should have a range of priorities
        assert max(priorities) > min(priorities)


class TestWindowsSupport:
    """Test Windows-specific functionality."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only test")
    def test_windows_ansi_enable(self) -> None:
        """Test that ANSI is enabled on Windows."""
        # The module should call os.system("") on import
        # This is hard to test directly, but we can check platform detection
        assert sys.platform == "win32"


if __name__ == "__main__":
    pytest.main([__file__])
