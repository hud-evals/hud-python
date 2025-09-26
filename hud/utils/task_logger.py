"""Per-task file logging utilities."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from hud.settings import settings

try:
    from hud.types import MCPToolCall, MCPToolResult, Task, Trace
except ImportError:  # pragma: no cover - typing only during runtime issues
    MCPToolCall = MCPToolResult = Task = Trace = Any  # type: ignore


def _json_default(value: Any) -> Any:
    """Best-effort serializer for complex objects."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except TypeError:
            return value.model_dump(mode="json")  # type: ignore[arg-type]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _sanitize_filename_component(value: str | None) -> str:
    """Create a filesystem-safe component from a string."""
    if not value:
        return "task"
    normalized = re.sub(r"\s+", "_", value.strip().lower())
    sanitized = re.sub(r"[^a-z0-9_.-]", "", normalized)
    sanitized = sanitized.strip("_.-")
    if not sanitized:
        return "task"
    return sanitized[:64]


class TaskLogger:
    """Manages file-based logging for individual tasks."""

    def __init__(
        self,
        task_run_id: str,
        task_id: str | None = None,
        task_name: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        self.task_run_id = task_run_id
        self.task_id = task_id
        self.task_name = task_name
        self.log_dir = Path(log_dir or settings.task_log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        filename_parts = [self.task_run_id]
        if task_id:
            filename_parts.insert(0, _sanitize_filename_component(task_id))
        elif task_name:
            filename_parts.insert(0, _sanitize_filename_component(task_name))
        filename = "-".join(filter(None, filename_parts)) + ".log"
        self.log_path = self.log_dir / filename

        self.log_level = getattr(logging, settings.task_log_level.upper(), logging.DEBUG)
        self.logger_name = f"hud.task.{self.task_run_id}"
        self.logger = self.setup_file_logger()

    def setup_file_logger(self) -> logging.Logger:
        """Create a dedicated logger instance for this task."""
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)
        logger.propagate = False

        # Avoid duplicate handlers if logger reused
        if not any(
            isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(self.log_path)  # noqa: SIM103
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)
            self._file_handler = file_handler
        else:
            self._file_handler = next(
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
                and getattr(handler, "baseFilename", None) == str(self.log_path)
            )

        return logger

    # ------------------------------------------------------------------
    # Public logging helpers
    # ------------------------------------------------------------------
    def log_task_start(self, task: Task) -> None:
        """Log task metadata, prompt, and configuration."""
        content = task.model_dump(mode="python") if hasattr(task, "model_dump") else dict(task)
        payload = {
            "task_run_id": self.task_run_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task": content,
        }
        self._log_structured("Task start", payload, level=logging.DEBUG)

    def log_agent_step(self, *, step: int, action: str, details: dict[str, Any]) -> None:
        """Log each agent step with full context."""
        payload = {
            "step": step,
            "action": action,
            "details": details,
        }
        self._log_structured("Agent step", payload, level=logging.DEBUG)

    def log_tool_call(self, tool_call: MCPToolCall) -> None:
        """Log tool invocation details."""
        content = tool_call.model_dump(mode="python") if hasattr(tool_call, "model_dump") else {
            "name": getattr(tool_call, "name", None),
            "arguments": getattr(tool_call, "arguments", None),
        }
        self._log_structured("Tool call", content, level=logging.INFO)

    def log_tool_result(self, result: MCPToolResult) -> None:
        """Log tool result details including errors."""
        content = result.model_dump(mode="python") if hasattr(result, "model_dump") else {
            "isError": getattr(result, "isError", None),
            "content": getattr(result, "content", None),
        }
        self._log_structured("Tool result", content, level=logging.INFO)

    def log_mcp_communication(self, direction: str, content: dict[str, Any]) -> None:
        """Log MCP protocol messages."""
        self._log_structured(f"MCP {direction}", content, level=logging.DEBUG)

    def log_environment_state(self, state: dict[str, Any]) -> None:
        """Log environment observations/screenshots."""
        self._log_structured("Environment state", state, level=logging.INFO)

    def log_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Log timing, memory usage, etc."""
        self._log_structured("Performance metrics", metrics, level=logging.DEBUG)

    def log_trace_completion(self, result: Trace) -> None:
        """Log final task results and evaluation."""
        content = result.model_dump(mode="python") if hasattr(result, "model_dump") else dict(result)
        self._log_structured("Trace completion", content, level=logging.INFO)

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Close file handlers and cleanup resources."""
        handler = getattr(self, "_file_handler", None)
        if handler and handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.flush()
            handler.close()
        self._file_handler = None

    def log_console_message(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        logger_name: str | None = None,
    ) -> None:
        """Write a console-formatted message to the log file."""
        handler = getattr(self, "_file_handler", None)
        if handler is None:
            return

        lines = str(message).rstrip("\n").splitlines() or [""]
        timestamp = datetime.now().strftime("%H:%M:%S")
        channel = logger_name or self.logger_name
        formatted_lines = [f"{timestamp} - {channel} - {lines[0]}"]
        if len(lines) > 1:
            formatted_lines.extend(lines[1:])

        handler.acquire()
        try:
            for line in formatted_lines:
                handler.stream.write(line + handler.terminator)
            handler.flush()
        finally:
            handler.release()

    # ------------------------------------------------------------------
    def _log_structured(
        self,
        label: str,
        payload: Any,
        *,
        level: int = logging.INFO,
        logger_name: str | None = None,
    ) -> None:
        """Helper to log structured payloads using verbose-friendly format."""
        channel = logger_name or "hud.task"
        message = f"{label}:\n{self._format_payload(payload)}"
        self.log_console_message(message, level=level, logger_name=channel)

    @staticmethod
    def _format_payload(data: Any) -> str:
        """Pretty-print structured data for logs."""
        try:
            return json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
        except TypeError:
            return str(data)


class TaskLogHandler(logging.Handler):
    """Logging handler that routes standard logging records into a task log."""

    def __init__(self, task_logger: TaskLogger) -> None:
        super().__init__()
        self.task_logger = task_logger
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_hud_console_logged", False):
            return

        file_handler = getattr(self.task_logger, "_file_handler", None)
        if file_handler is None:
            return

        try:
            msg = self.format(record)
            file_handler.acquire()
            try:
                file_handler.stream.write(msg + file_handler.terminator)
                file_handler.flush()
            finally:
                file_handler.release()
        except Exception:
            self.handleError(record)


__all__ = ["TaskLogger", "TaskLogHandler"]
