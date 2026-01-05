"""Step logging with full traces and error preservation.

This module implements comprehensive logging following the following principles:
- Keep error messages: Stack traces are preserved for debugging
- Log every step: Input, output, tool calls, results
- Structured logs: YAML format for human readability
- Per-agent logs: Each sub-agent writes its own log file
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from hud.multi_agent.schemas import StepLog

logger = logging.getLogger(__name__)


# Pattern to match base64 image data in strings (long alphanumeric sequences)
_BASE64_PATTERN = re.compile(r"'data':\s*'[A-Za-z0-9+/=]{100,}'")


def _sanitize_response_string(response: str | None) -> str | None:
    """Remove base64 data from response strings for logging.

    Args:
        response: Response string that may contain base64 data

    Returns:
        Sanitized string safe for logging
    """
    if not response:
        return response

    if not isinstance(response, str):
        response = str(response)

    # Replace base64 data patterns with placeholder
    sanitized = _BASE64_PATTERN.sub("'data': '[BASE64_DATA_REMOVED]'", response)

    # Also truncate if still too long (>10KB)
    if len(sanitized) > 10000:
        sanitized = sanitized[:5000] + f"... [TRUNCATED {len(response)} chars total]"

    return sanitized


def _sanitize_tool_results(results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Remove base64 images and large data from tool results for logging.

    Args:
        results: List of tool result dicts

    Returns:
        Sanitized list safe for logging
    """
    if not results:
        return []

    sanitized = []
    for result in results:
        sanitized_result = {}
        for key, value in result.items():
            if key == "content":
                # Handle content that may contain images
                if isinstance(value, list):
                    # Filter out image blocks, keep text
                    sanitized_content = []
                    for item in value:
                        if isinstance(item, dict):
                            if item.get("type") == "image":
                                sanitized_content.append(
                                    {"type": "image", "source": "[BASE64_IMAGE_REMOVED]"}
                                )
                            elif item.get("type") == "base64":
                                sanitized_content.append(
                                    {"type": "base64", "data": "[BASE64_DATA_REMOVED]"}
                                )
                            else:
                                sanitized_content.append(item)
                        else:
                            sanitized_content.append(item)
                    sanitized_result[key] = sanitized_content
                elif isinstance(value, str) and len(value) > 5000:
                    # Truncate very long string content
                    sanitized_result[key] = value[:1000] + f"... [TRUNCATED {len(value)} chars]"
                else:
                    sanitized_result[key] = value
            else:
                sanitized_result[key] = value
        sanitized.append(sanitized_result)

    return sanitized


class SubAgentLogger:
    """Logger for a single sub-agent execution instance.

    Each invocation of a sub-agent gets its own YAML log file.
    If coder is called twice, you get: coder_1.yaml, coder_2.yaml

    Example:
        logger = SubAgentLogger(run_dir=Path(".logs/abc123"), agent_name="coder", invocation=1)

        logger.log_execution_step(
            tool="bash",
            arguments={"command": "python chart.py"},
            result="Chart saved",
            duration_ms=1200
        )

        logger.finalize(output="Created chart", artifacts=["./chart.png"])
        # Writes to .logs/abc123/coder_1.yaml
    """

    def __init__(
        self,
        run_dir: Path,
        agent_name: str,
        prompt: str = "",
        invocation: int = 1,
    ) -> None:
        """Initialize sub-agent logger.

        Args:
            run_dir: Directory for this run's logs
            agent_name: Name of the sub-agent (e.g., "coder", "researcher")
            prompt: The task prompt given to this sub-agent
            invocation: Which invocation of this agent (1st, 2nd, etc.)
        """
        self.run_dir = run_dir
        self.agent_name = agent_name
        self.invocation = invocation
        # Each invocation gets its own file: coder_1.yaml, coder_2.yaml, etc.
        self.log_file = run_dir / f"{agent_name}_{invocation}.yaml"

        self._invoked_at = datetime.now()
        self._prompt = prompt
        self._execution_steps: list[dict[str, Any]] = []
        self._step_count = 0

    def log_execution_step(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: str,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Log a single tool execution step.

        Args:
            tool: Name of the tool called
            arguments: Tool arguments
            result: Tool result (will be sanitized)
            duration_ms: Execution time in milliseconds
            error: Error message if tool failed
        """
        self._step_count += 1

        # Sanitize result
        if len(result) > 2000:
            result = result[:1000] + f"... [TRUNCATED {len(result)} chars]"

        step = {
            "step": self._step_count,
            "tool": tool,
            "arguments": arguments,
            "result": result,
        }

        if duration_ms is not None:
            step["duration_ms"] = round(duration_ms, 2)

        if error:
            step["error"] = error

        self._execution_steps.append(step)

    def finalize(
        self,
        output: str,
        success: bool = True,
        error: str | None = None,
        artifacts: list[str] | None = None,
    ) -> str:
        """Finalize and write the sub-agent log to YAML file.

        Args:
            output: Final output summary from the sub-agent
            success: Whether execution succeeded
            error: Error message if failed
            artifacts: List of file paths created/modified

        Returns:
            Path to the log file (relative to workspace)
        """
        end_time = datetime.now()
        duration_ms = (end_time - self._invoked_at).total_seconds() * 1000

        log_data = {
            "agent": self.agent_name,
            "invocation": self.invocation,
            "invoked_at": self._invoked_at.isoformat(),
            "prompt": self._prompt,
            "execution": self._execution_steps,
            "output": output,
            "success": success,
            "duration_ms": round(duration_ms, 2),
        }

        if artifacts:
            log_data["artifacts"] = artifacts

        if error:
            log_data["error"] = error

        # Write YAML file (each invocation gets its own file)
        with open(self.log_file, "w") as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return str(self.log_file)


class StepLogger:
    """Log every agent step with full context.

    Logs are saved in YAML format for human readability.
    Each sub-agent gets its own log file.

    Example:
        logger = StepLogger(log_dir=Path("./.logs"), run_id="abc123")

        # Create sub-agent logger
        sub_logger = logger.create_subagent_logger("coder", prompt="Create chart")

        # Log steps
        sub_logger.log_execution_step(...)
        sub_logger.finalize(...)

        # Main log saved to .logs/abc123/main.yaml
        # Sub-agent log saved to .logs/abc123/coder.yaml
    """

    def __init__(
        self,
        log_dir: Path | str = "./.logs",
        run_id: str | None = None,
        task: str = "",
        main_agent: str = "orchestrator",
    ) -> None:
        """Initialize the logger.

        Args:
            log_dir: Directory for log files
            run_id: Unique run identifier (auto-generated if not provided)
            task: The main task description
            main_agent: Name of the main orchestrator agent
        """
        self.log_dir = Path(log_dir)
        self.run_id = run_id or uuid4().hex[:12]
        self.task = task
        self.main_agent = main_agent

        # Create run-specific directory
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths (now YAML)
        self.main_log_file = self.run_dir / "main.yaml"
        self.errors_file = self.run_dir / "errors.log"
        self.summary_file = self.run_dir / "summary.yaml"

        # Tracking
        self._step_count = 0
        self._error_count = 0
        self._start_time = datetime.now()
        self._steps: list[dict[str, Any]] = []
        # Track all sub-agent loggers (list since same agent can be invoked multiple times)
        self._subagent_loggers: list[SubAgentLogger] = []
        # Track invocation count per agent name for numbering log files
        self._agent_invocation_counts: dict[str, int] = {}

    def create_subagent_logger(self, agent_name: str, prompt: str = "") -> SubAgentLogger:
        """Create a logger for a sub-agent invocation.

        Each call creates a new logger with incrementing invocation number.
        E.g., first call to coder -> coder_1.yaml, second -> coder_2.yaml

        Args:
            agent_name: Name of the sub-agent (e.g., "coder", "researcher")
            prompt: The task prompt given to this sub-agent

        Returns:
            SubAgentLogger instance for the sub-agent to use
        """
        # Increment invocation count for this agent
        if agent_name not in self._agent_invocation_counts:
            self._agent_invocation_counts[agent_name] = 0
        self._agent_invocation_counts[agent_name] += 1
        invocation = self._agent_invocation_counts[agent_name]

        sub_logger = SubAgentLogger(
            run_dir=self.run_dir,
            agent_name=agent_name,
            prompt=prompt,
            invocation=invocation,
        )
        self._subagent_loggers.append(sub_logger)
        return sub_logger

    def _write_main_log(self) -> None:
        """Write the main orchestrator log to YAML file."""
        log_data = {
            "run_id": self.run_id,
            "started_at": self._start_time.isoformat(),
            "task": self.task,
            "main_agent": self.main_agent,
            "steps": self._steps,
        }

        with open(self.main_log_file, "w") as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    async def log_step(
        self,
        agent_id: str,
        input_prompt: str,
        input_context_size: int = 0,
        output_response: str | None = None,
        output_tool_calls: list[dict[str, Any]] | None = None,
        output_tool_results: list[dict[str, Any]] | None = None,
        model: str = "unknown",
        parent_step_id: str | None = None,
        error: str | None = None,
        context_tokens_before: int | None = None,
        context_tokens_after: int | None = None,
        compactions_performed: int = 0,
        token_usage: dict[str, int] | None = None,
    ) -> str:
        """Log a single agent step.

        Args:
            agent_id: ID of the agent taking this step
            input_prompt: The prompt/input for this step
            input_context_size: Token count of input context
            output_response: Agent's text response
            output_tool_calls: Tool calls made
            output_tool_results: Results from tool calls
            model: Model name used
            parent_step_id: Parent step if this is a sub-agent step
            error: Error message if step failed
            context_tokens_before: Context tokens before step
            context_tokens_after: Context tokens after step
            compactions_performed: Number of context compactions
            token_usage: Token usage stats from model

        Returns:
            Step ID for reference
        """
        self._step_count += 1
        step_id = f"{self.run_id}_{self._step_count:04d}"
        timestamp = datetime.now()

        # Create step entry for main log (minimal)
        step_entry: dict[str, Any] = {
            "step_id": self._step_count,
            "agent": agent_id,
            "timestamp": timestamp.isoformat(),
        }

        if input_prompt:
            # Truncate long prompts for main log
            step_entry["prompt"] = (
                input_prompt[:200] + "..." if len(input_prompt) > 200 else input_prompt
            )

        if output_response:
            step_entry["response"] = _sanitize_response_string(output_response)

        if output_tool_calls:
            # Only log tool names in main log, not full details
            step_entry["tools_called"] = [tc.get("name", "unknown") for tc in output_tool_calls]

        if error:
            step_entry["error"] = error
            self._error_count += 1
            await self.log_error(agent_id, error, step_id)

        if model != "unknown":
            step_entry["model"] = model

        self._steps.append(step_entry)
        self._write_main_log()

        return step_id

    async def log_subagent_call(
        self,
        subagent_name: str,
        prompt: str,
        result: dict[str, Any],
    ) -> None:
        """Log a sub-agent call in the main orchestrator log.

        Args:
            subagent_name: Name of the sub-agent called
            prompt: Prompt given to sub-agent
            result: Minimal result from sub-agent (SubAgentResult format)
        """
        self._step_count += 1
        timestamp = datetime.now()

        step_entry = {
            "step_id": self._step_count,
            "agent": self.main_agent,
            "action": "call_subagent",
            "subagent": subagent_name,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "timestamp": timestamp.isoformat(),
            "result": {
                "success": result.get("success", True),
                "output": result.get("output", "")[:500],
                "artifacts": result.get("artifacts", []),
                "log_file": result.get("log_file"),
            },
        }

        if result.get("error"):
            step_entry["result"]["error"] = result["error"]

        self._steps.append(step_entry)
        self._write_main_log()

    async def log_error(
        self,
        agent_id: str,
        error: str,
        step_id: str | None = None,
    ) -> None:
        """Log an error with full stack trace.

        IMPORTANT: Keep error messages! Agent learns from stack traces.

        Args:
            agent_id: Agent that encountered error
            error: Full error message/stack trace
            step_id: Associated step ID
        """
        timestamp = datetime.now().isoformat()

        with open(self.errors_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Step ID: {step_id}\n")
            f.write(f"Agent: {agent_id}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{error}\n")

    async def log_tool_call(
        self,
        step_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        agent_id: str,
    ) -> str:
        """Log a tool call.

        Args:
            step_id: Parent step ID
            tool_name: Name of tool called
            tool_args: Arguments passed
            agent_id: Agent making the call

        Returns:
            Tool call ID
        """
        call_id = f"{step_id}_tool_{uuid4().hex[:6]}"

        # Tool calls are now logged directly by SubAgentLogger in sub_agent.py
        # This method just returns the call_id for correlation

        return call_id

    async def log_tool_result(
        self,
        call_id: str,
        result: str,
        success: bool = True,
        duration_ms: float | None = None,
    ) -> None:
        """Log a tool result.

        Args:
            call_id: Tool call ID from log_tool_call
            result: Tool result (may be truncated for context)
            success: Whether tool succeeded
            duration_ms: Execution time
        """
        # Tool results are now primarily logged in sub-agent logs
        pass

    async def log_context_event(
        self,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log a context management event.

        Args:
            event_type: Type of event (compaction, summarization, offload)
            details: Event details
        """
        self._step_count += 1
        timestamp = datetime.now()

        step_entry = {
            "step_id": self._step_count,
            "type": "context_event",
            "event_type": event_type,
            "details": details,
            "timestamp": timestamp.isoformat(),
        }

        self._steps.append(step_entry)
        self._write_main_log()

    async def finalize(
        self,
        success: bool = True,
        final_result: Any = None,
        error: str | None = None,
    ) -> None:
        """Finalize the run and write summary.

        Args:
            success: Whether run completed successfully
            final_result: Final result of the run
            error: Error message if run failed
        """
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()

        summary = {
            "run_id": self.run_id,
            "success": success,
            "start_time": self._start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "total_steps": self._step_count,
            "error_count": self._error_count,
            "subagent_logs": [
                sub_logger.log_file.name for sub_logger in self._subagent_loggers
            ],
        }

        if final_result:
            summary["final_result"] = str(final_result)[:1000]

        if error:
            summary["error"] = error

        # Write summary YAML
        with open(self.summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Final write of main log
        self._write_main_log()

        logger.info(
            f"Run {self.run_id} completed: "
            f"{'SUCCESS' if success else 'FAILED'}, "
            f"{self._step_count} steps, "
            f"{self._error_count} errors, "
            f"{duration:.1f}s"
        )

    def get_steps(self) -> list[dict[str, Any]]:
        """Read all steps from the log file.

        Returns:
            List of step entries
        """
        return self._steps

    def get_errors(self) -> str:
        """Read all errors from the error log.

        Returns:
            Error log contents
        """
        if not self.errors_file.exists():
            return ""
        return self.errors_file.read_text()

    def get_summary(self) -> dict[str, Any] | None:
        """Read the run summary.

        Returns:
            Summary dict or None if not finalized
        """
        if not self.summary_file.exists():
            return None
        with open(self.summary_file) as f:
            return yaml.safe_load(f)


__all__ = ["StepLogger", "SubAgentLogger", "_sanitize_tool_results"]
