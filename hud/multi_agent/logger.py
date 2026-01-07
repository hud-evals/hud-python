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


class StepLogger:
    """Log every agent step with full context.

    Logs are saved in YAML format for human readability.

    Example:
        logger = StepLogger(log_dir=Path("./.logs"), run_id="abc123")

        # Log steps
        await logger.log_step(...)

        # Main log saved to .logs/abc123/main.yaml
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

        # _build_result always includes 'output' with primary content
        output = result.get("output", "")
        if isinstance(output, str) and len(output) > 500:
            output = output[:500] + "..."

        step_entry = {
            "step_id": self._step_count,
            "agent": self.main_agent,
            "action": "call_subagent",
            "subagent": subagent_name,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "timestamp": timestamp.isoformat(),
            "result": {
                "success": result.get("success", True),
                "output": output,
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
        return call_id

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


__all__ = ["StepLogger"]
