"""Pi CLI harness over a workspace SSH capability."""

from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hud.agents.cli import WINDOWS_SHELLS, CLIAgent, model_endpoint, resolve_executable, run_jsonl
from hud.agents.types import AgentStep, PiConfig, ToolStep, Usage
from hud.types import ContentBlock, ImageContent, Step, TextContent, ToolCall, ToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities.ssh import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

_PI_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]+")

_MANAGED_PI_PATHS = {
    "linux-x64": "/usr/local/lib/agents/pi/pi",
}

_MAX_STEPS_EXTENSION = Path(__file__).parent / "max-steps.js"


class PiEvents:
    """Translate ``pi --mode json`` events into canonical HUD steps."""

    def __init__(self, run: Run, *, started_at: str, max_steps: int) -> None:
        self.run = run
        self.message_started_at = started_at
        self.max_steps = max_steps
        self.turns = 0
        self.aborted = False
        self.pending_calls: dict[str, tuple[ToolCall, str]] = {}
        self.settled = False
        self.stop_reason: str | None = None
        self.error: str | None = None

    def consume(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        event = json.loads(line)
        received_at = now_iso()
        match event["type"]:
            case "session":
                self.run.trace.extra["pi_session_id"] = event["id"]
            case "turn_end":
                self.turns += 1
            case "message_start":
                self.message_started_at = received_at
            case "message_end":
                self.record(event["message"], received_at)
            case "agent_settled":
                self.settled = True
                self.aborted = event["aborted"]

    def record(self, message: dict[str, Any], received_at: str) -> None:
        match message["role"]:
            case "user":
                return
            case "assistant":
                self.record_assistant(message, received_at)
            case "toolResult":
                call, started_at = self.pending_calls.pop(message["toolCallId"])
                content: list[ContentBlock] = [
                    TextContent(type="text", text=block["text"])
                    if block["type"] == "text"
                    else ImageContent(type="image", data=block["data"], mimeType=block["mimeType"])
                    for block in message["content"]
                ]
                self.run.record(
                    ToolStep(
                        call=call,
                        result=ToolResult(
                            call_id=call.id, content=content, isError=message["isError"]
                        ),
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
            case _:
                self.run.record(
                    Step(
                        source="agent",
                        extra={"pi_message": message},
                        started_at=self.message_started_at,
                        ended_at=received_at,
                    )
                )

    def record_assistant(self, message: dict[str, Any], received_at: str) -> None:
        texts: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        for block in message["content"]:
            match block["type"]:
                case "text":
                    texts.append(block["text"])
                case "thinking":
                    thoughts.append(block["thinking"])
                case "toolCall":
                    calls.append(
                        ToolCall(id=block["id"], name=block["name"], arguments=block["arguments"])
                    )
        usage = message["usage"]
        stop_reason = self.stop_reason = message["stopReason"]
        content = "".join(texts) or None
        self.run.record(
            AgentStep(
                content=content,
                reasoning="\n".join(thoughts) or None,
                tool_calls=calls,
                finish_reason=stop_reason,
                model=message.get("responseModel") or message["model"],
                usage=Usage(
                    prompt_tokens=usage["input"] + usage["cacheRead"] + usage["cacheWrite"],
                    completion_tokens=usage["output"],
                    cached_tokens=usage["cacheRead"],
                ),
                raw=message,
                started_at=self.message_started_at,
                ended_at=received_at,
            )
        )
        if content is not None:
            self.run.trace.content = content
        for call in calls:
            self.pending_calls[call.id] = (call, received_at)
        # Pi exits 0 after a failed model call; the failure is the last message's stop reason.
        if stop_reason in ("error", "aborted"):
            self.error = message.get("errorMessage") or f"pi model call {stop_reason}"
        else:
            self.error = None

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self.run.trace
        error = self.error
        if self.aborted and 0 < self.max_steps < self.turns:
            trace.stop_reason = "max_steps"
            error = None
        elif self.aborted and error is None:
            error = "pi aborted the run"
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = error or stderr.strip() or f"pi CLI exited with return code {returncode}"
        elif error is None and not self.settled:
            error = "pi CLI exited without an agent_settled event"
        elif error is None and self.stop_reason == "length":
            trace.stop_reason = "length"

        if error is not None and stderr:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)


class PiAgent(CLIAgent[PiConfig]):
    """Runs ``pi --mode json`` over SSH inside the environment workspace.

    Pi reads a throwaway agent directory, so the workspace user's Pi login and settings
    stay out of the run.
    """

    config_cls = PiConfig
    provider = "openai"

    async def __call__(self, run: Run) -> None:
        endpoint = model_endpoint(run, self.provider, gateway=self.config.gateway)
        if endpoint is None:
            raise RuntimeError(
                "pi needs a model endpoint: set HUD_API_KEY to use the HUD gateway, or "
                "OPENAI_API_KEY (with OPENAI_BASE_URL) for an OpenAI-compatible endpoint"
            )
        ssh = cast("SSHClient", await run.client.open("ssh"))
        if (shell := ssh.capability.params.get("shell", "bash")) in WINDOWS_SHELLS:
            raise RuntimeError(f"pi runs only in POSIX workspaces, not a {shell} shell")
        executable = await resolve_executable(ssh, "pi", _MANAGED_PI_PATHS, run.runtime_config)
        manifest = run.client.manifest
        assert manifest is not None

        env = {"INFERENCE_API_KEY": endpoint.key}
        model: dict[str, Any] = {**self.config.model_info, "id": self.config.model}
        if body := self.config.request_body:
            model["samplingParams"] = body
        provider: dict[str, Any] = {
            "baseUrl": endpoint.url,
            "api": "openai-completions",
            "apiKey": "$INFERENCE_API_KEY",
            "models": [model],
        }
        if endpoint.headers:
            provider["headers"] = endpoint.headers
        files = {"models.json": json.dumps({"providers": {"inference": provider}})}

        mcp_servers: dict[str, dict[str, Any]] = {}
        for cap in manifest.bindings:
            if cap.protocol.partition("/")[0] != "mcp":
                continue
            if cap.params.get("transport") != "streamable-http":
                raise ValueError(
                    f"MCP server {cap.name!r} uses transport {cap.params.get('transport')!r}; "
                    "pi reaches only streamable-HTTP MCP servers"
                )
            if not _PI_SERVER_NAME.fullmatch(cap.name):
                raise ValueError(
                    f"MCP server name {cap.name!r} is not a pi server name; "
                    "use letters, digits, '_' and '-'"
                )
            server: dict[str, Any] = {"url": cap.url, "exposure": "direct"}
            if token := cap.params.get("auth_token"):
                # Pi expands the variable, so the token stays out of the config file.
                variable = f"HUD_MCP_TOKEN_{len(mcp_servers)}"
                env[variable] = token
                server["headers"] = {"Authorization": f"Bearer ${{{variable}}}"}
            mcp_servers[cap.name] = server
        if mcp_servers:
            files["mcp.json"] = json.dumps({"mcpServers": mcp_servers})

        args = [
            executable,
            "--mode",
            "json",
            "--no-session",
            # Trusts the workspace's own .pi resources, which belong to the environment.
            "--approve",
            "--offline",
            "--provider",
            "inference",
            "--model",
            self.config.model,
        ]
        connection = endpoint.connection
        cli = " ".join(shlex.quote(arg) for arg in args)
        if self.config.system_prompt is not None:
            # Passed as a file: Pi reads a value that names an existing file as that file.
            files["system.md"] = self.config.system_prompt
            cli += ' --append-system-prompt "$pi_home/system.md"'
        # The step limit stays outside extensions/, so no configured extension can replace it.
        files.update(
            {f"extensions/{name}": source for name, source in self.config.extensions.items()}
        )
        if self.config.max_steps > 0:
            files["max-steps.js"] = _MAX_STEPS_EXTENSION.read_text()
            env["HUD_PI_MAX_STEPS"] = str(self.config.max_steps)
            cli += ' --extension "$pi_home/max-steps.js"'
        statements = ['pi_home=$(mktemp -d "${TMPDIR:-/tmp}/hud-pi.XXXXXX") || exit 1']
        if connection is None:
            statements.append("trap 'rm -rf -- \"$pi_home\"' EXIT")
        statements.append('export PI_CODING_AGENT_DIR="$pi_home"')
        statements.append('mkdir "$pi_home/extensions"')
        statements.extend(
            f'printf %s {shlex.quote(body)} > "$pi_home/{name}"' for name, body in files.items()
        )
        statements.append('export PATH="$HOME/.local/bin:$PATH"')
        # A guarded connection is bound to the launched process, so exec the CLI itself; the
        # sandbox is discarded after the rollout, so its agent directory needs no trap.
        exec_ = "exec env " if connection is not None else ""
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        statements.append(f"{exec_}{env_prefix} {cli}")
        command = "; ".join(statements)

        logger.info("SSH exec pi CLI (%d chars)", len(command))
        events = PiEvents(run, started_at=now_iso(), max_steps=self.config.max_steps)
        returncode, stderr = await run_jsonl(
            ssh,
            command,
            events.consume,
            input_text=run.prompt_text,
            connections=(connection,) if connection is not None else (),
        )
        logger.info("exit=%s stderr=%d", returncode, len(stderr))
        events.finish(returncode=returncode, stderr=stderr)
