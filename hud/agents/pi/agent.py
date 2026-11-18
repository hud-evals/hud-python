"""Pi CLI harness over a workspace SSH capability."""

from __future__ import annotations

import json
import logging
import re
import shlex
from typing import TYPE_CHECKING, Any, cast

from hud.agents.cli import WINDOWS_SHELLS, CLIAgent, model_endpoint, resolve_executable, run_jsonl
from hud.agents.types import AgentStep, PiConfig, ToolStep, Usage
from hud.types import ContentBlock, ImageContent, Step, TextContent, ToolCall, ToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities.ssh import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

# Pi accepts these characters in an MCP server name and treats '-' and '_' as the same.
_PI_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]+")

_MANAGED_PI_PATHS = {
    "linux-x64": "/usr/local/lib/agents/pi/pi",
}


class PiEvents:
    """Translate ``pi --mode json`` events into canonical HUD steps."""

    def __init__(self, run: Run, *, started_at: str) -> None:
        self.run = run
        self.message_started_at = started_at
        self.pending_calls: dict[str, tuple[ToolCall, str]] = {}
        self.settled = False
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
            case "message_start":
                self.message_started_at = received_at
            case "message_end":
                self.record(event["message"], received_at)
            case "agent_settled":
                self.settled = True
                if event["aborted"]:
                    self.error = "pi aborted the run"

    def record(self, message: dict[str, Any], received_at: str) -> None:
        match message["role"]:
            case "system" | "user":
                return
            case "assistant":
                self.record_assistant(message, received_at)
            case "toolResult":
                call, started_at = self.pending_calls.pop(message["toolCallId"])
                content: list[ContentBlock] = []
                for block in message["content"]:
                    match block["type"]:
                        case "text":
                            content.append(TextContent(type="text", text=block["text"]))
                        case "image":
                            content.append(
                                ImageContent(
                                    type="image", data=block["data"], mimeType=block["mimeType"]
                                )
                            )
                        case kind:
                            raise ValueError(f"unsupported Pi tool result block {kind!r}")
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
        stop_reason = message["stopReason"]
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
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = error or stderr.strip() or f"pi CLI exited with return code {returncode}"
        elif error is None and not self.settled:
            error = "pi CLI exited without an agent_settled event"
        elif error is None and self.pending_calls:
            missing = ", ".join(sorted(self.pending_calls))
            error = f"pi CLI exited without results for tool calls: {missing}"

        if error is not None and stderr:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)


class PiAgent(CLIAgent[PiConfig]):
    """Runs ``pi --mode json`` over SSH inside the environment workspace.

    Pi reads a throwaway agent directory holding the model endpoint and the environment's
    MCP servers, so the workspace user's Pi login and settings stay out of the run.
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
        provider: dict[str, Any] = {
            "baseUrl": endpoint.url,
            "api": "openai-completions",
            "apiKey": "$INFERENCE_API_KEY",
            "models": [{"id": self.config.model}],
        }
        if endpoint.headers:
            provider["headers"] = endpoint.headers
        files = {"models.json": {"providers": {"inference": provider}}}

        mcp_servers: dict[str, dict[str, Any]] = {}
        taken: dict[str, str] = {}
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
            if (other := taken.setdefault(cap.name.replace("-", "_"), cap.name)) != cap.name:
                raise ValueError(f"pi treats MCP servers {other!r} and {cap.name!r} as one name")
            # Declared like built-in tools, as Claude Code and Codex declare them.
            server: dict[str, Any] = {"url": cap.url, "exposure": "direct"}
            if token := cap.params.get("auth_token"):
                # Pi expands the variable, which keeps the token out of the config file.
                variable = f"HUD_MCP_TOKEN_{len(mcp_servers)}"
                env[variable] = token
                server["headers"] = {"Authorization": f"Bearer ${{{variable}}}"}
            mcp_servers[cap.name] = server
        if mcp_servers:
            files["mcp.json"] = {"mcpServers": mcp_servers}

        args = [
            executable,
            "--mode",
            "json",
            "--no-session",
            # The workspace is the environment's, so its .pi and .agents resources apply.
            "--approve",
            "--offline",
            "--provider",
            "inference",
            "--model",
            self.config.model,
        ]
        connection = endpoint.connection
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        cli = " ".join(shlex.quote(arg) for arg in args)
        statements = ['pi_home=$(mktemp -d "${TMPDIR:-/tmp}/hud-pi.XXXXXX") || exit 1']
        # Under exec no trap survives; the sandbox holding a guarded connection is discarded
        # after the rollout.
        if connection is None:
            statements.append("trap 'rm -rf -- \"$pi_home\"' EXIT")
        statements.append('export PI_CODING_AGENT_DIR="$pi_home"')
        statements.extend(
            f'printf %s {shlex.quote(json.dumps(body))} > "$pi_home/{name}"'
            for name, body in files.items()
        )
        statements.append('export PATH="$HOME/.local/bin:$PATH"')
        # A guarded connection is bound to the launched process, so exec the CLI itself.
        exec_ = "exec env " if connection is not None else ""
        statements.append(f"{exec_}{env_prefix} {cli}")
        command = "; ".join(statements)

        logger.info("SSH exec pi CLI (%d chars)", len(command))
        events = PiEvents(run, started_at=now_iso())
        returncode, stderr = await run_jsonl(
            ssh,
            command,
            events.consume,
            input_text=run.prompt_text,
            connections=(connection,) if connection is not None else (),
        )
        logger.info("exit=%s stderr=%d", returncode, len(stderr))
        events.finish(returncode=returncode, stderr=stderr)
