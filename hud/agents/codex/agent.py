"""Codex CLI harness over a workspace SSH capability."""

from __future__ import annotations

import json
import logging
import shlex
from typing import TYPE_CHECKING, Any, cast

from hud.agents.cli import (
    WINDOWS_SHELLS,
    CLIAgent,
    model_endpoint,
    powershell,
    powershell_quote,
    resolve_executable,
    run_jsonl,
)
from hud.agents.types import AgentStep, CodexCLIConfig, ToolStep
from hud.types import Step, TextContent, ToolCall, ToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.capabilities import SSHClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

_MANAGED_CODEX_PATHS = {
    "linux-x64": "/usr/local/lib/agents/codex/bin/codex",
    "linux-x64-musl": "/usr/local/lib/agents/codex/bin/codex",
}


class CodexEvents:
    """Translate ``codex exec --json`` events into canonical HUD steps."""

    def __init__(self, run: Run, *, model: str, started_at: str) -> None:
        self.run = run
        self.model = model
        self.agent_started_at = started_at
        self.item_started_at: dict[str, str] = {}
        self.saw_completion = False
        self.error: str | None = None

    def consume(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        event = json.loads(line)
        received_at = now_iso()
        match event.get("type"):
            case "thread.started":
                self.run.trace.extra["codex_thread_id"] = event["thread_id"]
            case "turn.started":
                self.agent_started_at = received_at
            case "item.started":
                item = event["item"]
                self.item_started_at[item["id"]] = received_at
            case "item.completed":
                self.record(event["item"], received_at)
            case "turn.completed":
                self.run.trace.extra["usage"] = event["usage"]
                self.saw_completion = True
            case "turn.failed":
                self.error = event["error"]["message"]
            case "error":
                self.error = event["message"]

    def finish(self, *, returncode: int, stderr: str) -> None:
        trace = self.run.trace
        error = self.error
        if returncode != 0:
            trace.extra["returncode"] = returncode
            error = error or stderr.strip() or f"codex CLI exited with return code {returncode}"
        elif error is None and not self.saw_completion:
            error = "codex CLI exited without a turn.completed event"

        if error is not None and stderr and self.error is None:
            trace.extra["stderr"] = stderr
        if error is not None:
            raise RuntimeError(error)

    def record(self, item: dict[str, Any], received_at: str) -> None:
        item_id = item["id"]
        started_at = self.item_started_at.pop(item_id, self.agent_started_at)
        match item["type"]:
            case "agent_message":
                text = item["text"]
                self.run.trace.content = text
                self.run.record(
                    AgentStep(
                        content=text,
                        model=self.model,
                        raw=item,
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
            case "reasoning":
                self.run.record(
                    AgentStep(
                        reasoning=item["text"],
                        model=self.model,
                        raw=item,
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
            case "command_execution" | "file_change" | "mcp_tool_call" | "web_search":
                self.record_tool(item, started_at, received_at)
            case _:
                self.run.record(
                    Step(
                        source="agent",
                        extra={"codex_item": item},
                        started_at=started_at,
                        ended_at=received_at,
                    )
                )
        self.agent_started_at = received_at

    def record_tool(self, item: dict[str, Any], started_at: str, ended_at: str) -> None:
        call_id = item["id"]
        match item["type"]:
            case "command_execution":
                call = ToolCall(
                    id=call_id,
                    name="shell",
                    arguments={"command": item["command"]},
                )
                result = ToolResult(
                    call_id=call_id,
                    content=[TextContent(type="text", text=item["aggregated_output"])],
                    isError=item["status"] != "completed" or item["exit_code"] not in (None, 0),
                )
            case "file_change":
                changes = item["changes"]
                call = ToolCall(
                    id=call_id,
                    name="apply_patch",
                    arguments={"changes": changes},
                )
                result = ToolResult(
                    call_id=call_id,
                    content=[
                        TextContent(
                            type="text",
                            text="\n".join(
                                f"{change['kind']}: {change['path']}" for change in changes
                            ),
                        )
                    ],
                    isError=item["status"] != "completed",
                )
            case "mcp_tool_call":
                call = ToolCall(
                    id=call_id,
                    name=item["tool"],
                    provider_name=f"{item['server']}.{item['tool']}",
                    arguments=item["arguments"],
                )
                raw_result = item.get("result") or {}
                error = item.get("error")
                result = ToolResult.model_validate(
                    {
                        "call_id": call_id,
                        "content": raw_result.get("content")
                        or ([{"type": "text", "text": error["message"]}] if error else []),
                        "structuredContent": raw_result.get("structured_content"),
                        "_meta": raw_result.get("_meta"),
                        "isError": item["status"] == "failed",
                    }
                )
            case "web_search":
                call = ToolCall(
                    id=call_id,
                    name="web_search",
                    arguments={"query": item["query"], "action": item["action"]},
                )
                result = ToolResult(
                    call_id=call_id,
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(item["action"], separators=(",", ":")),
                        )
                    ],
                    isError=False,
                )
            case _:
                raise ValueError(f"unsupported Codex tool item {item['type']!r}")

        self.run.record(
            ToolStep(
                call=call,
                result=result,
                extra={"codex_item": item},
                started_at=started_at,
                ended_at=ended_at,
            )
        )


class CodexCLIAgent(CLIAgent[CodexCLIConfig]):
    """Runs ``codex exec`` over SSH inside the environment workspace."""

    config_cls = CodexCLIConfig
    provider = "openai"

    async def __call__(self, run: Run) -> None:
        endpoint = model_endpoint(run, self.provider, gateway=self.config.gateway)
        ssh = cast("SSHClient", await run.open("ssh"))
        connection = endpoint.connection if endpoint is not None else None
        executable = await resolve_executable(
            ssh, "codex", _MANAGED_CODEX_PATHS, run.runtime_config
        )
        if connection is not None:
            # A launcher script (the npm package's Node shim) runs the real client as a
            # child, which the connection refuses; without this Codex retries silently.
            head = await ssh.run(
                f"head -c 2 -- {shlex.quote(executable)}", check=True, encoding=None
            )
            if head.stdout == b"#!":
                raise RuntimeError(
                    f"{executable} is a launcher script, but a process-bound inference "
                    "connection only reaches the launched process; put the native codex "
                    "binary on PATH or provide the managed runtime bundle"
                )
        env: dict[str, str] = {}
        args = [
            executable,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            # The workspace already isolates a guarded process; Codex's inner sandbox would
            # block the relay socket it needs.
            "danger-full-access" if connection is not None else self.config.sandbox,
            "--model",
            self.config.model,
        ]
        if self.config.reasoning_effort is not None:
            args.extend(
                ["-c", f"model_reasoning_effort={json.dumps(self.config.reasoning_effort)}"]
            )

        if endpoint is not None:
            env["INFERENCE_API_KEY"] = endpoint.key
            overrides = {
                "model_provider": "inference",
                "model_providers.inference.name": "Inference",
                "model_providers.inference.base_url": endpoint.url,
                "model_providers.inference.env_key": "INFERENCE_API_KEY",
                "model_providers.inference.wire_api": "responses",
            }
            for name, value in overrides.items():
                args.extend(["-c", f"{name}={json.dumps(value)}"])
            if endpoint.headers:
                pairs = ",".join(
                    f"{json.dumps(name)}={json.dumps(value)}"
                    for name, value in endpoint.headers.items()
                )
                args.extend(["-c", f"model_providers.inference.http_headers={{{pairs}}}"])

        args.append("-")
        # Injected credentials get a throwaway CODEX_HOME; otherwise an ambient login applies.
        if ssh.capability.params.get("shell", "bash") in WINDOWS_SHELLS:
            invocation = (
                f"& {powershell_quote(executable)} "
                f"{' '.join(powershell_quote(arg) for arg in args[1:])}; "
                "$hudExitCode=$LASTEXITCODE"
            )
            statements = [
                *(f"$env:{key}={powershell_quote(value)}" for key, value in env.items()),
                invocation,
            ]
            if env:
                statements = [
                    "$codexHome=Join-Path ([System.IO.Path]::GetTempPath()) "
                    "('hud-codex-' + [System.Guid]::NewGuid())",
                    "New-Item -ItemType Directory -Force -Path $codexHome | Out-Null",
                    "$env:CODEX_HOME=$codexHome",
                    *statements[:-1],
                    f"try {{ {invocation} }} finally {{ Remove-Item -Recurse -Force $codexHome }}",
                ]
            command = powershell(";".join([*statements, "exit $hudExitCode"]))
        else:
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
            cli = " ".join(shlex.quote(arg) for arg in args)
            statements = ['export PATH="$HOME/.local/bin:$PATH"', f"{env_prefix} {cli}".strip()]
            if connection is not None:
                # A guarded connection is bound to the launched process, so exec the CLI itself;
                # the sandbox is discarded after the rollout, so its CODEX_HOME needs no trap.
                statements = [
                    'codex_home=$(mktemp -d "${TMPDIR:-/tmp}/hud-codex.XXXXXX") || exit 1',
                    'export CODEX_HOME="$codex_home"',
                    statements[0],
                    f"exec env {env_prefix} {cli}",
                ]
            elif env:
                statements = [
                    'codex_home=$(mktemp -d "${TMPDIR:-/tmp}/hud-codex.XXXXXX") || exit 1',
                    "trap 'rm -rf -- \"$codex_home\"' EXIT",
                    'export CODEX_HOME="$codex_home"',
                    *statements,
                ]
            command = "; ".join(statements)

        logger.info("SSH exec codex CLI (%d chars)", len(command))
        events = CodexEvents(run, model=self.config.model, started_at=now_iso())
        returncode, stderr = await run_jsonl(
            ssh,
            command,
            events.consume,
            input_text=run.prompt_text,
            connections=(connection,) if connection is not None else (),
        )
        logger.info("exit=%s stderr=%d", returncode, len(stderr))
        events.finish(returncode=returncode, stderr=stderr)
