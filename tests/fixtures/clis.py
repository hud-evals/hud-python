"""Stand-ins for the ``claude`` and ``codex`` executables a CLI agent launches.

Each records what the agent handed it (argv, the variables it set, stdin, ``.hud_*``
files), runs the commands it is told to, sends one model request where its configuration
points (``provider`` standing in for the CLI's built-in endpoint), prints the JSONL stream
the real CLI would, and exits as told. :func:`answer`, :func:`shell`, :func:`fault` and
:func:`prompt` speak either CLI's stream by its executable name.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

#: Variables the CLI agents set for the process they launch.
AGENT_VARIABLES = ("ANTHROPIC_", "CLAUDE_", "CODEX_", "DISABLE_", "HUD_", "IS_SANDBOX")

STUB = """#!{python}
import json, os, subprocess, sys, time, tomllib, urllib.request

behavior = json.load(open({behavior!r}))
record = {{
    "argv": sys.argv[1:],
    "env": {{
        key: value
        for key, value in os.environ.items()
        if key.startswith({prefixes!r}) and value
    }},
    "stdin": sys.stdin.read(),
    "cwd": os.getcwd(),
    "files": {{
        name: open(name).read() for name in sorted(os.listdir(".")) if name.startswith(".hud_")
    }},
    "pid": os.getpid(),
}}


def drive_computer(server, actions):
    relay = subprocess.Popen(
        [server["command"], *server["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def ask(message):
        relay.stdin.write(json.dumps(message) + "\\n")
        relay.stdin.flush()
        while True:
            reply = json.loads(relay.stdout.readline())
            if reply.get("id") == message["id"]:
                return reply

    ask({{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {{
        "protocolVersion": "2025-06-18", "capabilities": {{}},
        "clientInfo": {{"name": "stub", "version": "0"}}}}}})
    initialized = {{"jsonrpc": "2.0", "method": "notifications/initialized"}}
    relay.stdin.write(json.dumps(initialized) + "\\n")
    results = []
    for index, arguments in enumerate(actions, start=1):
        reply = ask({{"jsonrpc": "2.0", "id": index, "method": "tools/call",
                     "params": {{"name": "computer", "arguments": arguments}}}})
        result = reply["result"]
        results.append({{
            "isError": result.get("isError", False),
            "content": [block.get("text") or block["type"] for block in result["content"]],
        }})
    # Like the real CLI, end the stdio server rather than wait for it to drain.
    relay.terminate()
    relay.wait()
    return results


def call_model(provider):
    headers = {{"content-type": "application/json"}}
    if {name!r} == "claude":
        url = (os.environ.get("ANTHROPIC_BASE_URL") or provider) + "/v1/messages"
        if os.environ.get("ANTHROPIC_API_KEY"):
            headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
        if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            headers["authorization"] = "Bearer " + os.environ["ANTHROPIC_AUTH_TOKEN"]
        for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
            header, _, value = line.partition(":")
            headers[header.strip()] = value.strip()
        body = {{"model": os.environ["ANTHROPIC_MODEL"], "max_tokens": 1,
                 "messages": [{{"role": "user", "content": "hi"}}]}}
    else:
        argv = sys.argv[1:]
        config = tomllib.loads("\\n".join(v for flag, v in zip(argv, argv[1:]) if flag == "-c"))
        chosen = config.get("model_providers", {{}}).get(config.get("model_provider"), {{}})
        url = chosen.get("base_url", provider) + "/responses"
        if key := os.environ.get(chosen.get("env_key", "CODEX_API_KEY")):
            headers["authorization"] = "Bearer " + key
        headers.update(chosen.get("http_headers", {{}}))
        body = {{"model": argv[argv.index("--model") + 1], "input": "hi"}}
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    urllib.request.urlopen(request).read()


for command in behavior["commands"]:
    subprocess.run(command, shell=True, check=True)

if behavior.get("provider"):
    call_model(behavior["provider"])

if behavior.get("computer"):
    servers = json.load(open(".hud_mcp_config.json"))["mcpServers"]
    record["computer"] = drive_computer(servers["computer-use"], behavior["computer"])

with open(os.path.join({captures!r}, str(os.getpid()) + ".json"), "w") as capture:
    json.dump(record, capture)

sys.stdout.write(behavior["stdout"])
sys.stdout.flush()
sys.stderr.write(behavior.get("stderr", ""))
sys.stderr.flush()
if behavior.get("hang"):
    time.sleep(60)
sys.exit(behavior.get("exit", 0))
"""


@dataclass
class Stub:
    """A stub CLI installed in ``bin``; ``captures()`` reads what each invocation recorded."""

    root: Path
    name: str
    stdout: str = ""
    stderr: str = ""
    exit: int = 0
    hang: bool = False
    provider: str = ""
    commands: list[str] = field(default_factory=list[str])
    computer: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    @property
    def bin(self) -> Path:
        return self.root / "bin"

    @property
    def captures_dir(self) -> Path:
        return self.root / "captures"

    def install(self) -> None:
        """Write the stub and its behavior, discarding what earlier invocations recorded."""
        self.bin.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        for capture in self.captures_dir.glob("*.json"):
            capture.unlink()
        behavior = self.root / "behavior.json"
        behavior.write_text(
            json.dumps(
                {
                    "stdout": self.stdout,
                    "stderr": self.stderr,
                    "exit": self.exit,
                    "hang": self.hang,
                    "provider": self.provider,
                    "commands": self.commands,
                    "computer": self.computer,
                }
            )
        )
        script = self.bin / self.name
        script.write_text(
            STUB.format(
                python=sys.executable,
                behavior=str(behavior),
                name=self.name,
                prefixes=AGENT_VARIABLES,
                captures=str(self.captures_dir),
            )
        )
        script.chmod(0o755)

    def captures(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text()) for path in sorted(self.captures_dir.glob("*.json"))]

    def shell_env(self, *, path: str | None = None) -> dict[str, str]:
        """The workspace shell environment, with every agent variable this process holds
        blanked so a capture shows only what the agent set.

        Workspace sessions are login shells, and system profiles prepend their own
        directories (where a real ``claude`` may live), so the session's home gets a
        profile that puts ``path`` (default: the stub, then the system) on ``PATH`` last.
        """
        home = self.root / "home"
        home.mkdir(parents=True, exist_ok=True)
        search = path or f"{self.bin}:/usr/local/bin:/usr/bin:/bin"
        (home / ".bash_profile").write_text(f"export PATH={search}\n")
        blanked = {key: "" for key in os.environ if key.startswith(AGENT_VARIABLES)}
        return {**blanked, "HOME": str(home)}


def process_exited(pid: int, *, within: float = 10.0) -> bool:
    """Whether ``pid`` has exited (gone or a zombie), polling for up to ``within`` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        time.sleep(0.05)
    return False


def jsonl(*events: dict[str, Any]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


# ─── what the real CLIs print ──────────────────────────────────────────


def claude_assistant(*content: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": list(content),
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    }


def claude_result(text: str = "done", *, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": text,
        "session_id": "session-1",
        "duration_ms": 5,
        "num_turns": 2,
        "total_cost_usd": 0.01,
    }


def codex_item(kind: str, item_id: str, **fields: Any) -> dict[str, Any]:
    return {"type": "item.completed", "item": {"id": item_id, "type": kind, **fields}}


CODEX_STARTED = {"type": "thread.started", "thread_id": "thread-1"}
CODEX_COMPLETED = {
    "type": "turn.completed",
    "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 8},
}


# ─── either CLI, by executable name ────────────────────────────────────


def answer(cli: str, text: str) -> str:
    """The stream ``cli`` prints when it answers ``text``."""
    if cli == "claude":
        return jsonl(claude_assistant({"type": "text", "text": text}), claude_result(text))
    return jsonl(
        CODEX_STARTED, codex_item("agent_message", "message-1", text=text), CODEX_COMPLETED
    )


def shell(cli: str, command: str, then: str) -> str:
    """The stream ``cli`` prints when it runs ``command`` and then answers ``then``."""
    if cli == "claude":
        return jsonl(
            claude_assistant(
                {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": command}},
                stop="tool_use",
            ),
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ran"}],
                },
            },
            claude_assistant({"type": "text", "text": then}),
            claude_result(then),
        )
    return jsonl(
        CODEX_STARTED,
        codex_item(
            "command_execution",
            "command-1",
            command=command,
            aggregated_output="ran",
            exit_code=0,
            status="completed",
        ),
        codex_item("agent_message", "message-1", text=then),
        CODEX_COMPLETED,
    )


def fault(cli: str, message: str) -> str:
    """The stream ``cli`` prints when its turn fails with ``message``."""
    if cli == "claude":
        return jsonl(claude_result(message, is_error=True))
    return jsonl({"type": "turn.started"}, {"type": "turn.failed", "error": {"message": message}})


def prompt(cli: str, stdin: str) -> str:
    """The prompt ``cli`` read from its stdin: Claude's stream-json message, or codex's text."""
    if cli == "claude":
        return json.loads(stdin)["message"]["content"][0]["text"]
    return stdin
