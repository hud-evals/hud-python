"""Stub ``claude`` and ``codex`` executables for the CLI agent scenarios.

A stub is a Python script placed on the workspace's ``PATH``. Each invocation
records what the agent handed it (argv, the environment variables the agent
set, stdin, its working directory and the ``.hud_*`` files there) to a capture
file, optionally drives a computer-use MCP server named in its MCP config,
then prints a canned JSONL stream, writes stderr and exits as told.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Variables the CLI agents set for the process they launch.
AGENT_VARIABLES = ("ANTHROPIC_", "CLAUDE_", "CODEX_", "DISABLE_", "HUD_", "IS_SANDBOX")

STUB = """#!{python}
import json, os, subprocess, sys, time

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
    computer: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    @property
    def bin(self) -> Path:
        return self.root / "bin"

    @property
    def captures_dir(self) -> Path:
        return self.root / "captures"

    def install(self) -> None:
        self.bin.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        behavior = self.root / "behavior.json"
        behavior.write_text(
            json.dumps(
                {
                    "stdout": self.stdout,
                    "stderr": self.stderr,
                    "exit": self.exit,
                    "hang": self.hang,
                    "computer": self.computer,
                }
            )
        )
        script = self.bin / self.name
        script.write_text(
            STUB.format(
                python=sys.executable,
                behavior=str(behavior),
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


def process_exited(pid: int, *, within: float = 5.0) -> bool:
    """Whether ``pid`` has exited (gone or a zombie), polling for up to ``within`` seconds."""
    status = Path(f"/proc/{pid}/status")
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            state = next(
                line for line in status.read_text().splitlines() if line.startswith("State:")
            )
        except FileNotFoundError:
            return True
        if "Z" in state.split()[1]:
            return True
        time.sleep(0.05)
    return False


def jsonl(*events: dict[str, Any]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)
