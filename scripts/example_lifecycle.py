"""Exercise built example images through the HUD task and capability protocols."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hud import Runtime, connect
from hud.capabilities import RFBClient, SSHClient
from hud.eval import Taskset

KEY = "example-ci-key"
SERVICE = "http://host.docker.internal:18080"
REQUESTS: list[str] = []
DOCKER = shutil.which("docker") or "/usr/bin/docker"
SECURITY = Path(__file__).resolve().parents[1] / "hud/eval/docker-seccomp.json"


class Services(BaseHTTPRequestHandler):
    """HTTP fixtures for data-file storage and the inference gateway."""

    def do_GET(self) -> None:
        if self.path.startswith("/v2/") and self.headers.get("Authorization") != f"Bearer {KEY}":
            self.send_error(401)
            return
        REQUESTS.append(self.path)
        if self.path == "/v2/data/notes":
            payload = json.dumps({"filename": "notes.txt"}).encode()
        elif self.path == "/v2/data/notes/download":
            payload = json.dumps({"url": f"{SERVICE}/notes.txt"}).encode()
        elif self.path == "/notes.txt":
            if self.headers.get("Authorization"):
                self.send_error(400, "storage must not receive the HUD key")
                return
            payload = b"The answer is violet.\n"
        elif self.path == "/page":
            payload = b"<html><title>Example</title><body>free encyclopedia</body></html>"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != "/chat/completions" or self.headers.get("Authorization") != f"Bearer {KEY}":
            self.send_error(401)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        assert "<criterion>" in prompt and "<response>" in prompt
        REQUESTS.append(self.path)
        verdict = "UNMET" if "wrong answer" in prompt else "MET"
        payload = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"criterion_status": verdict, "explanation": "fixture"}
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def docker(*args: str) -> str:
    return subprocess.run([DOCKER, *args], text=True, stdout=subprocess.PIPE, check=True).stdout


async def check(example: str, solved: bool) -> None:
    name = f"hud-example-{example}-{'pass' if solved else 'fail'}"
    docker(
        "run",
        "--detach",
        "--name",
        name,
        "--publish",
        "127.0.0.1:8765:8765",
        "--shm-size",
        "2g",
        "--add-host",
        "host.docker.internal:host-gateway",
        "--security-opt",
        f"seccomp={SECURITY}",
        "--security-opt",
        "systempaths=unconfined",
        "--security-opt",
        "apparmor=unconfined",
        "--env",
        f"HUD_API_URL={SERVICE}",
        "--env",
        f"HUD_GATEWAY_URL={SERVICE}",
        "--env",
        f"HUD_API_KEY={KEY}",
        "--env",
        "HUD_TELEMETRY_ENABLED=false",
        f"hud-example:{example}",
    )
    try:
        tasks = list(Taskset.from_file("tasks.py"))
        task = tasks[0]
        args = dict(task.args)
        if example == "cua":
            args["bash_checks"] = [
                {
                    **check,
                    "command": check["command"].replace(
                        "https://www.wikipedia.org/", f"{SERVICE}/page"
                    ),
                }
                for check in args["bash_checks"]
            ]
        elif example == "argument-hints":
            args.update(
                prompt="Read notes.txt and report the answer.",
                attachments=[{"file_id": "notes"}],
                criteria=[{"requirement": "Answers violet.", "weight": 1.0}],
            )

        async with connect(Runtime("tcp://127.0.0.1:8765"), ready_timeout=60) as client:
            frame = await client.start_task(task.id, args)
            assert frame.get("prompt"), frame
            if example == "coding" and solved:
                # Apply the golden source through the agent's shell, not through Docker.
                source = docker(
                    "exec",
                    name,
                    "git",
                    "-C",
                    "/hud/baseline",
                    "show",
                    "origin/flask_4992_golden:src/flask/config.py",
                )
                shell = await client.open("shell")
                assert isinstance(shell, SSHClient)
                await shell.write_text("/app/src/flask/config.py", source)
            elif example == "cua":
                screen = await client.open("screen")
                assert isinstance(screen, RFBClient)
                screenshot, mime_type = await screen.screenshot_png()
                assert mime_type == "image/png" and screenshot.startswith(b"\x89PNG\r\n\x1a\n")
                if solved:
                    screen.conn.keyboard.press("Control_L", "l")
                    screen.conn.keyboard.write(f"{SERVICE}/page")
                    screen.conn.keyboard.press("Return")
                    await screen.drain()
                    for _ in range(30):
                        if "/page" in REQUESTS:
                            break
                        await asyncio.sleep(0.2)
                    assert "/page" in REQUESTS
            elif example == "argument-hints":
                assert frame["data_files"] == [{"path": "files/notes.txt", "file_id": "notes"}]
                shell = await client.open("ssh")
                assert isinstance(shell, SSHClient)
                assert (
                    await shell.read_text("/workspace/files/notes.txt") == "The answer is violet.\n"
                )
            answer = "4" if example == "blank" else "violet, free encyclopedia"
            result = await client.grade({"answer": answer if solved else "wrong answer"})
            assert not result.get("isError"), result
            assert result["score"] == float(solved), result
        print(f"{example}: {'passing' if solved else 'failing'} task verified")
    finally:
        print(docker("logs", name))
        docker("rm", "--force", name)


async def main() -> None:
    example = sys.argv[1]
    server = ThreadingHTTPServer(("0.0.0.0", 18080), Services)  # noqa: S104 - reachable by containers
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        await check(example, False)
        await check(example, True)
        if example in {"cua", "argument-hints"}:
            assert REQUESTS.count("/chat/completions") == 2, REQUESTS
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    asyncio.run(main())
