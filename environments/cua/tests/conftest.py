"""Serve the environment the way HUD does, against local stand-ins for the desktop and the judge.

Nothing here needs a real desktop or a HUD account: a TCP listener stands in for
the VNC server, and a small HTTP server stands in for the inference gateway the
LLM judge calls. The environment runs in its own process, configured only
through environment variables, exactly as ``hud eval`` serves it.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Judge:
    """The inference gateway, judging every criterion MET. ``requests`` records what it was sent."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        judge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                judge.requests.append((self.headers["Authorization"], body))
                verdict = json.dumps({"criterion_status": "MET", "explanation": "stand-in judge"})
                reply = json.dumps({"choices": [{"message": {"content": verdict}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"


@pytest.fixture(autouse=True)
def judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the environment at the stand-in judge, with no HUD key and no user config."""
    stand_in = Judge()
    thread = threading.Thread(target=stand_in.server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HUD_API_KEY", "")
    monkeypatch.setenv("HUD_GATEWAY_URL", stand_in.url)
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "0")
    yield stand_in
    stand_in.server.shutdown()
    thread.join()


@pytest.fixture(autouse=True)
async def desktop(monkeypatch: pytest.MonkeyPatch):
    """A listener on the environment's VNC port, so it starts without a real desktop."""
    server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("VNC_PORT", str(port))
    async with server:
        yield port
