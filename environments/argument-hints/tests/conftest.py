"""Serve the environment the way HUD does, against a local stand-in for the HUD platform.

The stand-in answers the three things this environment calls: the data-file API
(metadata and a presigned download URL), the storage behind that URL, and the
inference gateway the LLM judge uses. The environment runs in its own process,
configured only through environment variables, exactly as ``hud eval`` serves it.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class DataFile:
    filename: str
    content: bytes
    downloadable: bool


@dataclass
class Received:
    method: str
    path: str
    authorization: str | None
    body: dict = field(default_factory=dict)


class Platform:
    """The HUD API, storage and gateway this environment talks to.

    The judge grades like a perfect answer: positive criteria MET, negative ones
    UNMET. An answer listed in ``rejected_answers`` gets an HTTP 400 instead.
    """

    def __init__(self) -> None:
        self.files: dict[str, DataFile] = {}
        self.rejected_answers: set[str] = set()
        self.received: list[Received] = []
        self.judging = threading.Barrier(1)
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                platform.received.append(Received("GET", self.path, self.headers["Authorization"]))
                parts = self.path.strip("/").split("/")
                if parts[0] == "storage" and parts[1] in platform.files:
                    self.reply(200, platform.files[parts[1]].content, "application/octet-stream")
                elif parts[:2] == ["v2", "data"] and parts[2] in platform.files:
                    data_file = platform.files[parts[2]]
                    if parts[3:] == ["download"]:
                        url = f"{platform.url}/storage/{parts[2]}" if data_file.downloadable else None
                        self.reply_json(200, {"url": url})
                    else:
                        self.reply_json(200, {"filename": data_file.filename})
                else:
                    self.reply_json(404, {"detail": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                platform.received.append(Received("POST", self.path, self.headers["Authorization"], body))
                prompt = body["messages"][-1]["content"]
                answer = prompt.split("<response>\n", 1)[1].split("\n</response>", 1)[0]
                platform.judging.wait(timeout=10)
                if answer in platform.rejected_answers:
                    self.reply_json(400, {"error": {"message": "judge rejected request"}})
                    return
                met = "<criterion_type>\npositive" in prompt
                verdict = {"criterion_status": "MET" if met else "UNMET", "explanation": "stand-in"}
                self.reply_json(200, {"choices": [{"message": {"content": json.dumps(verdict)}}]})

            def reply_json(self, status: int, payload: dict) -> None:
                self.reply(status, json.dumps(payload).encode(), "application/json")

            def reply(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def upload(self, file_id: str, filename: str, content: bytes = b"", *, downloadable: bool = True) -> None:
        """Make ``file_id`` known to the data-file API; ``downloadable=False`` withholds its URL."""
        self.files[file_id] = DataFile(filename, content, downloadable)

    def requests(self, method: str, prefix: str) -> list[Received]:
        return [item for item in self.received if item.method == method and item.path.startswith(prefix)]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture(autouse=True)
def platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Path):
    """Point the environment at the stand-in platform with a runtime key and no user config."""
    stand_in = Platform()
    thread = threading.Thread(target=stand_in.server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("HUD_API_KEY", "runtime-key")
    monkeypatch.setenv("HUD_API_URL", stand_in.url)
    monkeypatch.setenv("HUD_GATEWAY_URL", stand_in.url)
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "0")
    yield stand_in
    stand_in.server.shutdown()
    thread.join()
