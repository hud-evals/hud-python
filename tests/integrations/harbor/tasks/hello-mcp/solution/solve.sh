#!/bin/sh
set -eu
# Call the MCP server the way an agent in the workspace would.
python3 - <<'PY'
import json
from pathlib import Path
from urllib.request import Request, urlopen

url = "http://mcp-server:8000/mcp"
session_id = None


def post(payload):
    global session_id
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id is not None:
        headers["Mcp-Session-Id"] = session_id
    request = Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request) as response:
        body = response.read().decode()
        session_id = response.headers.get("Mcp-Session-Id") or session_id
    data = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    return json.loads(data[-1] if data else body) if body else None


post({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "hud-test", "version": "1"},
    },
})
post({"jsonrpc": "2.0", "method": "notifications/initialized"})
result = post({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "get_secret", "arguments": {}},
})
Path("/app/secret.txt").write_text(result["result"]["content"][0]["text"])
PY
