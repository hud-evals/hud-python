#!/usr/bin/env python3
"""Test CDP endpoint discovery"""

import asyncio
import subprocess
import socket
import time
import requests

async def test_cdp():
    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    print(f"Starting Chromium on port {port}")

    # Start chromium with CDP
    proc = subprocess.Popen([
        "chromium", "--headless", "--no-sandbox", "--disable-dev-shm-usage",
        f"--remote-debugging-port={port}", "about:blank"
    ])

    # Wait for startup
    time.sleep(3)

    try:
        # Query CDP endpoints
        response = requests.get(f"http://localhost:{port}/json")
        endpoints = response.json()

        print(f"Available CDP endpoints:")
        for endpoint in endpoints:
            print(f"  - {endpoint.get('webSocketDebuggerUrl', 'No WebSocket URL')}")
            print(f"    Type: {endpoint.get('type')}")
            print(f"    Title: {endpoint.get('title')}")
            print()

        # Get version info
        version_response = requests.get(f"http://localhost:{port}/json/version")
        version_info = version_response.json()
        print(f"Browser WebSocket URL: {version_info.get('webSocketDebuggerUrl')}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(test_cdp())