#!/usr/bin/env python3
"""Convert SheetBench-50.json to use Docker instead of remote mcp.hud.so"""

import json

def convert_mcp_config():
    # Read the original file
    with open('SheetBench-50.json', 'r') as f:
        data = json.load(f)

    # Docker-based mcp_config - use the correct SheetBench image
    docker_config = {
        "browser": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "-p", "8080:8080", "hudevals/hud-remote-browser:0.1.1"]
        }
    }

    # Replace mcp_config in each task
    for task in data:
        if 'mcp_config' in task:
            task['mcp_config'] = docker_config

    # Write back to file
    with open('SheetBench-50-docker.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Converted {len(data)} tasks to use Docker")
    print("📁 Saved as: SheetBench-50-docker.json")

if __name__ == "__main__":
    convert_mcp_config()