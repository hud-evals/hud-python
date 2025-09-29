#!/usr/bin/env python3
"""Convert SheetBench-50.json to use local Docker with kernel provider"""

import json
import os

def convert_mcp_config():
    # Read the original file
    with open('SheetBench-50.json', 'r') as f:
        data = json.load(f)

    # Read the OAuth credentials we just generated
    try:
        with open('google_credentials.json', 'r') as f:
            gcp_creds = json.load(f)
        print("✅ Found Google OAuth credentials")
    except FileNotFoundError:
        print("❌ google_credentials.json not found. Run: python setup_google_auth.py")
        return

    # Convert to base64 to avoid Docker escaping issues
    import base64
    gcp_creds_str = json.dumps(gcp_creds)
    gcp_creds_b64 = base64.b64encode(gcp_creds_str.encode()).decode()

    # Docker-based mcp_config with kernel provider + real GCP credentials
    docker_config = {
        "browser": {
            "command": "docker",
            "args": [
                "run", "--rm", "-i", "-p", "0:8080",
                "-e", "BROWSER_PROVIDER=kernel",
                "-v", f"{os.getcwd()}/google_credentials.json:/app/google_credentials.json",
                "-e", "GCP_CREDENTIALS_FILE=/app/google_credentials.json",
                "hudevals/hud-remote-browser:final-working"
            ]
        }
    }

    # Replace mcp_config in each task
    for task in data:
        if 'mcp_config' in task:
            task['mcp_config'] = docker_config

    # Write back to file
    with open('SheetBench-50-local.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Converted {len(data)} tasks to use local kernel provider")
    print("📁 Saved as: SheetBench-50-local.json")

if __name__ == "__main__":
    convert_mcp_config()
