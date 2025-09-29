#!/usr/bin/env python3
"""
Automatic Google Cloud setup for SheetBench - OAuth flow like rclone
No manual account creation needed!
"""

import json
import os
from pathlib import Path

SAMPLE_PREFIX = "764086051850-"
CLIENT_ID_ENV = "SHEETBENCH_CLIENT_ID"
CLIENT_SECRET_ENV = "SHEETBENCH_CLIENT_SECRET"
CREDS_FILE = Path("google_credentials.json")

def ensure_deps():
    """Import required Google auth libraries, installing if necessary."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
        from google.auth.transport.requests import Request  # noqa: F401
    except ImportError:
        print("❌ Missing required packages. Installing...")
        import subprocess
        import sys

        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "google-auth-oauthlib",
                "google-auth",
                "google-api-python-client"
            ])
        except subprocess.CalledProcessError as exc:
            print("🚫 Package installation failed. See pip output for details.")
            raise SystemExit(exc.returncode) from exc

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
            from google.auth.transport.requests import Request  # noqa: F401
        except ImportError as exc:  # pragma: no cover - defensive guard
            print("🚫 Dependencies unavailable even after installation.")
            raise SystemExit(1) from exc
        else:
            print("✅ Packages installed.")

def read_client_credentials():
    """Fetch OAuth client id/secret from environment and validate."""
    client_id = os.environ.get(CLIENT_ID_ENV, "").strip()
    client_secret = os.environ.get(CLIENT_SECRET_ENV, "").strip()

    if not client_id or not client_secret:
        raise SystemExit(
            "🚫 Set both SHEETBENCH_CLIENT_ID and SHEETBENCH_CLIENT_SECRET before running."
        )

    if client_id.startswith(SAMPLE_PREFIX):
        raise SystemExit(
            "🚫 You're using Google's sample client (764086051850). Create your own OAuth client."
        )

    return client_id, client_secret

def run_oauth_flow(client_id: str, client_secret: str):
    """Kick off the local OAuth consent flow and return credentials."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    scopes = [
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes)
    print("🌐 Launching browser for Google OAuth consent...")
    return flow.run_local_server(port=0, open_browser=True)

def write_credentials(creds) -> Path:
    """Persist authorized_user credentials to disk."""
    payload = {
        "type": "authorized_user",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    CREDS_FILE.write_text(json.dumps(payload, indent=2))
    return CREDS_FILE

def setup_google_auth():
    """Run the full OAuth setup and save credentials locally."""
    print("🔧 Setting up Google Cloud authentication for SheetBench...")
    print("📋 Expect a browser prompt. Use any Google account you want to grant access.")

    ensure_deps()
    client_id, client_secret = read_client_credentials()

    creds = run_oauth_flow(client_id, client_secret)
    print("✅ Authentication successful!")

    creds_path = write_credentials(creds)
    print(f"💾 Credentials saved to: {creds_path}")
    print("   These tokens already carry your project quota. No quota project headers needed.")
    return creds_path

if __name__ == "__main__":
    try:
        setup_google_auth()
    except SystemExit as exc:
        if exc.code not in (0, 1):
            raise
