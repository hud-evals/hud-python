#!/usr/bin/env python3
"""
Proper OAuth setup using YOUR OWN Google Cloud client (like rclone)
NO ADC behavior, NO quota project requirements
"""

import json
import os
from pathlib import Path

def setup_your_oauth():
    """Set up OAuth with your own Google Cloud client."""

    print("🔧 Setting up OAuth with YOUR OWN Google Cloud client...")
    print("📋 This avoids Google's sample client ADC behavior!")
    print()

    # Check for required environment variables
    client_id = os.environ.get("SHEETBENCH_CLIENT_ID")
    client_secret = os.environ.get("SHEETBENCH_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("❌ Missing required environment variables!")
        print()
        print("🎯 You need to create your own OAuth client:")
        print("   1. Go to https://console.cloud.google.com")
        print("   2. Create/select a project")
        print("   3. Enable Google Drive + Sheets APIs")
        print("   4. Go to APIs & Services > Credentials")
        print("   5. Create OAuth 2.0 Client ID (Desktop application)")
        print("   6. Copy the client ID and secret")
        print()
        print("Then run:")
        print("   export SHEETBENCH_CLIENT_ID=your-client-id.apps.googleusercontent.com")
        print("   export SHEETBENCH_CLIENT_SECRET=your-client-secret")
        print("   python setup_your_own_oauth.py")
        print()
        print("This is exactly what rclone does - uses its own client!")
        return

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("❌ Missing packages")
        import subprocess
        subprocess.check_call(["pip", "install", "google-auth-oauthlib"])
        from google_auth_oauthlib.flow import InstalledAppFlow

    # Use minimal scopes (like rclone)
    SCOPES = [
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    # Your own OAuth client configuration
    CLIENT_CONFIG = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"]
        }
    }

    print(f"🔑 Using YOUR OAuth client: {client_id[:30]}...")
    print("🌐 Starting OAuth flow...")
    print("   A browser window will open for Google login")

    # Run OAuth flow with YOUR client
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)

    # Save credentials
    creds_file = Path("google_credentials.json")
    creds_file.write_text(creds.to_json())

    print("✅ Authentication successful!")
    print(f"💾 Credentials saved to: {creds_file}")
    print()
    print("🎯 Benefits of using YOUR client:")
    print("   ✅ No ADC behavior")
    print("   ✅ No quota project requirements")
    print("   ✅ Bills to your project (where APIs are enabled)")
    print("   ✅ Works exactly like rclone!")
    print()
    print("Now test with: python test_google_sheets_direct.py")

if __name__ == "__main__":
    setup_your_oauth()