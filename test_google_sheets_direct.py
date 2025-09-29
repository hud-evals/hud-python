#!/usr/bin/env python3
"""
Direct test of Google Sheets API with our OAuth credentials
NO DOCKER - just test if our creds work
"""

import json
import webbrowser
import tempfile
import os
from pathlib import Path

def test_google_sheets():
    """Test Google Sheets API directly with our OAuth credentials."""

    print("🧪 Testing Google Sheets API with our OAuth credentials...")

    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google_auth_httplib2 import AuthorizedHttp
        import httplib2
    except ImportError:
        print("❌ Missing packages")
        import subprocess
        subprocess.check_call(["pip", "install", "google-auth", "google-api-python-client", "google-auth-httplib2"])
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google_auth_httplib2 import AuthorizedHttp
        import httplib2

    # Load our OAuth credentials
    creds_file = Path("google_credentials.json")
    if not creds_file.exists():
        print("❌ No google_credentials.json found. Run: python setup_google_auth.py")
        return

    print("📄 Loading OAuth credentials...")
    with open(creds_file, 'r') as f:
        creds_data = json.load(f)

    print(f"🔑 Credential type: {creds_data.get('type')}")
    print(f"🔗 Client ID: {creds_data.get('client_id', 'N/A')[:50]}...")

    # Guard against Google's sample client (causes ADC behavior)
    if creds_data.get("client_id","").startswith("764086051850-"):
        print("🚫 ERROR: You're using Google's sample client!")
        print("   This client triggers ADC behavior and requires quota projects.")
        print("   You need to create your own OAuth client in Google Cloud Console:")
        print("   1. Go to https://console.cloud.google.com")
        print("   2. Create/select a project")
        print("   3. Enable Google Drive + Sheets APIs")
        print("   4. Create an OAuth 2.0 Client ID (Desktop application)")
        print("   5. Export SHEETBENCH_CLIENT_ID and SHEETBENCH_CLIENT_SECRET before running setup_google_auth.py")
        print()
        print("   This is what rclone does - uses its own client, not Google's sample.")
        return False

    # Make absolutely sure ADC can't hijack the call
    for k in ("GOOGLE_APPLICATION_CREDENTIALS",
              "GOOGLE_CLOUD_PROJECT",
              "GOOGLE_CLOUD_QUOTA_PROJECT"):
        os.environ.pop(k, None)

    # Create credentials object from our authorized_user file
    scopes = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/spreadsheets'
    ]

    credentials = Credentials.from_authorized_user_info(creds_data, scopes=scopes)
    # Pre-refresh so we know we're using *these* creds
    if not credentials.valid:
        credentials.refresh(Request())

    print("🚀 Building Google Drive service...")

    try:
        # Build with an explicitly authorized HTTP so ADC can't be used anywhere
        authed_http = AuthorizedHttp(credentials, http=httplib2.Http())
        drive_service = build("drive", "v3", http=authed_http, cache_discovery=False)
        print("✅ Google Drive service created successfully!")

        # Create a simple test CSV file
        test_data = """Date,Price,Volume
2024-01-01,100.50,1000
2024-01-02,101.25,1200
2024-01-03,99.75,800
2024-01-04,102.00,1500
2024-01-05,103.25,1100"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_data)
            csv_path = f.name

        print(f"📊 Created test CSV: {csv_path}")

        # Upload and convert to Google Sheets
        from googleapiclient.http import MediaFileUpload

        file_metadata = {
            "name": "SheetBench Test Data",
            "mimeType": "application/vnd.google-apps.spreadsheet"
        }

        # Small file: a simple (non-resumable) upload avoids extra moving parts
        media = MediaFileUpload(csv_path, mimetype="text/csv", resumable=False)

        print("📤 Uploading to Google Drive and converting to Sheets...")

        request = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id,name,webViewLink"
        )
        created_file = request.execute(http=authed_http)

        sheet_id = created_file.get("id")
        sheet_url = created_file.get("webViewLink")
        sheet_name = created_file.get("name")

        print(f"🎉 SUCCESS! Created Google Sheet!")
        print(f"   📋 Name: {sheet_name}")
        print(f"   🆔 ID: {sheet_id}")
        print(f"   🔗 URL: {sheet_url}")

        # Set public permissions
        permission = {"type": "anyone", "role": "reader", "allowFileDiscovery": False}
        drive_service.permissions().create(
            fileId=sheet_id, body=permission, fields="id"
        ).execute(http=authed_http)
        print("✅ Set public read permissions")

        # Open in browser
        print("🌐 Opening Google Sheet in browser...")
        webbrowser.open(sheet_url)

        # Clean up
        os.unlink(csv_path)

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'resp'):
            print(f"   HTTP Status: {e.resp.status}")
            print(f"   HTTP Reason: {e.resp.reason}")
        return False

if __name__ == "__main__":
    success = test_google_sheets()
    if success:
        print("\n🎯 GOOGLE SHEETS API WORKS! Your credentials are good!")
        print("   The Docker issue is somewhere else...")
    else:
        print("\n💥 GOOGLE SHEETS API FAILED! Credentials/scopes are broken.")
        print("   Need to fix OAuth setup...")