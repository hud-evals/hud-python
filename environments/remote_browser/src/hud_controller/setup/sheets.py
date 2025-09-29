"""Setup functions for Google Sheets operations."""

import base64
import io
import json
import logging
import os
import httpx
from typing import Any, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.service_account import Credentials
from fastmcp import Context
from mcp.types import TextContent
from . import setup

logger = logging.getLogger(__name__)


async def navigate_to_google_sheet(page, sheet_url: str, max_attempts: int = 3):
    """Navigate to a Google Sheet with retry logic and loading issue handling.

    Args:
        page: Playwright page object
        sheet_url: URL of the Google Sheet
        max_attempts: Maximum number of navigation attempts

    Returns:
        bool: True if navigation was successful, False otherwise
    """
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                logger.info(f"Retrying navigation (attempt {attempt + 1}/{max_attempts})")

            # Navigate to the sheet
            await page.goto(sheet_url, wait_until="load", timeout=45000)

            # Wait for sheet to load
            try:
                await page.wait_for_selector(".grid-container", timeout=20000)
                logger.info("Sheet loaded successfully")

                # Check for loading issue popup
                await page.wait_for_timeout(2000)  # Give popup time to appear

                if await page.locator('text="Loading issue"').is_visible(timeout=1000):
                    logger.warning("Loading issue popup detected, reloading page")
                    await page.reload(wait_until="networkidle", timeout=30000)

                    # Wait for sheet to load again after reload
                    await page.wait_for_selector(".grid-container", timeout=20000)
                    logger.info("Sheet reloaded successfully")

                return True  # Success

            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning("Timeout waiting for sheet to load, will retry with refresh")
                    await page.reload(timeout=30000)
                else:
                    logger.warning("Timeout waiting for sheet to fully load after all attempts")
                    return False

        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Navigation failed: {str(e)}, retrying...")
                await page.wait_for_timeout(2000)  # Wait before retry
            else:
                logger.error(f"Navigation failed after all attempts: {str(e)}")
                raise

    return False


def get_gcp_credentials() -> Dict[str, str]:
    """Get GCP credentials from environment variable.

    Expects one of:
    1. GCP_CREDENTIALS_JSON - A JSON string or base64-encoded JSON
    2. GCP_CREDENTIALS_BASE64 - Base64 encoded JSON
    3. GCP_CREDENTIALS_FILE - Path to a JSON file
    4. Individual environment variables for each field (GCP_PROJECT_ID, etc.)

    Returns:
        Dict containing GCP service account credentials
    """
    # First try to get from JSON env var
    creds_json = os.getenv("GCP_CREDENTIALS_JSON")
    if creds_json:
        # Check if it's base64 encoded (doesn't start with { and no spaces)
        if not creds_json.startswith("{") and " " not in creds_json[:100]:
            try:
                import base64

                # Decode base64 first
                creds_json = base64.b64decode(creds_json).decode("utf-8")
                logger.info("Detected and decoded base64-encoded GCP credentials")
            except Exception as e:
                logger.debug(f"Not base64 encoded: {e}")

        # Parse as JSON
        try:
            return json.loads(creds_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GCP_CREDENTIALS_JSON: {e}")
            raise ValueError(
                "Invalid GCP_CREDENTIALS_JSON format. "
                "Use either: 1) Valid JSON, 2) Base64-encoded JSON, "
                "3) GCP_CREDENTIALS_BASE64 env var, or 4) Individual env vars"
            )

    # Try base64 encoded credentials
    creds_base64 = os.getenv("GCP_CREDENTIALS_BASE64")
    if creds_base64:
        try:
            import base64

            decoded = base64.b64decode(creds_base64).decode("utf-8")
            return json.loads(decoded)
        except Exception as e:
            logger.error("Failed to decode GCP_CREDENTIALS_BASE64: %s", e)
            raise ValueError(f"Invalid GCP_CREDENTIALS_BASE64: {e}")

    # Try loading from file
    creds_file = os.getenv("GCP_CREDENTIALS_FILE")
    if creds_file:
        try:
            with open(creds_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load GCP_CREDENTIALS_FILE from %s: %s", creds_file, e)
            raise ValueError(f"Could not load credentials from file {creds_file}: {e}")

    # Otherwise try to build from individual env vars
    required_fields = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    ]

    credentials = {}
    for field in required_fields:
        env_key = f"GCP_{field.upper()}"
        value = os.getenv(env_key)
        if not value:
            raise ValueError(f"Missing required GCP credential field: {env_key}")
        credentials[field] = value

    # Add universe_domain with default
    credentials["universe_domain"] = os.getenv("GCP_UNIVERSE_DOMAIN", "googleapis.com")

    return credentials


@setup.tool("sheets_from_xlsx")
async def sheets_from_xlsx(
    ctx: Context, file_url: Optional[str] = None, sheet_name: str = "Worksheet"
):
    """Create a Google Sheet from an Excel file URL.

    Args:
        file_url: URL of the Excel file to convert
        sheet_name: Name for the new Google Sheet (default: "Worksheet")

    Returns:
        Status dictionary with sheet information
    """
    logger.info("Starting sheets_from_xlsx setup")

    # Validate parameters
    if not file_url:
        logger.error("Missing required file_url parameter")
        return TextContent(text="Missing required parameter: file_url", type="text")

    logger.info(f"Downloading Excel file from: {file_url}")

    try:
        # Download the Excel file
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            file_bytes = response.content

            logger.info(f"Downloaded {len(file_bytes)} bytes")

            # Create Google Drive service
            # Use minimal scopes so it behaves like rclone and avoids verification pain
            scopes = [
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/spreadsheets",
            ]
            gcp_creds = get_gcp_credentials()

            # Handle both service account and OAuth user credentials
            if gcp_creds.get("type") == "service_account":
                credentials = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
            else:
                # Handle OAuth user credentials
                from google.oauth2.credentials import Credentials as UserCredentials
                # 🚫 Do NOT force a user quota project (causes 403 unless API is enabled there)
                gcp_creds.pop("quota_project_id", None)
                # Also sanitize any env override just in case
                os.environ.pop("GOOGLE_CLOUD_QUOTA_PROJECT", None)
                credentials = UserCredentials(
                    token=None,
                    refresh_token=gcp_creds.get("refresh_token"),
                    token_uri=gcp_creds.get("token_uri"),
                    client_id=gcp_creds.get("client_id"),
                    client_secret=gcp_creds.get("client_secret"),
                    scopes=scopes,
                )

            # Use credentials directly without quota project (like rclone)
            drive_service = build("drive", "v3", credentials=credentials)
            logger.info("Using OAuth client project for quota (like rclone)")

            # Upload to Google Drive with conversion
            file_metadata = {
                "name": sheet_name,
                "mimeType": "application/vnd.google-apps.spreadsheet",
            }

            media = MediaIoBaseUpload(
                io.BytesIO(file_bytes),
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                resumable=True,
            )

            logger.info("Uploading to Google Drive with conversion to Sheets")
            drive_file = (
                drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id,webViewLink")
                .execute()
            )

            sheet_id = drive_file.get("id")
            sheet_url = drive_file.get("webViewLink")

            logger.info(f"Created Google Sheet: {sheet_id}")

            # Set sharing permissions
            permission = {"type": "anyone", "role": "writer", "allowFileDiscovery": False}

            drive_service.permissions().create(
                fileId=sheet_id, body=permission, fields="id"
            ).execute()

        logger.info("Set sharing permissions")

        # Navigate to the sheet
        persistent_ctx = setup.env
        playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
        if playwright_tool and hasattr(playwright_tool, "page") and playwright_tool.page:
            page = playwright_tool.page
            logger.info(f"Navigating to sheet: {sheet_url}")

            # Use the unified navigation function
            await navigate_to_google_sheet(page, sheet_url, max_attempts=3)
        else:
            logger.warning("No playwright tool available for navigation")

        sheet_info = {"sheet_id": sheet_id, "sheet_url": sheet_url, "sheet_name": sheet_name}

        return TextContent(
            text=f"Created and navigated to Google Sheet '{sheet_name}': {sheet_url}", type="text"
        )

    except httpx.HTTPError as e:
        logger.error(f"HTTP error downloading file: {str(e)}")
        return TextContent(text=f"Failed to download Excel file: {str(e)}", type="text")
    except Exception as e:
        logger.error(f"Error in sheets_from_xlsx: {str(e)}")
        return TextContent(text=f"Failed to create sheet: {str(e)}", type="text")


@setup.tool("sheets_from_bytes")
async def sheets_from_bytes(
    ctx: Context, base64_bytes: Optional[str] = None, sheet_name: str = "Worksheet"
):
    """Create a Google Sheet from base64 encoded Excel bytes.

    Args:
        base64_bytes: Base64 encoded Excel file bytes
        sheet_name: Name for the new Google Sheet (default: "Worksheet")

    Returns:
        Status dictionary with sheet information
    """
    logger.info("Starting sheets_from_bytes setup")

    # Validate parameters
    if not base64_bytes:
        logger.error("Missing required base64_bytes parameter")
        return TextContent(text="Missing required parameter: base64_bytes", type="text")

    file_bytes_b64 = base64_bytes

    logger.info(f"Creating sheet from bytes, name: {sheet_name}")

    try:
        # Decode base64 bytes
        file_bytes = base64.b64decode(file_bytes_b64)
        logger.info(f"Decoded {len(file_bytes)} bytes")

        # Create Google Drive service
        # Use minimal scopes so it behaves like rclone and avoids verification pain
        scopes = [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/spreadsheets",
        ]
        gcp_creds = get_gcp_credentials()

        # Handle both service account and OAuth user credentials
        if gcp_creds.get("type") == "service_account":
            credentials = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
        else:
            # Handle OAuth user credentials
            from google.oauth2.credentials import Credentials as UserCredentials
            # 🚫 Do NOT force a user quota project (causes 403 unless API is enabled there)
            gcp_creds.pop("quota_project_id", None)
            # Also sanitize any env override just in case
            os.environ.pop("GOOGLE_CLOUD_QUOTA_PROJECT", None)
            credentials = UserCredentials(
                token=None,
                refresh_token=gcp_creds.get("refresh_token"),
                token_uri=gcp_creds.get("token_uri"),
                client_id=gcp_creds.get("client_id"),
                client_secret=gcp_creds.get("client_secret"),
                scopes=scopes,
            )

        # Use credentials directly without quota project (like rclone)
        drive_service = build("drive", "v3", credentials=credentials)
        logger.info("Using OAuth client project for quota (like rclone)")

        # Upload to Google Drive with conversion
        file_metadata = {
            "name": sheet_name,
            "mimeType": "application/vnd.google-apps.spreadsheet",
        }

        media = MediaIoBaseUpload(
            io.BytesIO(file_bytes),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            resumable=True,
        )

        logger.info("Uploading to Google Drive with conversion to Sheets")
        drive_file = (
            drive_service.files()
            .create(body=file_metadata, media_body=media, fields="id,webViewLink")
            .execute()
        )

        sheet_id = drive_file.get("id")
        sheet_url = drive_file.get("webViewLink")

        logger.info(f"Created Google Sheet: {sheet_id}")

        # Set sharing permissions
        permission = {"type": "anyone", "role": "writer", "allowFileDiscovery": False}

        drive_service.permissions().create(fileId=sheet_id, body=permission, fields="id").execute()

        logger.info("Set sharing permissions")

        # Navigate to the sheet
        persistent_ctx = setup.env
        playwright_tool = getattr(persistent_ctx, "playwright_tool", None)
        if playwright_tool and hasattr(playwright_tool, "page") and playwright_tool.page:
            page = playwright_tool.page
            logger.info(f"Navigating to sheet: {sheet_url}")

            # Use the unified navigation function
            await navigate_to_google_sheet(page, sheet_url, max_attempts=2)
        else:
            logger.warning("No playwright tool available for navigation")

        sheet_info = {"sheet_id": sheet_id, "sheet_url": sheet_url, "sheet_name": sheet_name}

        return TextContent(
            text=f"Created and navigated to Google Sheet '{sheet_name}': {sheet_url}", type="text"
        )

    except Exception as e:
        logger.error(f"Error in sheets_from_bytes: {str(e)}")
        return TextContent(text=f"Failed to create sheet: {str(e)}", type="text")
