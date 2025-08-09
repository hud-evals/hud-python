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
from hud.tools.types import SetupResult
from . import setup

logger = logging.getLogger(__name__)


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
        return SetupResult(content="Missing required parameter: file_url", isError=True)

    logger.info(f"Downloading Excel file from: {file_url}")

    try:
        # Download the Excel file
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            file_bytes = response.content

            logger.info(f"Downloaded {len(file_bytes)} bytes")

            # Create Google Drive service
            scopes = ["https://www.googleapis.com/auth/drive"]
            gcp_creds = get_gcp_credentials()
            credentials = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
            drive_service = build("drive", "v3", credentials=credentials)

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
        playwright_tool = setup.env
        if playwright_tool and hasattr(playwright_tool, "page") and playwright_tool.page:
            page = playwright_tool.page
            logger.info(f"Navigating to sheet: {sheet_url}")
            await page.goto(sheet_url, wait_until="domcontentloaded", timeout=15000)

            # Wait for sheet to load
            try:
                await page.wait_for_selector(".grid-container", timeout=10000)
                logger.info("Sheet loaded successfully")
            except:
                logger.warning("Timeout waiting for sheet to fully load")

        sheet_info = {"sheet_id": sheet_id, "sheet_url": sheet_url, "sheet_name": sheet_name}

        return SetupResult(
            content=f"Created and navigated to Google Sheet: {sheet_url}", info=sheet_info
        )

    except httpx.HTTPError as e:
        logger.error(f"HTTP error downloading file: {str(e)}")
        return SetupResult(content=f"Failed to download Excel file: {str(e)}", isError=True)
    except Exception as e:
        logger.error(f"Error in sheets_from_xlsx: {str(e)}")
        return SetupResult(content=f"Failed to create sheet: {str(e)}", isError=True)


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
        return SetupResult(content="Missing required parameter: base64_bytes", isError=True)

    file_bytes_b64 = base64_bytes

    logger.info(f"Creating sheet from bytes, name: {sheet_name}")

    try:
        # Decode base64 bytes
        file_bytes = base64.b64decode(file_bytes_b64)
        logger.info(f"Decoded {len(file_bytes)} bytes")

        # Create Google Drive service
        scopes = ["https://www.googleapis.com/auth/drive"]
        gcp_creds = get_gcp_credentials()
        credentials = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
        drive_service = build("drive", "v3", credentials=credentials)

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
        playwright_tool = setup.env
        if playwright_tool and hasattr(playwright_tool, "page") and playwright_tool.page:
            page = playwright_tool.page
            logger.info(f"Navigating to sheet: {sheet_url}")
            await page.goto(sheet_url, wait_until="domcontentloaded", timeout=15000)

            # Wait for sheet to load
            try:
                await page.wait_for_selector(".grid-container", timeout=10000)
                logger.info("Sheet loaded successfully")
            except:
                logger.warning("Timeout waiting for sheet to fully load")

        sheet_info = {"sheet_id": sheet_id, "sheet_url": sheet_url, "sheet_name": sheet_name}

        return SetupResult(
            content=f"Created and navigated to Google Sheet: {sheet_url}", info=sheet_info
        )

    except Exception as e:
        logger.error(f"Error in sheets_from_bytes: {str(e)}")
        return SetupResult(content=f"Failed to create sheet: {str(e)}", isError=True)
