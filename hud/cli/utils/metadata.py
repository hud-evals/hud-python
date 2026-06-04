"""Registry metadata helpers for the HUD CLI."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests
import yaml

from hud.settings import settings

from .api import hud_headers


def fetch_lock_from_registry(reference: str) -> dict[str, Any] | None:
    """Fetch lock file from HUD registry."""
    try:
        # Reference should be org/name:tag format
        # If no tag specified, append :latest
        if "/" in reference and ":" not in reference:
            reference = f"{reference}:latest"

        # URL-encode the path segments to handle special characters in tags
        url_safe_path = "/".join(quote(part, safe="") for part in reference.split("/"))
        registry_url = f"{settings.hud_api_url.rstrip('/')}/registry/envs/{url_safe_path}"

        headers = hud_headers()

        response = requests.get(registry_url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Parse the lock YAML from the response
            if "lock" in data:
                return yaml.safe_load(data["lock"])
            elif "lock_data" in data:
                return data["lock_data"]
            else:
                # Try to treat the whole response as lock data
                return data

        return None
    except Exception:
        return None
