import asyncio
import logging
import os
from typing import Any, Callable, Dict
from urllib.parse import urlparse

import httpx
from edgar import Company, Filing
import asyncio
import logging
import os
import socket
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


async def execute_fetch(url: str, exa_api_key: str, max_length: int = 2500) -> Dict[str, Any]:
    """Execute the actual Exa contents API call."""
    contents_url = "https://api.exa.ai/contents"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            contents_url,
            headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
            json={
                "urls": [url],
                "text": {"maxCharacters": max_length, "includeHtmlTags": False},
                "highlights": {"numSentences": 5, "highlightsPerUrl": 3},
                "summary": {"query": "main takeaways"},
                "livecrawl": "fallback",
            },
        )
        response.raise_for_status()
        return response.json()


async def execute_search(query: str, exa_api_key: str, max_results: int = 1) -> Dict[str, Any]:
    """Execute the actual Exa search API call."""
    search_url = "https://api.exa.ai/search"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            search_url,
            headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": max_results,
                "type": "keyword",
                "userLocation": "us",
                "contents": {"text": {"maxCharacters": 1000}},
            },
        )
        response.raise_for_status()
        return response.json()
