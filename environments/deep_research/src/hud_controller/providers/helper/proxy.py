# Copied simplified global proxy helper
import os, random, asyncio, logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


# ----------------------- provider helpers ---------------------------
async def _decodo_proxy() -> Optional[Dict[str, Any]]:
    user = os.getenv("DECODO_USERNAME")
    pwd = os.getenv("DECODO_PASSWORD")
    if not user or not pwd:
        return None
    host = os.getenv("DECODO_HOST", "us.decodo.com")
    rotating = os.getenv("DECODO_ROTATING", "true").lower() == "true"
    if rotating:
        port = 10000
        logger.info("Using Decodo rotating proxy (port 10000)")
        return {
            "type": "custom",
            "server": f"{host}:{port}",
            "username": user,
            "password": pwd,
            "active": True,
        }
    logger.info("Searching Decodo ports 10001-11000 …")
    tried = set()
    for _ in range(5):
        port = random.randint(10001, 11000)
        while port in tried:
            port = random.randint(10001, 11000)
        tried.add(port)
        proxy_url = f"http://{user}:{pwd}@{host}:{port}"
        try:
            async with httpx.AsyncClient(proxy=proxy_url, timeout=5.0) as client:
                if (await client.get("http://httpbin.org/ip")).status_code == 200:
                    logger.info("Decodo port %s works", port)
                    return {
                        "type": "custom",
                        "server": f"{host}:{port}",
                        "username": user,
                        "password": pwd,
                        "active": True,
                    }
        except Exception:
            continue
    logger.warning("No working Decodo port found")
    return None


def _standard_proxy() -> Optional[Dict[str, Any]]:
    server = os.getenv("PROXY_SERVER")
    if not server:
        return None
    return {
        "type": "custom",
        "server": server,
        "username": os.getenv("PROXY_USERNAME"),
        "password": os.getenv("PROXY_PASSWORD"),
        "active": True,
    }


# ----------------------- public API ---------------------------------
async def get_proxy_config() -> Optional[Dict[str, Any]]:
    provider = os.getenv("PROXY_PROVIDER", "auto").lower()

    if provider == "none":
        logger.info("Proxy explicitly disabled")
        return None

    if provider == "decodo":
        config = await _decodo_proxy()
        if not config:
            logger.warning("Decodo proxy requested but credentials not found")
        return config

    if provider == "standard":
        config = _standard_proxy()
        if not config:
            logger.warning("Standard proxy requested but PROXY_SERVER not set")
        return config

    # auto or unknown - let browser use its default
    return None
