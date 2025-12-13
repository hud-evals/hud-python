#!/usr/bin/env python3
"""
Fetch Gunter's Space Page data via Exa API.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv("/home/rs/projects/hud-python/.env")

EXA_API_KEY = os.getenv("EXA_API_KEY")
print(f"Using EXA_API_KEY: {EXA_API_KEY[:10]}...")


def exa_fetch(url: str, max_chars: int = 500000) -> str:
    """Fetch content from a URL using Exa API."""
    print(f"Fetching: {url}")
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json={
                "urls": [url],
                "text": {"maxCharacters": max_chars},
            },
        )
        response.raise_for_status()
        data = response.json()
    
    results = data.get("results", [])
    if not results:
        return ""
    
    content = results[0].get("text", "")
    print(f"  Got {len(content)} characters")
    return content


# Fetch 2023 launches
content_2023 = exa_fetch("https://space.skyrocket.de/doc_chr/lau2023.htm")
with open("/tmp/gunter_2023.txt", "w") as f:
    f.write(content_2023)
print(f"Saved to /tmp/gunter_2023.txt")

# Count lines with launch data
lines = content_2023.split('\n')
print(f"\nTotal lines: {len(lines)}")

# Look for table rows (missions)
mission_lines = [l for l in lines if '2023' in l and ('Falcon' in l or 'Long March' in l or 'Soyuz' in l or 'Atlas' in l or 'Ariane' in l)]
print(f"Mission-like lines found: {len(mission_lines)}")

print("\nFirst 30 lines:")
for i, line in enumerate(lines[:30]):
    print(f"{i}: {line[:150]}")

