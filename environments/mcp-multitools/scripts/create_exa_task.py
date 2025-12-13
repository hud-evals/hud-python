#!/usr/bin/env python3
"""
Script to create Exa tasks with verified expected answers.
Wikipedia (3 eras) + Gunter's Space Page (simplified: ID, Date, Rocket only)
"""

import os
import json
import re
import httpx
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv("/home/rs/projects/hud-python/.env")

EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY not found")

print(f"Using EXA_API_KEY: {EXA_API_KEY[:10]}...")


def exa_fetch(url: str, max_chars: int = 500000) -> str:
    """Fetch content from a URL using Exa API."""
    print(f"Fetching: {url}")
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json={"urls": [url], "text": {"maxCharacters": max_chars}},
        )
        response.raise_for_status()
        data = response.json()
    
    results = data.get("results", [])
    content = results[0].get("text", "") if results else ""
    print(f"  Got {len(content)} characters")
    return content


def clean_text(text: str) -> str:
    """Remove markdown links and clean text."""
    # Handle markdown links with parentheses in URLs like (spacecraft)
    # Match [text](url) where url can contain nested parens
    while True:
        # Find markdown links and remove them, keeping the text
        match = re.search(r'\[([^\]]+)\]\(https?://[^)]*\)', text)
        if not match:
            break
        # Check if there's an unmatched paren in the URL
        url_start = match.start() + len(match.group(1)) + 2  # after [text](
        url_end = match.end() - 1
        url = text[url_start:url_end]
        # Count parens to find the real end
        if '(' in url:
            # Find balanced closing paren
            paren_count = 1
            real_end = match.end()
            for i, c in enumerate(text[match.end():]):
                if c == '(':
                    paren_count += 1
                elif c == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        real_end = match.end() + i + 1
                        break
            text = text[:match.start()] + match.group(1) + text[real_end:]
        else:
            text = text[:match.start()] + match.group(1) + text[match.end():]
    
    # Remove remaining markdown links (simpler ones)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove citation markers
    text = re.sub(r'\\\[\d+\\\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    
    # Clean up stray characters
    text = text.replace('_', '')
    text = re.sub(r'^\s*\)\s*', '', text)  # Remove leading )
    text = re.sub(r'\s*\(\s*$', '', text)  # Remove trailing (
    
    # Normalize slashes with spaces
    text = re.sub(r'\s*/\s*', '/', text)
    
    # Remove special Unicode characters like ⚀
    text = re.sub(r'[⚀⚁⚂⚃⚄⚅]', '', text)
    
    return text.strip()


def parse_wiki_missions(content: str) -> List[Dict]:
    """Parse Wikipedia lunar missions table."""
    missions = []
    seen = set()
    
    for line in content.split('\n'):
        if not line.startswith('|'):
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 7:
            continue
        if not parts[0].isdigit():
            continue
        
        year_match = re.search(r'(19\d{2}|20\d{2})', parts[2])
        if not year_match:
            continue
        
        num = parts[0]
        if num in seen:
            continue
        seen.add(num)
        
        missions.append({
            "name": clean_text(parts[1]),
            "year": year_match.group(1),
            "operator": clean_text(parts[3]),
            "outcome": clean_text(parts[-1])
        })
    
    return missions


def parse_gunter_launches_simple(content: str) -> List[Dict]:
    """Parse Gunter's launches - ONLY ID, Date, Rocket (skip mission names to avoid ambiguity)."""
    launches = []
    
    for line in content.split('\n'):
        # Look for lines with launch ID pattern
        if not re.search(r'\|\s*2023-\d{3}\s*\|', line) and not re.search(r'\|\s*2023-F\d{2}\s*\|', line):
            continue
        
        # Extract launch ID
        id_match = re.search(r'\|\s*(2023-\d{3}|2023-F\d{2})\s*\|', line)
        if not id_match:
            continue
        launch_id = id_match.group(1)
        
        # Extract date (DD.MM.YYYY format)
        date_match = re.search(r'\|\s*(\d{2}\.\d{2}\.2023)\s*\|', line)
        if not date_match:
            continue
        date = date_match.group(1)
        
        # Extract rocket - in format [Rocket Name](doc_lau... url)
        rocket_match = re.search(r'\[([^\]]+)\]\(https://space\.skyrocket\.de/doc_lau[^)]+\)', line)
        if rocket_match:
            rocket = rocket_match.group(1)
        else:
            continue  # Skip if no rocket found
        
        launches.append({
            "id": launch_id,
            "date": date,
            "rocket": rocket
        })
    
    return launches


def format_wiki_answer(missions: List[Dict]) -> str:
    """Format Wikipedia missions."""
    entries = [f"{m['name']}: {m['year']}, {m['operator']}, {m['outcome']}" for m in missions]
    return "; ".join(entries)


def format_gunter_answer(launches: List[Dict]) -> str:
    """Format Gunter launches - simple format: ID: Date, Rocket"""
    entries = [f"{l['id']}: {l['date']}, {l['rocket']}" for l in launches]
    return "; ".join(entries)


def main():
    print("="*70)
    print("CREATING TRIPLE-SOURCE EXA TASK")
    print("="*70)
    
    # 1. Fetch/load Wikipedia lunar missions
    wiki_file = "/tmp/wiki_lunar.txt"
    if os.path.exists(wiki_file):
        print(f"\n--- Using cached Wikipedia data ---")
        with open(wiki_file, 'r') as f:
            wiki_content = f.read()
    else:
        print("\n--- Fetching Wikipedia 'List of missions to the Moon' ---")
        wiki_content = exa_fetch("https://en.wikipedia.org/wiki/List_of_missions_to_the_Moon", 500000)
        with open(wiki_file, "w") as f:
            f.write(wiki_content)
    
    wiki_missions = parse_wiki_missions(wiki_content)
    print(f"Parsed {len(wiki_missions)} total Wikipedia missions")
    
    # Group by year
    by_year = defaultdict(list)
    for m in wiki_missions:
        by_year[m['year']].append(m)
    
    # Show year distribution
    print("\nMissions per year:")
    for year in sorted(by_year.keys()):
        if len(by_year[year]) > 0:
            print(f"  {year}: {len(by_year[year])} missions")
    
    # Create 3 eras
    era_1960s = []
    for y in ['1966', '1967', '1968', '1969']:
        era_1960s.extend(by_year.get(y, []))
    
    era_1990s = []
    for y in ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999']:
        era_1990s.extend(by_year.get(y, []))
    
    era_2020s = []
    for y in ['2019', '2020', '2021', '2022', '2023', '2024', '2025']:
        era_2020s.extend(by_year.get(y, []))
    
    print(f"\n=== WIKIPEDIA ERAS ===")
    print(f"era_1960s (1966-1969): {len(era_1960s)} missions")
    print(f"era_1990s (1990-1999): {len(era_1990s)} missions")
    print(f"era_2020s (2019-2025): {len(era_2020s)} missions")
    
    # 2. Fetch/load Gunter's 2023 launches
    gunter_file = "/tmp/gunter_2023.txt"
    if os.path.exists(gunter_file):
        print(f"\n--- Using cached Gunter data ---")
        with open(gunter_file, 'r') as f:
            gunter_content = f.read()
    else:
        print("\n--- Fetching Gunter's Space Page 2023 ---")
        gunter_content = exa_fetch("https://space.skyrocket.de/doc_chr/lau2023.htm", 500000)
        with open(gunter_file, "w") as f:
            f.write(gunter_content)
    
    gunter_launches = parse_gunter_launches_simple(gunter_content)
    print(f"Parsed {len(gunter_launches)} Gunter launches (ID, Date, Rocket only)")
    
    # Take first 100 launches for manageable size
    gunter_subset = gunter_launches[:100]
    print(f"Using first {len(gunter_subset)} launches")
    
    # Show samples
    print("\nSample Gunter launches:")
    for l in gunter_subset[:5]:
        print(f"  {l['id']}: {l['date']}, {l['rocket']}")
    
    # Generate answers
    era_1960s_answer = format_wiki_answer(era_1960s)
    era_1990s_answer = format_wiki_answer(era_1990s)
    era_2020s_answer = format_wiki_answer(era_2020s)
    gunter_answer = format_gunter_answer(gunter_subset)
    
    print(f"\n=== ANSWER STATISTICS ===")
    print(f"era_1960s: {len(era_1960s_answer)} chars ({len(era_1960s)} entries)")
    print(f"era_1990s: {len(era_1990s_answer)} chars ({len(era_1990s)} entries)")
    print(f"era_2020s: {len(era_2020s_answer)} chars ({len(era_2020s)} entries)")
    print(f"launches_2023: {len(gunter_answer)} chars ({len(gunter_subset)} entries)")
    total = len(era_1960s_answer) + len(era_1990s_answer) + len(era_2020s_answer) + len(gunter_answer)
    print(f"TOTAL: {total} chars")
    
    # Create task with VERY explicit formatting
    task = {
        "id": "task-exa-space-multi-source",
        "prompt": f"""Go to Wikipedia page 'List of missions to the Moon' and extract lunar mission data.

PART 1 - WIKIPEDIA (3 eras):
Extract ALL lunar missions from these year ranges:
- Years 1966, 1967, 1968, 1969 → store as 'era_1960s' (expect ~{len(era_1960s)} entries)
- Years 1990-1999 → store as 'era_1990s' (expect ~{len(era_1990s)} entries)
- Years 2019-2025 → store as 'era_2020s' (expect ~{len(era_2020s)} entries)

Wikipedia format - copy EXACTLY from table columns:
{{Mission}}: {{Year}}, {{Operator}}, {{Outcome}}

Examples:
Luna 9: 1966, Lavochkin, Success
Apollo 11: 1969, NASA, Success
Chandrayaan-3: 2023, ISRO, Success

PART 2 - EXTERNAL LAUNCH DATABASE:
In Wikipedia's 'External links' or 'See also', find a comprehensive space launch chronology (like Gunter's Space Page). Navigate to their 2023 orbital launches page.

Extract the FIRST {len(gunter_subset)} launches. For each launch extract ONLY:
- Launch ID (e.g., "2023-001" or "2023-F03" for failures)
- Date (keep DD.MM.YYYY format exactly as shown)
- Rocket name (full name as shown, including version info)

DO NOT include satellite/mission names - they have complex formatting.

Store as 'launches_2023' with format:
{{LaunchID}}: {{Date}}, {{Rocket}}

Examples:
2023-001: 03.01.2023, Falcon-9 v1.2 (Block 5)
2023-002: 08.01.2023, CZ-7A
2023-F01: 09.01.2023, LauncherOne

ALL answers must be semicolon-separated, ALL entries on ONE line.

Say 'Task completed.' when done.""",
        "mcp_config": {
            "local": {
                "command": "docker",
                "args": ["run", "--rm", "-i", "--env-file", "/home/rs/projects/hud-python/.env", "mcp-multitools:latest"]
            }
        },
        "setup_tool": {"name": "setup", "arguments": {}},
        "evaluate_tool": {
            "name": "evaluate",
            "arguments": {
                "exact_values": {
                    "era_1960s": era_1960s_answer,
                    "era_1990s": era_1990s_answer,
                    "era_2020s": era_2020s_answer,
                    "launches_2023": gunter_answer
                }
            }
        },
        "agent_config": {
            "allowed_tools": ["search", "fetch", "scratchpad_write", "scratchpad_read"]
        }
    }
    
    # Save task
    output_file = "/home/rs/projects/hud-python/environments/mcp-multitools/task_jsons/rs_exa_hard.json"
    with open(output_file, 'w') as f:
        json.dump([task], f, indent=2)
    print(f"\nTask saved to: {output_file}")
    
    # Show answer previews
    print("\n" + "="*70)
    print("FULL ANSWERS (for verification)")
    print("="*70)
    
    print(f"\n--- era_1960s ({len(era_1960s)} entries) ---")
    print(era_1960s_answer)
    
    print(f"\n--- era_1990s ({len(era_1990s)} entries) ---")
    print(era_1990s_answer if era_1990s_answer else "(no missions in 1990-1999)")
    
    print(f"\n--- era_2020s ({len(era_2020s)} entries) ---")
    print(era_2020s_answer)
    
    print(f"\n--- launches_2023 ({len(gunter_subset)} entries) ---")
    print(gunter_answer[:1500] + "..." if len(gunter_answer) > 1500 else gunter_answer)


if __name__ == "__main__":
    main()
