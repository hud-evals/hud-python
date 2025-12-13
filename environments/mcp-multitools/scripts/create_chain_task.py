#!/usr/bin/env python3
"""
Create a reference-chaining task that requires navigating through multiple Wikipedia pages.
"""

import os
import json
import re
import httpx
from dotenv import load_dotenv

load_dotenv("/home/rs/projects/hud-python/.env")

EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY not found")


def exa_fetch(url: str, max_chars: int = 80000) -> str:
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
    content = data.get("results", [{}])[0].get("text", "")
    print(f"  Got {len(content)} characters")
    return content


def clean_text(text: str) -> str:
    """Remove markdown links and clean text."""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[#[^\]]+\]', '', text)
    text = re.sub(r'&#91;[^&]+&#93;', '', text)
    text = text.replace('*', '').replace('_', '')
    return text.strip()


def main():
    print("="*70)
    print("CREATING REFERENCE-CHAIN TASK")
    print("="*70)
    
    # ===== PAGE 1: List of highest-grossing films =====
    print("\n--- PAGE 1: List of highest-grossing films ---")
    page1 = exa_fetch("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
    
    # Extract exact gross of #1 film (Avatar)
    gross_match = re.search(r'Avatar.*?\$([0-9,]+)', page1)
    if gross_match:
        exact_gross = "$" + gross_match.group(1)
        print(f"Exact gross of #1 film: {exact_gross}")
    else:
        exact_gross = "$2,923,710,708"  # Fallback
        print(f"Using fallback gross: {exact_gross}")
    
    # Extract top 5 films
    top_5 = []
    for line in page1.split('\n'):
        if '|' not in line:
            continue
        # Match pattern: rank|rank|title|$amount|year
        match = re.match(r'(\d+)\|(\d+)\|\*?\[?([^\]|*]+)', line)
        if match and len(top_5) < 5:
            rank = match.group(1)
            title = clean_text(match.group(3))
            # Extract gross
            gross_m = re.search(r'\$([0-9,]+)', line)
            gross = "$" + gross_m.group(1) if gross_m else ""
            # Extract year
            year_m = re.search(r'\|(\d{4})\|', line)
            year = year_m.group(1) if year_m else ""
            if title and gross and year:
                top_5.append(f"{rank}. {title}: {gross}, {year}")
    
    print(f"Top 5 films: {top_5}")
    
    # ===== PAGE 2: Avatar (2009 film) - Chain Level 1 =====
    print("\n--- PAGE 2: Avatar (2009 film) ---")
    page2 = exa_fetch("https://en.wikipedia.org/wiki/Avatar_(2009_film)")
    
    # Verify director is James Cameron
    if "James Cameron" in page2:
        print("  Director: James Cameron ✓")
    
    # ===== PAGE 3: James Cameron filmography - Chain Level 2+3 =====
    print("\n--- PAGE 3: James Cameron filmography ---")
    page3 = exa_fetch("https://en.wikipedia.org/wiki/James_Cameron_filmography")
    
    # Extract first 5 films from filmography
    director_films = []
    for line in page3.split('\n'):
        if '|' not in line:
            continue
        # Match Year|Title pattern
        match = re.search(r'\|?(19\d{2}|20\d{2})\|\*?\[?([^\]|*\(]+)', line)
        if match and len(director_films) < 5:
            year = match.group(1)
            title = clean_text(match.group(2)).strip()
            if title and len(title) > 2 and 'Year' not in title:
                director_films.append(f"{year}: {title}")
    
    print(f"Director's first 5 films: {director_films}")
    
    # ===== PAGE 4: Avengers: Endgame - Direct Reference =====
    print("\n--- PAGE 4: Avengers: Endgame ---")
    page4 = exa_fetch("https://en.wikipedia.org/wiki/Avengers:_Endgame")
    
    # Extract cast - looking for actor|character patterns
    cast = []
    for line in page4.split('\n'):
        # Look for cast table entries
        if '[' in line and 'as' in line.lower():
            # Pattern: [Actor Name](url) as Character
            match = re.search(r'\[([^\]]+)\].*?as\s+([^,\.\[]+)', line, re.IGNORECASE)
            if match and len(cast) < 5:
                actor = clean_text(match.group(1))
                character = clean_text(match.group(2)).strip()
                if actor and character and 'http' not in actor:
                    cast.append(f"{actor}: {character}")
    
    # If that didn't work, try the cast list format
    if len(cast) < 5:
        cast = []
        for line in page4.split('\n'):
            if '|' in line:
                # Pattern: |Actor|Character|
                parts = [p.strip() for p in line.split('|') if p.strip()]
                for i, p in enumerate(parts):
                    if 'Robert Downey' in p or 'Chris Evans' in p or 'Hemsworth' in p:
                        actor = clean_text(p)
                        if i+1 < len(parts):
                            character = clean_text(parts[i+1])
                            if len(cast) < 5:
                                cast.append(f"{actor}: {character}")
    
    print(f"Cast (first 5): {cast}")
    
    # ===== BUILD ANSWERS =====
    print("\n" + "="*70)
    print("BUILDING ANSWERS")
    print("="*70)
    
    top_5_answer = "; ".join(top_5)
    director_films_answer = "; ".join(director_films)
    cast_answer = "; ".join(cast) if cast else "Robert Downey Jr.: Tony Stark; Chris Evans: Steve Rogers; Mark Ruffalo: Bruce Banner; Chris Hemsworth: Thor; Scarlett Johansson: Natasha Romanoff"
    
    print(f"\nexact_gross: {exact_gross}")
    print(f"\ntop_5_films: {top_5_answer}")
    print(f"\ndirector_films: {director_films_answer}")
    print(f"\nfilm_2_cast: {cast_answer}")
    
    # ===== CREATE TASK =====
    task = {
        "id": "task-exa-reference-chain",
        "prompt": """Go to Wikipedia's 'List of highest-grossing films' page.

PART A - EXACT VALUE:
Find the exact worldwide gross (in US dollars, with commas) of the #1 highest-grossing film of all time.
Store as 'exact_gross' (format: $X,XXX,XXX,XXX)

PART B - MAIN TABLE (5 entries):
Extract the top 5 films from the main ranking table.
Format: {Rank}. {Title}: ${Gross}, {Year}
Store as 'top_5_films', semicolon-separated.

PART C - REFERENCE CHAIN (3 levels, 5 entries):
Follow this chain of references:
1. Click on the TITLE of the #1 ranked film
2. On that film's Wikipedia page, find who DIRECTED it and click on their name
3. On the director's page, find the link to their FILMOGRAPHY and navigate there
4. From the filmography table, extract the FIRST 5 feature films (by year)
Format: {Year}: {Title}
Store as 'director_films', semicolon-separated.

PART D - DIRECT REFERENCE (5 entries):
From the original list, click on the #2 ranked film.
Extract 5 main cast members and their character names.
Format: {Actor}: {Character}
Store as 'film_2_cast', semicolon-separated.

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
                    "exact_gross": exact_gross,
                    "top_5_films": top_5_answer,
                    "director_films": director_films_answer,
                    "film_2_cast": cast_answer
                }
            }
        },
        "agent_config": {
            "allowed_tools": ["search", "fetch", "scratchpad_write", "scratchpad_read"]
        }
    }
    
    # Save task
    output_file = "/home/rs/projects/hud-python/environments/mcp-multitools/task_jsons/rs_exa_chain.json"
    with open(output_file, 'w') as f:
        json.dump([task], f, indent=2)
    print(f"\nTask saved to: {output_file}")


if __name__ == "__main__":
    main()

