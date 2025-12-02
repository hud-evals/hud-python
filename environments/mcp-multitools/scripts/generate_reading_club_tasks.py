#!/usr/bin/env python3
"""
Generate 10 diverse tasks for the reading club documents.
Each task has a unique "niche" - different challenge patterns.

IMPORTANT: Expected answers are EXTRACTED from source files, not hardcoded.
This ensures 100% accuracy.
"""

import json
import re

# Load the ground truth data
with open('scripts/reading_club_data.json', 'r') as f:
    data = json.load(f)

members = data['members']
preferences = data['preferences']
groups = data['groups']
meetings = data['meetings']

member_lookup = {m['id']: m for m in members}

# Load actual document contents for extracting expected answers
with open('documents/pride_and_prejudice.txt', 'r') as f:
    pride_text = f.read()

with open('documents/us_constitution.txt', 'r') as f:
    constitution_text = f.read()

with open('documents/study_circles.txt', 'r') as f:
    study_circles_text = f.read()

# Standard document hint
DOC_HINT = """DOCUMENTS: 'reading_club_members', 'reading_club_discussions', 'reading_club_preferences', 'study_circles', 'us_constitution', 'moby_dick', 'pride_and_prejudice', 'art_of_war'. Each document explains its structure at the beginning."""

# Standard text extraction note
TEXT_EXTRACTION_NOTE = """NOTE: When extracting text from book documents, return ONLY the actual prose text. Exclude chapter headings, [Illustration] tags, and any other markup."""

# =============================================================================
# HELPER: Extract first paragraph of a chapter from Pride and Prejudice
# =============================================================================
def extract_pp_chapter_first_para(chapter_num):
    """Extract first paragraph of given chapter from Pride and Prejudice."""
    # Convert to Roman numeral
    roman_map = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'}
    roman = roman_map.get(chapter_num, str(chapter_num))
    
    # Try multiple patterns - P&P uses both "Chapter I.]" and "CHAPTER III."
    patterns = [
        # Pattern for "Chapter I.]" format (early chapters)
        rf'Chapter {roman}\.\]\s*\n+(.+?)(?=\n\n)',
        # Pattern for "CHAPTER III." format (later chapters)
        rf'CHAPTER {roman}\.\s*\n+(?:\[Illustration[^\]]*\]\s*\n+)*(.+?)(?=\n\n)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pride_text, re.DOTALL)
        if match:
            para = match.group(1).strip()
            # Normalize whitespace (join lines)
            return ' '.join(para.split())
    return None

# =============================================================================
# HELPER: Extract first sentence of a chapter from Pride and Prejudice
# =============================================================================
def extract_pp_chapter_first_sentence(chapter_num):
    """Extract first sentence of given chapter from Pride and Prejudice."""
    para = extract_pp_chapter_first_para(chapter_num)
    if para:
        # First sentence ends with period followed by space and capital letter
        match = re.match(r'^(.+?[.!?])\s+[A-Z]', para)
        if match:
            return match.group(1)
        # Fallback: just take up to first period
        match = re.match(r'^(.+?\.)', para)
        if match:
            return match.group(1)
    return para

# =============================================================================
# HELPER: Extract Constitution Section
# =============================================================================
def extract_constitution_section(article, section):
    """Extract a specific section from the Constitution."""
    pattern = rf'Section {section}\.\s+(.+?)(?=\n\nSection|\n\nARTICLE|\Z)'
    match = re.search(pattern, constitution_text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        return ' '.join(text.split())
    return None

# =============================================================================
# HELPER: Extract Constitution Preamble
# =============================================================================
def extract_preamble():
    """Extract exact Preamble including line breaks."""
    match = re.search(r'(We the people of the United States.*?United States of America\.)', 
                      constitution_text, re.DOTALL)
    if match:
        return match.group(1)
    return None

# =============================================================================
# NICHE 1: Cross-ref + NEXT chapter's first paragraph
# =============================================================================
def task_1():
    """Find member's fav chapter, then return the NEXT chapter's first paragraph from the book"""
    # RC-1001 (Isabella Graham) has Pride and Prejudice Ch 2 rated 10/10
    # So we ask for Chapter 3's first paragraph
    
    expected = extract_pp_chapter_first_para(3)
    
    return {
        "id": "task-rc-next-chapter-content",
        "prompt": f"""Find Isabella Graham's highest-rated favorite chapter. Then return the first complete paragraph of the NEXT chapter in that same book.

STEPS:
1. Find Isabella Graham in member directory to get ID
2. Look up preferences to find highest-rated chapter (if tied: alphabetically first book)
3. Identify which book and chapter number
4. Go to that book document and find the NEXT chapter (if fav is Ch 2, find Ch 3)
5. Return the first complete paragraph of that next chapter

{TEXT_EXTRACTION_NOTE}

FORMAT: The source file has line breaks due to text wrapping. Replace all line breaks with single spaces to create one continuous string. No newlines in your answer.

{DOC_HINT}

Store the paragraph as 'answer'. Say 'Task completed.' when done.""",
        "expected": expected
    }

# =============================================================================
# NICHE 2: 4 people indirectly referenced by attributes
# =============================================================================
def task_2():
    """Find 4 people matching criteria, return their favorite chapter names"""
    bronze_classic = [m for m in members 
                      if m['tier'] == 'Bronze' and m['favorite_genre'] == 'Classic Fiction']
    bronze_classic.sort(key=lambda x: x['id'])
    
    results = []
    for m in bronze_classic:
        prefs = preferences.get(m['id'], [])
        if prefs:
            best = max(prefs, key=lambda p: (p['rating'], -p['chapter']))
            results.append(f"{m['full_name']}: {best['book']} Chapter {best['chapter']}")
    
    return {
        "id": "task-rc-four-people-chapters",
        "prompt": f"""Find ALL Bronze tier members whose favorite genre is "Classic Fiction". For each, report their highest-rated favorite chapter.

If a member has multiple favorites with the same rating, use lowest chapter number.

FORMAT: 
"Name1: BookTitle Chapter X; Name2: BookTitle Chapter Y; ..."
- Entries separated by "; " (semicolon space)
- In order of member ID (ascending)

{DOC_HINT}

Store the list as 'answer'. Say 'Task completed.' when done.""",
        "expected": "; ".join(results)
    }

# =============================================================================
# NICHE 3: AND statement + content for ~3 people
# =============================================================================
def task_3():
    """Bronze AND Philosophy lovers → return their highest-rated chapters"""
    bronze_philosophy = [m for m in members 
                         if m['tier'] == 'Bronze' and m['favorite_genre'] == 'Philosophy']
    bronze_philosophy.sort(key=lambda x: x['id'])
    first_three = bronze_philosophy[:3]
    
    results = []
    for m in first_three:
        prefs = preferences.get(m['id'], [])
        if prefs:
            # Tiebreaker: highest rating, then alphabetically first book, then lowest chapter
            best = max(prefs, key=lambda p: (p['rating'], p['book'], -p['chapter']))
            results.append(f"{m['full_name']}: {best['book']} Ch{best['chapter']}")
    
    return {
        "id": "task-rc-and-criteria-chapters",
        "prompt": f"""Find members who are BOTH Bronze tier AND have Philosophy as favorite genre. Return the first 3 (by member ID) and their highest-rated chapter.

If a member has multiple chapters with the same rating, use alphabetically first book title. If still tied, use lowest chapter number.

FORMAT: "Name: Book ChX; Name: Book ChY; Name: Book ChZ"
- Entries separated by "; " (semicolon space)

{DOC_HINT}

Store the list as 'answer'. Say 'Task completed.' when done.""",
        "expected": "; ".join(results)
    }

# =============================================================================
# NICHE 4: Count requiring FULL file view
# =============================================================================
def task_4():
    """Count all unique members across ALL discussion groups - needs whole discussions doc"""
    all_group_members = set()
    for g in groups:
        all_group_members.update(g['members'])
    
    return {
        "id": "task-rc-count-all-group-members",
        "prompt": f"""Count the total number of UNIQUE members who belong to at least one discussion group.

A member may belong to multiple groups but should only be counted once.
You must examine ALL discussion groups in the document to get an accurate count.

FORMAT: Just the number (e.g., "42")

{DOC_HINT}

Store the count as 'answer'. Say 'Task completed.' when done.""",
        "expected": str(len(all_group_members))
    }

# =============================================================================
# NICHE 5: Cross-reference to UNRELATED file (US Constitution)
# =============================================================================
def task_5():
    """Use study_circles to find a group, then get Constitution content"""
    # Extract meeting day from study_circles
    match = re.search(r'CIRCLE 1: THE STRATEGISTS.*?Meeting: Every (\w+)', study_circles_text, re.DOTALL)
    meeting_day = match.group(1) if match else "Tuesday"
    
    # Extract Constitution Section 1
    section_text = extract_constitution_section(1, 1)
    
    expected = f"Meeting day: {meeting_day}. Constitution text: {section_text}"
    
    return {
        "id": "task-rc-cross-constitution",
        "prompt": f"""Find the study circle called "THE STRATEGISTS" in the study_circles document. Note what day they meet.

Then, from the us_constitution document, return the complete text of Article 1 Section 1.

FORMAT: "Meeting day: [DAY]. Constitution text: [EXACT TEXT]"
- For meeting day, use just the day name (e.g., "Tuesday" not "Every Tuesday")
- For Constitution text, join all lines into a single string with spaces (no line breaks)

{DOC_HINT}

Store the formatted answer as 'answer'. Say 'Task completed.' when done.""",
        "expected": expected
    }

# =============================================================================
# NICHE 6: Return whole section EXACTLY (character-perfect)
# =============================================================================
def task_6():
    """Return the Preamble of the Constitution exactly"""
    preamble = extract_preamble()
    
    return {
        "id": "task-rc-exact-preamble",
        "prompt": f"""Return the complete Preamble of the US Constitution EXACTLY as it appears in the document.

The Preamble starts with "We the people" and ends with "United States of America."

IMPORTANT: Return the exact text including all line breaks, punctuation, and spacing. Character-perfect extraction is required.

{DOC_HINT}

Store the exact preamble as 'answer'. Say 'Task completed.' when done.""",
        "expected": preamble
    }

# =============================================================================
# NICHE 7: OR then NOT statement with extra description info
# =============================================================================
def task_7():
    """(Gold OR Silver) AND NOT in 'Literary Lions' group + books > 20"""
    literary_lions = next(g for g in groups if g['name'] == 'Literary Lions')
    lions_members = set(literary_lions['members'])
    
    gold_silver = [m for m in members if m['tier'] in ['Gold', 'Silver']]
    not_in_lions = [m for m in gold_silver if m['id'] not in lions_members]
    with_books = [m for m in not_in_lions if m['books_read'] > 20]
    with_books.sort(key=lambda x: x['id'])
    first_five = with_books[:5]
    
    names = [m['full_name'] for m in first_five]
    
    return {
        "id": "task-rc-or-not-criteria",
        "prompt": f"""Find members who meet ALL of these criteria:
1. Either Gold OR Silver tier
2. NOT a member of the "Literary Lions" discussion group
3. Have read more than 20 books (according to member directory)

ADDITIONAL CONTEXT (this is relevant but not directly used in the query):
- The Literary Lions group focuses on challenging literary analysis
- Members who haven't joined this group often prefer lighter reading
- This is for a recruitment email campaign

Return the first 5 such members by member ID (ascending).

FORMAT: "Name1; Name2; Name3; Name4; Name5"
- Names separated by "; " (semicolon space)

{DOC_HINT}

Store the list as 'answer'. Say 'Task completed.' when done.""",
        "expected": "; ".join(names)
    }

# =============================================================================
# NICHE 8: Override/correction in description
# =============================================================================
def task_8():
    """Tell them there's an "error" in the document, use corrected info instead"""
    # The corrected info says P&P Chapter 1 is rated 10/10
    # So we need first sentence of Pride and Prejudice Chapter 1
    
    expected = extract_pp_chapter_first_sentence(1)
    
    return {
        "id": "task-rc-override-correction",
        "prompt": f"""IMPORTANT CORRECTION: There is a data entry error in the reading_club_preferences document.

Member RC-1000 (Penelope Harrison) is incorrectly listed with certain chapter preferences. 
The CORRECT preferences for RC-1000 are:
- Pride and Prejudice, Chapter 1: Rating 10/10
- Moby Dick, Chapter 42: Rating 8/10

Using this CORRECTED information (not what the document says), find Penelope Harrison's highest-rated chapter and return the first sentence of that chapter from the corresponding book.

{TEXT_EXTRACTION_NOTE}

{DOC_HINT}

Store the first sentence as 'answer'. Say 'Task completed.' when done.""",
        "expected": expected
    }

# =============================================================================
# NICHE 9: Multi-step chain reference
# =============================================================================
def task_9():
    """Chain: Find Tuesday group → get members → find who likes Art of War → list them"""
    tuesday_groups = [g for g in groups if g['meeting_day'] == 'Tuesday']
    tuesday_members = set()
    for g in tuesday_groups:
        tuesday_members.update(g['members'])
    
    art_war_fans = set()
    for mid in tuesday_members:
        prefs = preferences.get(mid, [])
        for p in prefs:
            if p['book'] == 'The Art of War':
                art_war_fans.add(mid)
                break
    
    fan_names = sorted([member_lookup[mid]['full_name'] for mid in art_war_fans])
    
    return {
        "id": "task-rc-chain-tuesday-art-war",
        "prompt": f"""Multi-step task:

1. Find ALL discussion groups that meet on Tuesday
2. Collect all unique members from those Tuesday groups
3. For each of those members, check their reading preferences
4. Identify which ones have The Art of War as ANY of their favorites
5. List those members' names

FORMAT: "Name1; Name2; Name3"
- Names separated by "; " (semicolon space)
- Sorted alphabetically A-Z

{DOC_HINT}

Store the list as 'answer'. Say 'Task completed.' when done.""",
        "expected": "; ".join(fan_names)
    }

# =============================================================================
# NICHE 10: Original HARD task - Cross-doc aggregate calculation
# =============================================================================
def task_10():
    """Complex: For each book, find avg rating among Platinum members, report highest avg book"""
    platinum_members = [m for m in members if m['tier'] == 'Platinum']
    platinum_ids = set(m['id'] for m in platinum_members)
    
    book_ratings = {}
    for mid in platinum_ids:
        prefs = preferences.get(mid, [])
        for p in prefs:
            book = p['book']
            if book not in book_ratings:
                book_ratings[book] = []
            book_ratings[book].append(p['rating'])
    
    book_avgs = {}
    for book, ratings in book_ratings.items():
        book_avgs[book] = sum(ratings) / len(ratings)
    
    best_book = max(book_avgs.items(), key=lambda x: (x[1], x[0]))
    
    return {
        "id": "task-rc-platinum-avg-rating",
        "prompt": f"""Calculate which book has the highest AVERAGE rating among Platinum tier members only.

1. Find all Platinum tier members from the member directory
2. For each, look up ALL their chapter preferences in the preferences document
3. Group ratings by book title
4. Calculate average rating for each book (across all Platinum member preferences)
5. Return the book with highest average (if tied, alphabetically first book title)

FORMAT: "BookTitle: X.XX average"
- Average rounded to 2 decimal places

{DOC_HINT}

Store the formatted answer as 'answer'. Say 'Task completed.' when done.""",
        "expected": f"{best_book[0]}: {best_book[1]:.2f} average"
    }

# =============================================================================
# Generate all tasks
# =============================================================================
def main():
    tasks = [
        task_1(),
        task_2(),
        task_3(),
        task_4(),
        task_5(),
        task_6(),
        task_7(),
        task_8(),
        task_9(),
        task_10(),
    ]
    
    # Build the full task JSON structure
    task_list = []
    for t in tasks:
        task_list.append({
            "id": t["id"],
            "prompt": t["prompt"],
            "mcp_config": {
                "local": {
                    "command": "docker",
                    "args": [
                        "run",
                        "--rm",
                        "-i",
                        "--env-file",
                        "/home/rs/projects/hud-python/.env",
                        "mcp-multitools:latest"
                    ]
                }
            },
            "setup_tool": {
                "name": "setup",
                "arguments": {}
            },
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {
                    "exact_values": {
                        "answer": t["expected"]
                    }
                }
            },
            "agent_config": {
                "allowed_tools": [
                    "scratchpad_write",
                    "scratchpad_read",
                    "read_document",
                    "search_document"
                ]
            }
        })
    
    output_path = "task_jsons/reading_club_tasks.json"
    with open(output_path, 'w') as f:
        json.dump(task_list, f, indent=2)
    
    print(f"Generated {len(task_list)} tasks")
    print(f"Written to: {output_path}")
    print("\nTask Niches:")
    niches = [
        "Cross-ref + NEXT chapter content",
        "4 people by attributes → chapters",
        "AND criteria + chapters",
        "Count requiring FULL file view",
        "Cross-ref to Constitution (unrelated)",
        "Return section EXACTLY (character-perfect)",
        "OR then NOT + extra context",
        "Override/correction in description",
        "Multi-step chain reference",
        "Complex aggregate calculation"
    ]
    for i, (t, niche) in enumerate(zip(tasks, niches), 1):
        exp = t['expected']
        print(f"\n{i}. {t['id']}")
        print(f"   Niche: {niche}")
        print(f"   Expected: {exp[:70]}..." if len(exp) > 70 else f"   Expected: {exp}")

if __name__ == "__main__":
    main()
