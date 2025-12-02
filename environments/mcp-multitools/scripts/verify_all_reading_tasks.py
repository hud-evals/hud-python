#!/usr/bin/env python3
"""
Comprehensive verification script for reading_club_tasks.json
Programmatically extracts answers from source documents and compares to JSON graders.
"""

import json
import re
import os

# Paths
BASE_DIR = "/home/rs/projects/hud-python/environments/mcp-multitools"
DATA_JSON = os.path.join(BASE_DIR, "scripts/reading_club_data.json")
TASKS_JSON = os.path.join(BASE_DIR, "task_jsons/reading_club_tasks.json")
DOCS_DIR = os.path.join(BASE_DIR, "documents")

def load_data():
    """Load ground truth data."""
    with open(DATA_JSON, 'r') as f:
        return json.load(f)

def load_tasks():
    """Load task definitions."""
    with open(TASKS_JSON, 'r') as f:
        return json.load(f)

def read_doc(filename):
    """Read a document file."""
    filepath = os.path.join(DOCS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def get_members_by_criteria(data, tier=None, genre=None):
    """Get members matching criteria."""
    results = []
    for m in data['members']:
        if tier and m['tier'] != tier:
            continue
        if genre and m['favorite_genre'] != genre:
            continue
        results.append(m)
    return sorted(results, key=lambda x: x['id'])

def get_member_preferences(data, member_id):
    """Get preferences for a member."""
    return data['preferences'].get(member_id, [])

def get_highest_rated_chapter(prefs, book_filter=None, tiebreaker='alpha_book'):
    """Get highest rated chapter with tiebreaker logic."""
    filtered = prefs if not book_filter else [p for p in prefs if p['book'] == book_filter]
    if not filtered:
        return None
    
    max_rating = max(p['rating'] for p in filtered)
    top_rated = [p for p in filtered if p['rating'] == max_rating]
    
    if len(top_rated) == 1:
        return top_rated[0]
    
    if tiebreaker == 'alpha_book':
        top_rated.sort(key=lambda x: (x['book'], x['chapter']))
    elif tiebreaker == 'lowest_chapter':
        top_rated.sort(key=lambda x: x['chapter'])
    
    return top_rated[0]

def get_group_members(data, group_name):
    """Get all members of a discussion group."""
    for group in data['groups']:
        if group['name'] == group_name:
            return group['members']
    return []

def get_tuesday_groups(data):
    """Get all groups that meet on Tuesday."""
    return [g for g in data['groups'] if g['meeting_day'] == 'Tuesday']

def get_literary_lions_members(data):
    """Get members of Literary Lions."""
    for group in data['groups']:
        if group['name'] == 'Literary Lions':
            return set(group['members'])
    return set()

# ============ ROMAN NUMERAL HELPERS ============

def to_roman(n):
    """Convert number to roman numeral."""
    romans = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 
              8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII', 13: 'XIII',
              58: 'LVIII', 45: 'XLV', 20: 'XX', 34: 'XXXIV'}
    return romans.get(n, str(n))

# ============ BOOK EXTRACTION FUNCTIONS ============

def extract_moby_dick_chapter_start(moby_text, chapter_num):
    """Find the start of a Moby Dick chapter (skip table of contents)."""
    # Pattern: "CHAPTER X. Title.\n\n" followed by actual prose (not another chapter heading)
    pattern = rf'CHAPTER {chapter_num}\.\s+[^\n]+\n\n'
    for match in re.finditer(pattern, moby_text):
        after = moby_text[match.end():]
        # Check if this is followed by prose, not another chapter heading
        if not after.startswith('CHAPTER'):
            return after
    return None

def extract_moby_dick_first_sentence(moby_text, chapter_num):
    """Extract first sentence of a Moby Dick chapter."""
    after = extract_moby_dick_chapter_start(moby_text, chapter_num)
    if not after:
        return None
    
    # Get first paragraph
    lines = []
    for line in after.split('\n'):
        if line.strip() == '':
            break
        lines.append(line.strip())
    
    text = ' '.join(lines)
    # First sentence ends at period followed by space
    match = re.match(r'^(.+?\.)\s', text)
    if match:
        return match.group(1)
    if text.endswith('.'):
        return text
    return text

def extract_moby_dick_paragraph(moby_text, chapter_num, para_index):
    """Extract a specific paragraph from Moby Dick chapter."""
    after = extract_moby_dick_chapter_start(moby_text, chapter_num)
    if not after:
        return None
    
    paragraphs = []
    current = []
    for line in after.split('\n'):
        if line.strip() == '':
            if current:
                paragraphs.append(' '.join(current))
                current = []
        else:
            current.append(line.strip())
        if re.match(r'CHAPTER \d+\.', line.strip()):
            break
    
    if current:
        paragraphs.append(' '.join(current))
    
    if para_index < len(paragraphs):
        return paragraphs[para_index]
    return None

def extract_pp_chapter_start(pp_text, chapter_num):
    """Find start of Pride and Prejudice chapter."""
    roman = to_roman(chapter_num)
    
    # Chapter 1 is special: "Chapter I.]"
    if chapter_num == 1:
        pattern = r'Chapter I\.\]\n+'
    else:
        # Other chapters: "CHAPTER II." or "CHAPTER III."
        pattern = rf'CHAPTER {roman}\.\n+'
    
    match = re.search(pattern, pp_text, re.IGNORECASE)
    if match:
        return pp_text[match.end():]
    return None

def extract_pp_first_sentence(pp_text, chapter_num):
    """Extract first sentence of P&P chapter."""
    after = extract_pp_chapter_start(pp_text, chapter_num)
    if not after:
        return None
    
    # Skip [Illustration] markers and blank lines
    lines = []
    in_content = False
    for line in after.split('\n'):
        stripped = line.strip()
        if stripped.startswith('[Illustration'):
            continue
        if stripped == '':
            if in_content:
                break
            continue
        in_content = True
        lines.append(stripped)
    
    text = ' '.join(lines)
    match = re.match(r'^(.+?\.)\s', text)
    if match:
        return match.group(1)
    if text.endswith('.'):
        return text
    return text

def extract_pp_paragraph(pp_text, chapter_num, para_index):
    """Extract specific paragraph from P&P chapter."""
    after = extract_pp_chapter_start(pp_text, chapter_num)
    if not after:
        return None
    
    paragraphs = []
    current = []
    
    for line in after.split('\n'):
        stripped = line.strip()
        if stripped.startswith('[Illustration'):
            continue
        if stripped == '':
            if current:
                paragraphs.append(' '.join(current))
                current = []
        else:
            current.append(stripped)
        if re.match(r'CHAPTER [IVXLC]+\.', stripped):
            break
        if len(paragraphs) > para_index + 1:
            break
    
    if current and len(paragraphs) <= para_index:
        paragraphs.append(' '.join(current))
    
    if para_index < len(paragraphs):
        return paragraphs[para_index]
    return None

def extract_aow_chapter_start(aow_text, chapter_num):
    """Find start of Art of War chapter."""
    roman = to_roman(chapter_num)
    
    # Pattern: "Chapter IV. Tactical Dispositions"
    pattern = rf'^Chapter {roman}\.\s+[^\n]+\n'
    match = re.search(pattern, aow_text, re.MULTILINE)
    if match:
        after = aow_text[match.end():]
        # Find the next chapter to limit scope
        next_ch = re.search(r'^Chapter [IVXLC]+\.', after, re.MULTILINE)
        if next_ch:
            return after[:next_ch.start()]
        return after
    return None

def extract_aow_first_sentence(aow_text, chapter_num):
    """Extract first sentence of Art of War chapter."""
    after = extract_aow_chapter_start(aow_text, chapter_num)
    if not after:
        return None
    
    # Find first numbered point
    match = re.search(r'^1\.\s+(.+?)(?=\n\n|\n\[|\n2\.)', after, re.MULTILINE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
        text = ' '.join(text.split())
        sent_match = re.match(r'^(.+?\.)', text)
        if sent_match:
            return sent_match.group(1)
        return text
    return None

def extract_aow_sections(aow_text, chapter_num, section_nums):
    """Extract specific sections from Art of War chapter."""
    after = extract_aow_chapter_start(aow_text, chapter_num)
    if not after:
        return None
    
    sections = []
    for num in section_nums:
        pattern = rf'^{num}\.\s+(.+?)(?=\n\n|\n\d+\.|\n\[)'
        match = re.search(pattern, after, re.MULTILINE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
            text = ' '.join(text.split())
            sections.append(f"{num}. {text}")
    
    return sections

# ============ VERIFICATION FUNCTIONS ============

def verify_task_1(data, tasks):
    """Task: task-rc-next-chapter-content - Isabella Graham's next chapter paragraph."""
    task = next(t for t in tasks if t['id'] == 'task-rc-next-chapter-content')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    pp_text = read_doc('pride_and_prejudice.txt')
    computed = extract_pp_paragraph(pp_text, 3, 0)
    
    return computed == expected, computed or "None", expected

def verify_task_2(data, tasks):
    """Task: task-rc-four-people-chapters - Bronze + Classic Fiction members."""
    task = next(t for t in tasks if t['id'] == 'task-rc-four-people-chapters')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    members = get_members_by_criteria(data, tier='Bronze', genre='Classic Fiction')
    
    results = []
    for m in members:
        prefs = get_member_preferences(data, m['id'])
        if prefs:
            highest = get_highest_rated_chapter(prefs, tiebreaker='lowest_chapter')
            if highest:
                results.append(f"{m['full_name']}: {highest['book']} Chapter {highest['chapter']}")
    
    computed = "; ".join(results)
    return computed == expected, computed, expected

def verify_task_3(data, tasks):
    """Task: task-rc-and-criteria-chapters - Bronze + Philosophy first 3."""
    task = next(t for t in tasks if t['id'] == 'task-rc-and-criteria-chapters')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    members = get_members_by_criteria(data, tier='Bronze', genre='Philosophy')[:3]
    
    results = []
    for m in members:
        prefs = get_member_preferences(data, m['id'])
        if prefs:
            highest = get_highest_rated_chapter(prefs, tiebreaker='alpha_book')
            if highest:
                results.append(f"{m['full_name']}: {highest['book']} Ch{highest['chapter']}")
    
    computed = "; ".join(results)
    return computed == expected, computed, expected

def verify_task_4(data, tasks):
    """Task: task-rc-count-all-group-members - Count unique group members."""
    task = next(t for t in tasks if t['id'] == 'task-rc-count-all-group-members')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    all_members = set()
    for group in data['groups']:
        all_members.update(group['members'])
    
    computed = str(len(all_members))
    return computed == expected, computed, expected

def verify_task_5(data, tasks):
    """Task: task-rc-cross-constitution - Strategists meeting day + constitution."""
    task = next(t for t in tasks if t['id'] == 'task-rc-cross-constitution')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    # Read study_circles to find THE STRATEGISTS
    sc_text = read_doc('study_circles.txt')
    match = re.search(r'THE STRATEGISTS.*?Meeting:\s*Every\s+(\w+)', sc_text, re.DOTALL | re.IGNORECASE)
    meeting_day = match.group(1) if match else "Unknown"
    
    # Read constitution for Article 1 Section 1
    const_text = read_doc('us_constitution.txt')
    # Pattern: "Section 1.  All legislative..."
    match = re.search(r'Section 1\.\s+(.+?)(?=\n\nSection)', const_text, re.DOTALL)
    if match:
        section_text = ' '.join(match.group(1).strip().split())
    else:
        section_text = ""
    
    computed = f"Meeting day: {meeting_day}. Constitution text: {section_text}"
    return computed == expected, computed, expected

def verify_task_6(data, tasks):
    """Task: task-rc-exact-preamble - Exact preamble with line breaks."""
    task = next(t for t in tasks if t['id'] == 'task-rc-exact-preamble')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    const_text = read_doc('us_constitution.txt')
    match = re.search(r'(We the people.*?United States of America\.)', const_text, re.DOTALL)
    
    computed = match.group(1) if match else ""
    return computed == expected, computed, expected

def verify_task_7(data, tasks):
    """Task: task-rc-or-not-criteria - Gold/Silver, NOT Literary Lions, >20 books."""
    task = next(t for t in tasks if t['id'] == 'task-rc-or-not-criteria')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    lions_members = get_literary_lions_members(data)
    
    qualifying = []
    for m in data['members']:
        if m['tier'] not in ['Gold', 'Silver']:
            continue
        if m['id'] in lions_members:
            continue
        if m['books_read'] <= 20:
            continue
        qualifying.append(m)
    
    qualifying.sort(key=lambda x: x['id'])
    first_5 = qualifying[:5]
    
    computed = "; ".join(m['full_name'] for m in first_5)
    return computed == expected, computed, expected

def verify_task_8(data, tasks):
    """Task: task-rc-override-correction - Use corrected preferences for RC-1000."""
    task = next(t for t in tasks if t['id'] == 'task-rc-override-correction')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    pp_text = read_doc('pride_and_prejudice.txt')
    computed = extract_pp_first_sentence(pp_text, 1)
    
    return computed == expected, computed or "None", expected

def verify_task_9(data, tasks):
    """Task: task-rc-chain-tuesday-art-war - Tuesday groups + Art of War favorites."""
    task = next(t for t in tasks if t['id'] == 'task-rc-chain-tuesday-art-war')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    tuesday_groups = get_tuesday_groups(data)
    tuesday_members = set()
    for g in tuesday_groups:
        tuesday_members.update(g['members'])
    
    art_of_war_lovers = []
    for mid in tuesday_members:
        prefs = get_member_preferences(data, mid)
        for p in prefs:
            if p['book'] == 'The Art of War':
                member = next(m for m in data['members'] if m['id'] == mid)
                art_of_war_lovers.append(member['full_name'])
                break
    
    art_of_war_lovers.sort()
    computed = "; ".join(art_of_war_lovers)
    return computed == expected, computed, expected

def verify_task_10(data, tasks):
    """Task: task-rc-platinum-avg-rating - Highest avg rating book for Platinum."""
    task = next(t for t in tasks if t['id'] == 'task-rc-platinum-avg-rating')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    platinum_members = [m for m in data['members'] if m['tier'] == 'Platinum']
    
    book_ratings = {}
    for m in platinum_members:
        prefs = get_member_preferences(data, m['id'])
        for p in prefs:
            book = p['book']
            if book not in book_ratings:
                book_ratings[book] = []
            book_ratings[book].append(p['rating'])
    
    averages = {}
    for book, ratings in book_ratings.items():
        averages[book] = sum(ratings) / len(ratings)
    
    max_avg = max(averages.values())
    top_books = [b for b, avg in averages.items() if avg == max_avg]
    top_books.sort()
    winner = top_books[0]
    
    computed = f"{winner}: {averages[winner]:.2f} average"
    return computed == expected, computed, expected

def verify_task_11(data, tasks):
    """Task: task-rc-second-paragraph-moby - Olivia Crawford's second paragraph."""
    task = next(t for t in tasks if t['id'] == 'task-rc-second-paragraph-moby')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    moby_text = read_doc('moby_dick.txt')
    computed = extract_moby_dick_paragraph(moby_text, 1, 1)
    
    return computed == expected, computed or "None", expected

def verify_task_12(data, tasks):
    """Task: task-rc-three-first-sentences - Silver + Fantasy first 3 members."""
    task = next(t for t in tasks if t['id'] == 'task-rc-three-first-sentences')
    expected_json = task['evaluate_tool']['arguments']['exact_values']['answer']
    expected = json.loads(expected_json)
    
    members = get_members_by_criteria(data, tier='Silver', genre='Fantasy')[:3]
    
    moby_text = read_doc('moby_dick.txt')
    pp_text = read_doc('pride_and_prejudice.txt')
    aow_text = read_doc('art_of_war.txt')
    
    computed = {}
    for i, m in enumerate(members):
        prefs = get_member_preferences(data, m['id'])
        highest = get_highest_rated_chapter(prefs, tiebreaker='alpha_book')
        
        book = highest['book']
        chapter = highest['chapter']
        
        if 'Pride' in book:
            sentence = extract_pp_first_sentence(pp_text, chapter)
        elif 'Moby' in book:
            sentence = extract_moby_dick_first_sentence(moby_text, chapter)
        elif 'Art' in book:
            sentence = extract_aow_first_sentence(aow_text, chapter)
        else:
            sentence = None
        
        computed[f"member{i+1}"] = sentence
    
    computed_json = json.dumps(computed)
    return computed_json == expected_json, computed_json, expected_json

def verify_task_13(data, tasks):
    """Task: task-rc-art-of-war-sections - Uma Bennett's Art of War sections."""
    task = next(t for t in tasks if t['id'] == 'task-rc-art-of-war-sections')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    aow_text = read_doc('art_of_war.txt')
    sections = extract_aow_sections(aow_text, 4, [1, 2, 3])
    
    if sections:
        computed = " | ".join(sections)
    else:
        computed = "Could not extract sections"
    
    return computed == expected, computed, expected

def verify_task_14(data, tasks):
    """Task: task-rc-group-moby-sentences - Chapter Chasers with Moby Dick."""
    task = next(t for t in tasks if t['id'] == 'task-rc-group-moby-sentences')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    group_members = get_group_members(data, 'Chapter Chasers')
    moby_text = read_doc('moby_dick.txt')
    
    results = []
    for mid in group_members:
        prefs = get_member_preferences(data, mid)
        moby_prefs = [p for p in prefs if p['book'] == 'Moby Dick']
        if not moby_prefs:
            continue
        
        highest = get_highest_rated_chapter(moby_prefs)
        member = next(m for m in data['members'] if m['id'] == mid)
        
        sentence = extract_moby_dick_first_sentence(moby_text, highest['chapter'])
        results.append((member['full_name'], sentence))
    
    results.sort(key=lambda x: x[0])
    computed = "; ".join(f"{name}: {sentence}" for name, sentence in results)
    
    return computed == expected, computed, expected

def verify_task_15(data, tasks):
    """Task: task-rc-full-pp-opening - Olivia Lawson's P&P first two paragraphs."""
    task = next(t for t in tasks if t['id'] == 'task-rc-full-pp-opening')
    expected = task['evaluate_tool']['arguments']['exact_values']['answer']
    
    pp_text = read_doc('pride_and_prejudice.txt')
    
    para1 = extract_pp_paragraph(pp_text, 1, 0)
    para2 = extract_pp_paragraph(pp_text, 1, 1)
    
    if para1 and para2:
        computed = f"{para1} || {para2}"
    else:
        computed = f"Could not extract: para1={para1}, para2={para2}"
    
    return computed == expected, computed, expected

def main():
    print("=" * 70)
    print("READING CLUB TASKS VERIFICATION")
    print("=" * 70)
    print()
    
    data = load_data()
    tasks = load_tasks()
    
    verifiers = [
        ("task-rc-next-chapter-content", verify_task_1),
        ("task-rc-four-people-chapters", verify_task_2),
        ("task-rc-and-criteria-chapters", verify_task_3),
        ("task-rc-count-all-group-members", verify_task_4),
        ("task-rc-cross-constitution", verify_task_5),
        ("task-rc-exact-preamble", verify_task_6),
        ("task-rc-or-not-criteria", verify_task_7),
        ("task-rc-override-correction", verify_task_8),
        ("task-rc-chain-tuesday-art-war", verify_task_9),
        ("task-rc-platinum-avg-rating", verify_task_10),
        ("task-rc-second-paragraph-moby", verify_task_11),
        ("task-rc-three-first-sentences", verify_task_12),
        ("task-rc-art-of-war-sections", verify_task_13),
        ("task-rc-group-moby-sentences", verify_task_14),
        ("task-rc-full-pp-opening", verify_task_15),
    ]
    
    passed = 0
    failed = 0
    
    for task_id, verifier in verifiers:
        try:
            match, computed, expected = verifier(data, tasks)
            if match:
                print(f"✅ PASS: {task_id}")
                passed += 1
            else:
                print(f"❌ FAIL: {task_id}")
                print(f"   COMPUTED: {computed[:200]}...")
                print(f"   EXPECTED: {expected[:200]}...")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {task_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(verifiers)} tasks")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
