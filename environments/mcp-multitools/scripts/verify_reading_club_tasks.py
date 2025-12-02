#!/usr/bin/env python3
"""
Verify all 10 reading club task answers by extracting from source files.
Compares JSON expected values against computed values from actual data.
"""

import json
import re

# Load task JSON
with open('task_jsons/reading_club_tasks.json', 'r') as f:
    tasks = json.load(f)

# Load reading club data
with open('scripts/reading_club_data.json', 'r') as f:
    data = json.load(f)

members = data['members']
preferences = data['preferences']
groups = data['groups']
meetings = data['meetings']

member_lookup = {m['id']: m for m in members}

# Load actual document contents
with open('documents/pride_and_prejudice.txt', 'r') as f:
    pride_text = f.read()

with open('documents/us_constitution.txt', 'r') as f:
    constitution_text = f.read()

with open('documents/study_circles.txt', 'r') as f:
    study_circles_text = f.read()

print("=" * 80)
print("VERIFICATION OF ALL 10 TASKS")
print("=" * 80)

all_passed = True

# =============================================================================
# Task 1: task-rc-next-chapter-content
# Isabella Graham's highest-rated fav is P&P Ch2 (rating 10)
# Next chapter is Ch3, first paragraph
# =============================================================================
def verify_task_1():
    # Get Isabella Graham's preferences
    isabella_id = next(m['id'] for m in members if m['full_name'] == 'Isabella Graham')
    prefs = preferences.get(isabella_id, [])
    best = max(prefs, key=lambda p: (p['rating'], p['book']))  # highest rating, then alpha book
    
    print(f"\n1. task-rc-next-chapter-content")
    print(f"   Isabella Graham ({isabella_id}) highest-rated: {best['book']} Ch{best['chapter']} (rating {best['rating']})")
    print(f"   Next chapter to find: Chapter {best['chapter'] + 1}")
    
    # Find Chapter III in Pride and Prejudice
    # The pattern is "CHAPTER III." followed by content
    match = re.search(r'CHAPTER III\.\s*\n+(?:\[Illustration\]\s*\n+)?([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n\[)', pride_text)
    
    if match:
        # Extract first paragraph (up to first double newline or next section)
        first_para_text = match.group(1).strip()
        # Clean up - join lines and get first sentence that forms the paragraph
        first_para = ' '.join(first_para_text.split())
        # Get just the first complete sentence (ends with period followed by space and capital or end)
        first_sentence = re.match(r'^[^.]+\.', first_para)
        if first_sentence:
            computed = first_sentence.group(0)
        else:
            computed = first_para
    else:
        computed = "COULD NOT EXTRACT"
    
    expected = tasks[0]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected[:80]}...")
    print(f"   Computed: {computed[:80]}...")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 2: task-rc-four-people-chapters
# Bronze + Classic Fiction members, their highest-rated chapters
# =============================================================================
def verify_task_2():
    print(f"\n2. task-rc-four-people-chapters")
    
    bronze_classic = [m for m in members 
                      if m['tier'] == 'Bronze' and m['favorite_genre'] == 'Classic Fiction']
    bronze_classic.sort(key=lambda x: x['id'])
    
    results = []
    for m in bronze_classic:
        prefs = preferences.get(m['id'], [])
        if prefs:
            best = max(prefs, key=lambda p: (p['rating'], -p['chapter']))
            results.append(f"{m['full_name']}: {best['book']} Chapter {best['chapter']}")
            print(f"   {m['id']}: {m['full_name']} -> {best['book']} Ch{best['chapter']} (rating {best['rating']})")
    
    computed = "; ".join(results)
    expected = tasks[1]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected}")
    print(f"   Computed: {computed}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 3: task-rc-and-criteria-chapters
# Bronze + Philosophy, first 3, highest-rated chapters
# =============================================================================
def verify_task_3():
    print(f"\n3. task-rc-and-criteria-chapters")
    
    bronze_philosophy = [m for m in members 
                         if m['tier'] == 'Bronze' and m['favorite_genre'] == 'Philosophy']
    bronze_philosophy.sort(key=lambda x: x['id'])
    first_three = bronze_philosophy[:3]
    
    results = []
    for m in first_three:
        prefs = preferences.get(m['id'], [])
        if prefs:
            best = max(prefs, key=lambda p: p['rating'])
            results.append(f"{m['full_name']}: {best['book']} Ch{best['chapter']}")
            print(f"   {m['id']}: {m['full_name']} -> {best['book']} Ch{best['chapter']} (rating {best['rating']})")
    
    computed = "; ".join(results)
    expected = tasks[2]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected}")
    print(f"   Computed: {computed}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 4: task-rc-count-all-group-members
# Count unique members across ALL groups
# =============================================================================
def verify_task_4():
    print(f"\n4. task-rc-count-all-group-members")
    
    all_group_members = set()
    for g in groups:
        all_group_members.update(g['members'])
        print(f"   {g['name']}: {len(g['members'])} members")
    
    computed = str(len(all_group_members))
    expected = tasks[3]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Total unique: {computed}")
    print(f"   Expected: {expected}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 5: task-rc-cross-constitution
# Strategists meeting day + Constitution Article 1 Section 1
# =============================================================================
def verify_task_5():
    print(f"\n5. task-rc-cross-constitution")
    
    # Find Strategists meeting day from study_circles
    match = re.search(r'CIRCLE 1: THE STRATEGISTS.*?Meeting: Every (\w+)', study_circles_text, re.DOTALL)
    meeting_day = match.group(1) if match else "NOT FOUND"
    print(f"   Strategists meeting day: {meeting_day}")
    
    # Extract Article 1 Section 1 from Constitution
    # Pattern: "Section 1." followed by the text until "Section 2."
    match = re.search(r'Section 1\.\s+(.+?)(?=\n\nSection 2\.)', constitution_text, re.DOTALL)
    if match:
        section_text = match.group(1).strip()
        # Normalize whitespace
        section_text = ' '.join(section_text.split())
    else:
        section_text = "NOT FOUND"
    
    print(f"   Constitution Section 1: {section_text[:60]}...")
    
    computed = f"Meeting day: {meeting_day}. Constitution text: {section_text}"
    expected = tasks[4]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected[:80]}...")
    print(f"   Computed: {computed[:80]}...")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 6: task-rc-exact-preamble
# Exact Preamble with line breaks
# =============================================================================
def verify_task_6():
    print(f"\n6. task-rc-exact-preamble")
    
    # Extract Preamble - starts with "We the people" ends with "United States of America."
    match = re.search(r'(We the people of the United States.*?United States of America\.)', constitution_text, re.DOTALL)
    if match:
        computed = match.group(1)
    else:
        computed = "NOT FOUND"
    
    expected = tasks[5]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected (repr): {repr(expected[:80])}...")
    print(f"   Computed (repr): {repr(computed[:80])}...")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    if not match_result:
        print(f"   DIFF:")
        for i, (e, c) in enumerate(zip(expected, computed)):
            if e != c:
                print(f"     Position {i}: expected {repr(e)}, got {repr(c)}")
                break
    
    return match_result

# =============================================================================
# Task 7: task-rc-or-not-criteria
# (Gold OR Silver) NOT in Literary Lions, books > 20, first 5
# =============================================================================
def verify_task_7():
    print(f"\n7. task-rc-or-not-criteria")
    
    literary_lions = next(g for g in groups if g['name'] == 'Literary Lions')
    lions_members = set(literary_lions['members'])
    print(f"   Literary Lions members: {lions_members}")
    
    gold_silver = [m for m in members if m['tier'] in ['Gold', 'Silver']]
    not_in_lions = [m for m in gold_silver if m['id'] not in lions_members]
    with_books = [m for m in not_in_lions if m['books_read'] > 20]
    with_books.sort(key=lambda x: x['id'])
    first_five = with_books[:5]
    
    for m in first_five:
        print(f"   {m['id']}: {m['full_name']} ({m['tier']}, {m['books_read']} books)")
    
    computed = "; ".join([m['full_name'] for m in first_five])
    expected = tasks[6]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected}")
    print(f"   Computed: {computed}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 8: task-rc-override-correction
# Use CORRECTED info: P&P Ch1 rated 10/10 -> first sentence of Ch1
# =============================================================================
def verify_task_8():
    print(f"\n8. task-rc-override-correction")
    
    # The task says to use corrected info: P&P Ch1 rated 10/10
    # So we need first sentence of Pride and Prejudice Chapter I
    
    # Find Chapter I in Pride and Prejudice
    match = re.search(r'Chapter I\.\]\s*\n+(?:\n)*(.+?)(?=\n\n)', pride_text, re.DOTALL)
    if match:
        first_para = match.group(1).strip()
        # Normalize and get first sentence
        first_para = ' '.join(first_para.split())
        # First sentence ends with period
        sentence_match = re.match(r'^(.+?\.)', first_para)
        if sentence_match:
            computed = sentence_match.group(1)
        else:
            computed = first_para
    else:
        computed = "NOT FOUND"
    
    expected = tasks[7]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected}")
    print(f"   Computed: {computed}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 9: task-rc-chain-tuesday-art-war
# Tuesday groups -> members -> Art of War fans
# =============================================================================
def verify_task_9():
    print(f"\n9. task-rc-chain-tuesday-art-war")
    
    tuesday_groups = [g for g in groups if g['meeting_day'] == 'Tuesday']
    print(f"   Tuesday groups: {[g['name'] for g in tuesday_groups]}")
    
    tuesday_members = set()
    for g in tuesday_groups:
        tuesday_members.update(g['members'])
    print(f"   Tuesday members: {len(tuesday_members)} total")
    
    art_war_fans = set()
    for mid in tuesday_members:
        prefs = preferences.get(mid, [])
        for p in prefs:
            if p['book'] == 'The Art of War':
                art_war_fans.add(mid)
                break
    
    fan_names = sorted([member_lookup[mid]['full_name'] for mid in art_war_fans])
    for name in fan_names:
        print(f"     - {name}")
    
    computed = "; ".join(fan_names)
    expected = tasks[8]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected}")
    print(f"   Computed: {computed}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# Task 10: task-rc-platinum-avg-rating
# Platinum members' average ratings by book
# =============================================================================
def verify_task_10():
    print(f"\n10. task-rc-platinum-avg-rating")
    
    platinum_members = [m for m in members if m['tier'] == 'Platinum']
    platinum_ids = set(m['id'] for m in platinum_members)
    print(f"   Platinum members: {len(platinum_ids)}")
    
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
        avg = sum(ratings) / len(ratings)
        book_avgs[book] = avg
        print(f"   {book}: {avg:.2f} average ({len(ratings)} ratings)")
    
    # Find best book
    best_book = max(book_avgs.items(), key=lambda x: (x[1], x[0]))
    
    computed = f"{best_book[0]}: {best_book[1]:.2f} average"
    expected = tasks[9]['evaluate_tool']['arguments']['exact_values']['answer']
    
    match_result = computed == expected
    print(f"   Expected: {expected}")
    print(f"   Computed: {computed}")
    print(f"   MATCH: {'✓ PASS' if match_result else '✗ FAIL'}")
    
    return match_result

# =============================================================================
# RUN ALL VERIFICATIONS
# =============================================================================

results = [
    verify_task_1(),
    verify_task_2(),
    verify_task_3(),
    verify_task_4(),
    verify_task_5(),
    verify_task_6(),
    verify_task_7(),
    verify_task_8(),
    verify_task_9(),
    verify_task_10(),
]

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

passed = sum(results)
failed = len(results) - passed

print(f"\nPassed: {passed}/10")
print(f"Failed: {failed}/10")

if failed > 0:
    print("\nFAILED TASKS:")
    for i, r in enumerate(results, 1):
        if not r:
            print(f"  - Task {i}")
else:
    print("\n✓ ALL TASKS VERIFIED SUCCESSFULLY!")

