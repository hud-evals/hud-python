#!/usr/bin/env python3
"""Verify task-bulk-retrieval-25-highest by solving it correctly."""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Students with their disambiguation hints
# Format: (first_name, last_name, school_name, date_of_birth or None)
students = [
    ("Joseph", "Lopez", "Madison Prep", None),
    ("Lillian", "Gonzalez", "Hamilton High", None),
    ("Carter", "Hernandez", "Monroe Secondary", None),
    ("Sophia", "Torres", None, None),  # No hint - only one in DB
    ("Hannah", "Scott", "Madison Prep", None),
    ("Joseph", "Williams", "Washington Academy", None),
    ("Lily", "Campbell", "Jackson High School", None),
    ("Chloe", "Rivera", "Madison Prep", None),
    ("Carter", "Brown", "Monroe Secondary", None),
    ("Michael", "Gonzalez", "Adams Institute", "2008-08-25"),
    ("John", "Allen", "Jackson High School", None),
    ("Ethan", "King", "Taylor High", None),
    ("Sophia", "Young", "Monroe Secondary", None),
    ("Daniel", "Lewis", "Taylor High", None),
    ("Carter", "Clark", "Jefferson Prep", None),
    ("Abigail", "Johnson", "Taylor High", None),
    ("Victoria", "Hall", "Jefferson Prep", None),
    ("Emily", "Torres", "Kennedy High", None),
    ("Evelyn", "Anderson", "Monroe Secondary", "2009-09-10"),
    ("Jack", "Lee", "Monroe Secondary", None),
    ("Lily", "Clark", "Kennedy High", None),
    ("Ava", "Williams", "Monroe Secondary", "2005-03-19"),
    ("Ethan", "Ramirez", "Franklin Academy", None),
    ("Sebastian", "Torres", "Jefferson Prep", "2012-07-14"),
    ("Ella", "Nguyen", "Adams Institute", None),
]

def fetch_all(table):
    """Fetch all rows from a table."""
    all_data = []
    offset = 0
    while True:
        result = supabase.table(table).select("*").range(offset, offset + 999).execute()
        all_data.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return all_data

print("Fetching data from Supabase...")
all_students = fetch_all("school_students")
all_enrollments = fetch_all("school_student_enrollments")
all_schools = fetch_all("school_schools")
all_scores = fetch_all("school_student_scores")

print(f"  Students: {len(all_students)}")
print(f"  Enrollments: {len(all_enrollments)}")
print(f"  Schools: {len(all_schools)}")
print(f"  Scores: {len(all_scores)}")

# Create lookup maps
school_map = {s['id']: s['name'] for s in all_schools}

# Build student -> school mapping (via enrollments)
# Note: A student can be enrolled in multiple schools
student_schools = {}
for e in all_enrollments:
    sid = e['student_id']
    school_id = e['school_id']
    school_name = school_map.get(school_id)
    if sid not in student_schools:
        student_schools[sid] = []
    student_schools[sid].append(school_name)

# Build student -> scores mapping
student_scores = {}
for sc in all_scores:
    sid = sc['student_id']
    if sid not in student_scores:
        student_scores[sid] = []
    student_scores[sid].append(sc['score'])

print("\n" + "="*60)
print("Finding each target student...")
print("="*60)

results = []

for first, last, school_hint, dob_hint in students:
    full_name = f"{first} {last}"
    
    # Find matching students
    matches = []
    for s in all_students:
        if s['first_name'] == first and s['last_name'] == last:
            # Check school hint
            if school_hint:
                schools_for_student = student_schools.get(s['id'], [])
                if school_hint not in schools_for_student:
                    continue
            # Check DOB hint (column is 'birth_date' not 'date_of_birth')
            if dob_hint:
                if s.get('birth_date') != dob_hint:
                    continue
            matches.append(s)
    
    if len(matches) == 0:
        print(f"WARNING: No match for {full_name} (school={school_hint}, dob={dob_hint})")
        continue
    elif len(matches) > 1:
        print(f"WARNING: Multiple matches for {full_name}: {[m['id'] for m in matches]}")
        # If we have DOB hint, we should have exactly one - something's wrong
        if dob_hint:
            print(f"  DOB hint was: {dob_hint}")
            for m in matches:
                print(f"    {m['id']}: DOB={m.get('birth_date')}")
        continue
    
    student = matches[0]
    student_id = student['id']
    
    # Get school name
    if school_hint:
        school_name = school_hint
    else:
        schools = student_schools.get(student_id, [])
        if len(schools) == 1:
            school_name = schools[0]
        else:
            print(f"WARNING: {full_name} has multiple schools: {schools}")
            school_name = schools[0] if schools else "Unknown"
    
    # Get test scores
    scores = student_scores.get(student_id, [])
    test_count = len(scores)
    highest_score = max(scores) if scores else 0.0
    
    # Round to 2 decimal places
    highest_score = round(highest_score, 2)
    
    results.append({
        'first': first,
        'last': last,
        'school': school_name,
        'count': test_count,
        'highest': highest_score,
    })
    
    print(f"  {full_name}: {school_name}, {test_count} tests, highest={highest_score:.2f}")

# Sort by highest score (DESC), then by last name, then by first name
results.sort(key=lambda x: (-x['highest'], x['last'], x['first']))

print("\n" + "="*60)
print("SORTED RESULTS:")
print("="*60)

# Format output
output_parts = []
for r in results:
    entry = f"{r['first']} {r['last']}: {r['school']}, {r['count']}, {r['highest']:.2f}"
    output_parts.append(entry)
    print(f"  {entry}")

my_answer = "; ".join(output_parts)

print("\n" + "="*60)
print("MY COMPUTED ANSWER:")
print("="*60)
print(my_answer)

# Expected answer from JSON
expected = "Carter Clark: Jefferson Prep, 8, 199.83; Emily Torres: Kennedy High, 8, 199.65; Lily Clark: Kennedy High, 8, 197.29; Sebastian Torres: Jefferson Prep, 7, 196.81; Joseph Williams: Washington Academy, 6, 183.56; Michael Gonzalez: Adams Institute, 3, 171.89; Lillian Gonzalez: Hamilton High, 6, 171.52; Ava Williams: Monroe Secondary, 7, 171.07; Ella Nguyen: Adams Institute, 8, 168.90; Joseph Lopez: Madison Prep, 8, 168.45; Sophia Young: Monroe Secondary, 4, 167.47; Chloe Rivera: Madison Prep, 8, 167.11; Sophia Torres: Franklin Academy, 4, 161.79; Ethan King: Taylor High, 8, 159.68; Ethan Ramirez: Franklin Academy, 5, 159.34; Carter Hernandez: Monroe Secondary, 5, 158.95; Abigail Johnson: Taylor High, 3, 150.57; Hannah Scott: Madison Prep, 7, 127.20; Evelyn Anderson: Monroe Secondary, 6, 117.37; Lily Campbell: Jackson High School, 5, 109.49; Jack Lee: Monroe Secondary, 3, 107.76; Victoria Hall: Jefferson Prep, 6, 95.80; Daniel Lewis: Taylor High, 5, 94.45; Carter Brown: Monroe Secondary, 4, 91.19; John Allen: Jackson High School, 3, 72.77"

print("\n" + "="*60)
print("EXPECTED ANSWER:")
print("="*60)
print(expected)

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)

if my_answer == expected:
    print("✅ EXACT MATCH!")
else:
    print("❌ MISMATCH!")
    # Find differences
    my_parts = my_answer.split("; ")
    exp_parts = expected.split("; ")
    
    print(f"\nMy count: {len(my_parts)}, Expected count: {len(exp_parts)}")
    
    for i, (mine, exp) in enumerate(zip(my_parts, exp_parts)):
        if mine != exp:
            print(f"\nDiff at position {i}:")
            print(f"  Mine:     {mine}")
            print(f"  Expected: {exp}")

