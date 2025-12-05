#!/usr/bin/env python3
"""Debug the missing students."""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Students with DOB hints that weren't found
problem_students = [
    ("Michael", "Gonzalez", "Adams Institute", "2008-08-25"),
    ("Evelyn", "Anderson", "Monroe Secondary", "2009-09-10"),
    ("Ava", "Williams", "Monroe Secondary", "2005-03-19"),
    ("Sebastian", "Torres", "Jefferson Prep", "2012-07-14"),
]

def fetch_all(table):
    all_data = []
    offset = 0
    while True:
        result = supabase.table(table).select("*").range(offset, offset + 999).execute()
        all_data.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return all_data

all_students = fetch_all("school_students")
all_enrollments = fetch_all("school_student_enrollments")
all_schools = fetch_all("school_schools")

school_map = {s['id']: s['name'] for s in all_schools}

# Build student -> schools mapping
student_schools = {}
for e in all_enrollments:
    sid = e['student_id']
    school_id = e['school_id']
    school_name = school_map.get(school_id)
    if sid not in student_schools:
        student_schools[sid] = []
    student_schools[sid].append(school_name)

print("="*80)
print("Debugging missing students with DOB hints")
print("="*80)

for first, last, school_hint, dob_hint in problem_students:
    full_name = f"{first} {last}"
    print(f"\n--- {full_name} (school={school_hint}, dob={dob_hint}) ---")
    
    # Find ALL students with this name
    name_matches = [s for s in all_students if s['first_name'] == first and s['last_name'] == last]
    print(f"  Students with name '{full_name}': {len(name_matches)}")
    
    for s in name_matches:
        sid = s['id']
        dob = s.get('date_of_birth')
        schools = student_schools.get(sid, [])
        print(f"    ID: {sid}, DOB: {dob}, Schools: {schools}")
        
        # Check if this student matches school hint
        school_match = school_hint in schools if schools else False
        dob_match = dob == dob_hint if dob else False
        
        print(f"      -> School match: {school_match}, DOB match: {dob_match}")

print("\n" + "="*80)
print("Checking DOB format in database")
print("="*80)

# Sample some students with DOBs
students_with_dob = [s for s in all_students if s.get('date_of_birth')][:5]
for s in students_with_dob:
    print(f"  {s['first_name']} {s['last_name']}: DOB={s['date_of_birth']} (type: {type(s['date_of_birth']).__name__})")

