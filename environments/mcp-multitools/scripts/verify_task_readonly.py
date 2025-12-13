#!/usr/bin/env python3
"""READ-ONLY verification of school enrollment task."""

import os
import re
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('/home/rs/projects/hud-python/.env')

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

print("Connecting to Supabase (READ-ONLY)...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("=" * 80)
print("STEP 1: Parse document")
print("=" * 80)

with open('documents/school_enrollment_report.txt', 'r') as f:
    content = f.read()

pattern = r'(STU-\d+):\s+([A-Za-z]+)\s+([A-Za-z]+)\s+-\s+(.+?)(?=\n|$)'
matches = re.findall(pattern, content)

doc_data = {}
for student_id, first, last, school in matches:
    doc_data[student_id] = {
        "first_name": first,
        "last_name": last,
        "school": school.strip()
    }

print(f"Document has {len(doc_data)} students\n")

print("=" * 80)
print("STEP 2: Query Supabase for those specific students")
print("=" * 80)

student_ids = list(doc_data.keys())
print(f"Querying {len(student_ids)} students...")

# Fetch enrollments for these students
enrollments = supabase.table("school_student_enrollments")\
    .select("student_id, school_id")\
    .in_("student_id", student_ids)\
    .execute()

# Fetch schools
schools = supabase.table("school_schools").select("id, name").execute()
school_map = {s["id"]: s["name"] for s in schools.data}

# Fetch student names
students = supabase.table("school_students")\
    .select("id, first_name, last_name")\
    .in_("id", student_ids)\
    .execute()
student_map = {s["id"]: s for s in students.data}

print(f"Found {len(enrollments.data)} enrollments in database\n")

print("=" * 80)
print("STEP 3: Find mismatches")
print("=" * 80)

mismatches = []
for enroll in enrollments.data:
    sid = enroll["student_id"]
    doc_entry = doc_data.get(sid)
    
    if not doc_entry:
        continue
    
    db_school = school_map.get(enroll["school_id"], "UNKNOWN")
    doc_school = doc_entry["school"]
    
    if db_school != doc_school:
        student_info = student_map[sid]
        mismatches.append({
            "student_id": sid,
            "first_name": student_info["first_name"],
            "last_name": student_info["last_name"],
            "doc_school": doc_school,
            "db_school": db_school
        })

# Sort by student_id
mismatches.sort(key=lambda x: x["student_id"])

print(f"Found {len(mismatches)} mismatches:\n")
for m in mismatches:
    print(f"{m['student_id']}: {m['first_name']} {m['last_name']}")
    print(f"  Doc says: {m['doc_school']}")
    print(f"  DB has:   {m['db_school']}")
    print()

print("=" * 80)
print("STEP 4: Format correct answer")
print("=" * 80)

answer_parts = []
for m in mismatches:
    part = f"{m['student_id']}: {m['first_name']} {m['last_name']}, document says {m['doc_school']}, actual {m['db_school']}"
    answer_parts.append(part)

correct_answer = "; ".join(answer_parts)
print("CORRECT ANSWER:")
print(correct_answer)

print("\n" + "=" * 80)
print("STEP 5: Compare with task's expected answer")
print("=" * 80)

expected = "STU-001112: Mia Garcia, document says Kennedy High, actual Jackson High School; STU-001489: James Sanchez, document says Adams Institute, actual Hamilton High; STU-003512: Michael Hill, document says Washington Academy, actual Lincoln High School; STU-003783: Isabella Hall, document says Franklin Academy, actual Polk Prep"

print("Expected:", expected)
print()
print("Computed:", correct_answer)
print()

if correct_answer == expected:
    print("✅ TASK IS CORRECT!")
else:
    print("❌ TASK NEEDS UPDATE")
    print("\nDifferences:")
    expected_set = set(expected.split('; '))
    computed_set = set(answer_parts)
    
    missing = expected_set - computed_set
    extra = computed_set - expected_set
    
    if missing:
        print("Missing from computed:")
        for item in missing:
            print(f"  - {item}")
    
    if extra:
        print("Extra in computed:")
        for item in extra:
            print(f"  + {item}")

