#!/usr/bin/env python3
"""Triple-check the bulk retrieval task answers."""

import os
import json
from dotenv import load_dotenv
from supabase import create_client
from collections import defaultdict

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_all(table, filters=None):
    """Fetch all rows from a table with pagination."""
    all_data = []
    offset = 0
    while True:
        query = supabase.table(table).select("*")
        if filters:
            for k, v in filters.items():
                query = query.eq(k, v)
        result = query.range(offset, offset + 999).execute()
        all_data.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
    return all_data

print("=" * 70)
print("LOADING ALL DATA FROM DATABASE")
print("=" * 70)

all_students = fetch_all("school_students")
all_enrollments = fetch_all("school_student_enrollments")
all_memberships = fetch_all("school_club_memberships")
all_scores = fetch_all("school_student_scores")
schools = supabase.table("school_schools").select("*").execute()

print(f"Students: {len(all_students)}")
print(f"Enrollments: {len(all_enrollments)}")
print(f"Memberships: {len(all_memberships)}")
print(f"Scores: {len(all_scores)}")
print(f"Schools: {len(schools.data)}")

# Build lookups
student_by_name = {}
for s in all_students:
    key = (s["first_name"], s["last_name"])
    if key not in student_by_name:
        student_by_name[key] = []
    student_by_name[key].append(s)

school_names = {s["id"]: s["name"] for s in schools.data}

student_to_school = {}
for e in all_enrollments:
    if e["student_id"] not in student_to_school:
        student_to_school[e["student_id"]] = e["school_id"]

student_math_scores = defaultdict(list)
for s in all_scores:
    if s.get("subject") == "Mathematics" and s.get("score") is not None:
        student_math_scores[s["student_id"]].append(float(s["score"]))

student_club_count = defaultdict(int)
for m in all_memberships:
    if m.get("is_active"):
        student_club_count[m["student_id"]] += 1

# Load tasks
with open("/home/rs/projects/mcp-multitools/task_jsons/rs_bulk_retrieval.json") as f:
    tasks = json.load(f)

print(f"\nLoaded {len(tasks)} tasks to verify")

# ============================================================
# TASK 1: task-bulk-student-info-50
# ============================================================
print("\n" + "=" * 70)
print("VERIFYING TASK 1: task-bulk-student-info-50")
print("=" * 70)

task1 = tasks[0]
expected_answer1 = task1["evaluate_tool"]["arguments"]["exact_values"]["answer"]

# Parse student names from prompt
prompt1 = task1["prompt"]
names_start = prompt1.find("Student names: ") + len("Student names: ")
names_end = prompt1.find("\n\nIMPORTANT:")
names_str = prompt1[names_start:names_end]
student_names_1 = [n.strip() for n in names_str.split(", ")]

print(f"Students in task: {len(student_names_1)}")

# Recalculate answer
# Format: "FirstName LastName: SchoolName, MathAvg, GPA"
# Sort by: Math score DESC, then last name, first name

task1_data = []
missing_students = []
for full_name in student_names_1:
    parts = full_name.split(" ", 1)
    fname, lname = parts[0], parts[1]
    
    candidates = student_by_name.get((fname, lname), [])
    
    # Find the one with math scores and school
    found = None
    for c in candidates:
        sid = c["id"]
        if sid in student_to_school and sid in student_math_scores:
            found = c
            break
    
    if not found:
        missing_students.append(full_name)
        continue
    
    sid = found["id"]
    school_id = student_to_school[sid]
    school_name = school_names.get(school_id, "Unknown")
    math_avg = round(sum(student_math_scores[sid]) / len(student_math_scores[sid]), 2)
    gpa = round(float(found.get("gpa", 0)), 2)
    
    task1_data.append({
        "first_name": fname,
        "last_name": lname,
        "school_name": school_name,
        "math_avg": math_avg,
        "gpa": gpa
    })

if missing_students:
    print(f"WARNING: Missing students: {missing_students}")

# Sort: math_avg DESC, then last_name ASC, first_name ASC
task1_data.sort(key=lambda x: (-x["math_avg"], x["last_name"], x["first_name"]))

# Build answer
recalc_answer1 = "; ".join([
    f"{d['first_name']} {d['last_name']}: {d['school_name']}, {d['math_avg']}, {d['gpa']}"
    for d in task1_data
])

print(f"\nExpected answer length: {len(expected_answer1)}")
print(f"Recalculated answer length: {len(recalc_answer1)}")

if expected_answer1 == recalc_answer1:
    print("✅ TASK 1 VERIFIED - Answers match exactly!")
else:
    print("❌ TASK 1 MISMATCH!")
    print(f"\nExpected first 300 chars:\n{expected_answer1[:300]}")
    print(f"\nRecalculated first 300 chars:\n{recalc_answer1[:300]}")
    
    # Find first difference
    for i, (e, r) in enumerate(zip(expected_answer1, recalc_answer1)):
        if e != r:
            print(f"\nFirst difference at position {i}:")
            print(f"  Expected: ...{expected_answer1[max(0,i-20):i+20]}...")
            print(f"  Recalc:   ...{recalc_answer1[max(0,i-20):i+20]}...")
            break

# ============================================================
# TASK 2: task-bulk-student-info-100-birthsort
# ============================================================
print("\n" + "=" * 70)
print("VERIFYING TASK 2: task-bulk-student-info-100-birthsort")
print("=" * 70)

task2 = tasks[1]
expected_answer2 = task2["evaluate_tool"]["arguments"]["exact_values"]["answer"]

# Parse student names from prompt
prompt2 = task2["prompt"]
names_start = prompt2.find("Student names: ") + len("Student names: ")
names_end = prompt2.find("\n\nIMPORTANT")
names_str = prompt2[names_start:names_end]
student_names_2 = [n.strip() for n in names_str.split(", ")]

print(f"Students in task: {len(student_names_2)}")

# Recalculate answer
# Format: "FirstName LastName: SchoolName, GPA, ClubCount"
# Sort by: birth_date ASC (oldest first), then last name, first name

task2_data = []
missing_students = []
for full_name in student_names_2:
    parts = full_name.split(" ", 1)
    fname, lname = parts[0], parts[1]
    
    candidates = student_by_name.get((fname, lname), [])
    
    # Find the one with birth_date, school, and clubs
    found = None
    for c in candidates:
        sid = c["id"]
        if (c.get("birth_date") and 
            sid in student_to_school and 
            student_club_count[sid] >= 1 and
            c.get("gpa")):
            found = c
            break
    
    if not found:
        missing_students.append(full_name)
        continue
    
    sid = found["id"]
    school_id = student_to_school[sid]
    school_name = school_names.get(school_id, "Unknown")
    gpa = round(float(found.get("gpa", 0)), 2)
    club_count = student_club_count[sid]
    birth_date = found["birth_date"]
    
    task2_data.append({
        "first_name": fname,
        "last_name": lname,
        "birth_date": birth_date,
        "school_name": school_name,
        "gpa": gpa,
        "club_count": club_count
    })

if missing_students:
    print(f"WARNING: Missing students: {missing_students}")

# Sort: birth_date ASC, then last_name ASC, first_name ASC
task2_data.sort(key=lambda x: (x["birth_date"], x["last_name"], x["first_name"]))

# Build answer (without birth_date)
recalc_answer2 = "; ".join([
    f"{d['first_name']} {d['last_name']}: {d['school_name']}, {d['gpa']}, {d['club_count']}"
    for d in task2_data
])

print(f"\nExpected answer length: {len(expected_answer2)}")
print(f"Recalculated answer length: {len(recalc_answer2)}")

if expected_answer2 == recalc_answer2:
    print("✅ TASK 2 VERIFIED - Answers match exactly!")
else:
    print("❌ TASK 2 MISMATCH!")
    print(f"\nExpected first 300 chars:\n{expected_answer2[:300]}")
    print(f"\nRecalculated first 300 chars:\n{recalc_answer2[:300]}")
    
    # Find first difference
    for i, (e, r) in enumerate(zip(expected_answer2, recalc_answer2)):
        if e != r:
            print(f"\nFirst difference at position {i}:")
            print(f"  Expected: ...{expected_answer2[max(0,i-20):i+20]}...")
            print(f"  Recalc:   ...{recalc_answer2[max(0,i-20):i+20]}...")
            break

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

