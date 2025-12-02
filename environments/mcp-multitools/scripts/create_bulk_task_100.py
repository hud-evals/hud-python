#!/usr/bin/env python3
"""Create a bulk student info retrieval task - 100 students, sorted by birth date."""

import os
import json
import random
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

print("Fetching data...")
all_students = fetch_all("school_students")
all_enrollments = fetch_all("school_student_enrollments")
all_memberships = fetch_all("school_club_memberships")
schools = supabase.table("school_schools").select("*").execute()

print(f"Students: {len(all_students)}, Enrollments: {len(all_enrollments)}, Memberships: {len(all_memberships)}")

school_names = {s["id"]: s["name"] for s in schools.data}

# Build student -> school mapping (first enrollment)
student_to_school = {}
for e in all_enrollments:
    if e["student_id"] not in student_to_school:
        student_to_school[e["student_id"]] = e["school_id"]

# Build student -> active club count
student_club_count = defaultdict(int)
for m in all_memberships:
    if m.get("is_active"):
        student_club_count[m["student_id"]] += 1

# Pick 100 students who have:
# - birth_date
# - school enrollment
# - at least 1 active club membership
# - GPA

eligible_students = [
    s for s in all_students 
    if s.get("birth_date") 
    and s["id"] in student_to_school 
    and student_club_count[s["id"]] >= 1
    and s.get("gpa")
]

print(f"Eligible students (have birth_date, school, clubs, GPA): {len(eligible_students)}")

random.seed(123)  # Different seed for different selection
selected_students = random.sample(eligible_students, min(100, len(eligible_students)))

print(f"Selected {len(selected_students)} students")

# Build the answer
# Sort by: birth_date (oldest first = ASC)
# Format: "firstname lastname: school_name, gpa, club_count"

student_data = []
for s in selected_students:
    sid = s["id"]
    fname = s["first_name"]
    lname = s["last_name"]
    birth_date = s["birth_date"]  # For sorting only
    school_id = student_to_school.get(sid)
    school_name = school_names.get(school_id, "Unknown")
    gpa = round(float(s.get("gpa", 0)), 2)
    club_count = student_club_count[sid]
    
    student_data.append({
        "first_name": fname,
        "last_name": lname,
        "birth_date": birth_date,
        "school_name": school_name,
        "gpa": gpa,
        "club_count": club_count
    })

# Sort by birth_date ASC (oldest first), then last_name, first_name for ties
student_data.sort(key=lambda x: (x["birth_date"], x["last_name"], x["first_name"]))

# Build the answer string (without birth_date!)
answer_parts = []
for sd in student_data:
    part = f"{sd['first_name']} {sd['last_name']}: {sd['school_name']}, {sd['gpa']}, {sd['club_count']}"
    answer_parts.append(part)

answer_string = "; ".join(answer_parts)

# Build the student list for the prompt (alphabetically for the prompt)
prompt_students = sorted(selected_students, key=lambda s: (s["last_name"], s["first_name"]))
student_names_list = ", ".join([f"{s['first_name']} {s['last_name']}" for s in prompt_students])

print(f"\nStudent names for prompt ({len(prompt_students)} students):")
print(student_names_list[:600] + "...")

print(f"\nExpected answer (first 600 chars):")
print(answer_string[:600] + "...")

print(f"\nFirst 5 entries (showing sort order by birth_date):")
for sd in student_data[:5]:
    print(f"  {sd['birth_date']}: {sd['first_name']} {sd['last_name']}")

# Create the task JSON
task = {
    "id": "task-bulk-student-info-100-birthsort",
    "prompt": f"""You are given a list of 100 student names. For EACH student, retrieve:
1. Their school name
2. Their GPA (rounded to 2 decimal places)
3. The number of clubs they are ACTIVE members of

Student names: {student_names_list}

IMPORTANT SORTING: Sort the results by birth date (OLDEST first, i.e., earliest birth date first). 
If birth dates are tied, sort alphabetically by last name, then first name.
Note: Birth date is used ONLY for sorting - do NOT include it in the output.

Format your answer as a semicolon-separated list, with each entry as:
"FirstName LastName: SchoolName, GPA, ClubCount"

Example format: "Alice Smith: Lincoln High School, 3.80, 2; Bob Jones: Jefferson Prep, 3.50, 1"

Database: supabase_execute_sql (project_id: 'qgubrffqclddzehyyvbt')
Tables have 'school_' prefix. Relevant tables: school_students (has birth_date, gpa), school_student_enrollments, school_schools, school_club_memberships (check is_active).

Store the complete formatted answer with key 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": {
        "local": {
            "command": "docker",
            "args": ["run", "--rm", "-i", "--env-file", "/home/rs/projects/hud-python/.env", "mcp-multitools:latest"]
        },
        "supabase": {
            "url": "https://mcp.supabase.com/mcp?read_only=true",
            "headers": {"Authorization": "Bearer sbp_6db19bff28d413a387bb9999cf989201887e5bdd"}
        }
    },
    "agent_config": {
        "allowed_tools": ["local_scratchpad_read", "local_scratchpad_write", "supabase_list_tables", "supabase_execute_sql"]
    },
    "setup_tool": {
        "name": "local_setup",
        "arguments": {}
    },
    "evaluate_tool": {
        "name": "local_evaluate",
        "arguments": {
            "exact_values": {
                "answer": answer_string
            }
        }
    }
}

# Add to the existing file
output_path = "/home/rs/projects/mcp-multitools/task_jsons/rs_bulk_retrieval.json"

# Read existing tasks
with open(output_path, "r") as f:
    existing_tasks = json.load(f)

# Append new task
existing_tasks.append(task)

# Write back
with open(output_path, "w") as f:
    json.dump(existing_tasks, f, indent=2)

print(f"\n\nTask appended to {output_path}")
print(f"Total tasks in file: {len(existing_tasks)}")
print(f"Answer length: {len(answer_string)} characters")

