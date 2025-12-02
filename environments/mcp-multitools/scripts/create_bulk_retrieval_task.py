#!/usr/bin/env python3
"""Create a bulk student info retrieval task."""

import os
import json
import random
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch students with pagination
print("Fetching students...")
all_students = []
offset = 0
while True:
    result = supabase.table("school_students").select("*").range(offset, offset + 999).execute()
    all_students.extend(result.data)
    if len(result.data) < 1000:
        break
    offset += 1000
print(f"Total students: {len(all_students)}")

# Fetch enrollments
print("Fetching enrollments...")
all_enrollments = []
offset = 0
while True:
    result = supabase.table("school_student_enrollments").select("*").range(offset, offset + 999).execute()
    all_enrollments.extend(result.data)
    if len(result.data) < 1000:
        break
    offset += 1000

# Fetch schools
schools = supabase.table("school_schools").select("*").execute()
school_names = {s["id"]: s["name"] for s in schools.data}

# Fetch scores (just Mathematics)
print("Fetching Math scores...")
all_scores = []
offset = 0
while True:
    result = supabase.table("school_student_scores").select("*").eq("subject", "Mathematics").range(offset, offset + 999).execute()
    all_scores.extend(result.data)
    if len(result.data) < 1000:
        break
    offset += 1000

# Build student -> school mapping (first enrollment)
student_to_school = {}
for e in all_enrollments:
    if e["student_id"] not in student_to_school:
        student_to_school[e["student_id"]] = e["school_id"]

# Build student -> avg math score
from collections import defaultdict
student_math_scores = defaultdict(list)
for s in all_scores:
    student_math_scores[s["student_id"]].append(float(s["score"]))

student_avg_math = {sid: sum(scores)/len(scores) for sid, scores in student_math_scores.items()}

# Pick 50 random students who have both school and math score
eligible_students = [
    s for s in all_students 
    if s["id"] in student_to_school and s["id"] in student_avg_math
]
random.seed(42)
selected_students = random.sample(eligible_students, min(50, len(eligible_students)))

print(f"Selected {len(selected_students)} students with school and math scores")

# Build the answer
# Format: "firstname lastname: school_name, math_avg, gpa"
# Sorted by: math score DESC, then alphabetically by last name, first name

student_data = []
for s in selected_students:
    sid = s["id"]
    fname = s["first_name"]
    lname = s["last_name"]
    school_id = student_to_school.get(sid)
    school_name = school_names.get(school_id, "Unknown")
    math_avg = student_avg_math.get(sid, 0)
    gpa = s.get("gpa") or 0
    
    student_data.append({
        "first_name": fname,
        "last_name": lname,
        "school_name": school_name,
        "math_avg": round(math_avg, 2),
        "gpa": round(float(gpa), 2) if gpa else 0.0
    })

# Sort by math_avg DESC, then last_name ASC, then first_name ASC
student_data.sort(key=lambda x: (-x["math_avg"], x["last_name"], x["first_name"]))

# Build the answer string
answer_parts = []
for sd in student_data:
    part = f"{sd['first_name']} {sd['last_name']}: {sd['school_name']}, {sd['math_avg']}, {sd['gpa']}"
    answer_parts.append(part)

answer_string = "; ".join(answer_parts)

# Build the student list for the prompt (just names, alphabetically for the prompt)
prompt_students = sorted(selected_students, key=lambda s: (s["last_name"], s["first_name"]))
student_names_list = ", ".join([f"{s['first_name']} {s['last_name']}" for s in prompt_students])

print(f"\nStudent names for prompt ({len(prompt_students)} students):")
print(student_names_list[:500] + "...")

print(f"\nExpected answer (first 500 chars):")
print(answer_string[:500] + "...")

# Create the task JSON
task = {
    "id": "task-bulk-student-info-50",
    "prompt": f"""You are given a list of 50 student names. For EACH student, retrieve:
1. Their school name
2. Their average Mathematics test score (rounded to 2 decimal places)
3. Their GPA (rounded to 2 decimal places)

Student names: {student_names_list}

IMPORTANT: Format your answer as a semicolon-separated list, with each entry as:
"FirstName LastName: SchoolName, MathAvg, GPA"

Sort the results by Mathematics score (highest first). If scores are tied, sort alphabetically by last name, then first name.

Example format: "Alice Smith: Lincoln High School, 95.50, 3.80; Bob Jones: Jefferson Prep, 92.00, 3.50"

Database: supabase_execute_sql (project_id: 'qgubrffqclddzehyyvbt')
Tables have 'school_' prefix. Relevant tables: school_students, school_student_enrollments, school_schools, school_student_scores.

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

# Write to file
output_path = "/home/rs/projects/mcp-multitools/task_jsons/rs_bulk_retrieval.json"
with open(output_path, "w") as f:
    json.dump([task], f, indent=2)

print(f"\n\nTask written to {output_path}")
print(f"Answer length: {len(answer_string)} characters")

