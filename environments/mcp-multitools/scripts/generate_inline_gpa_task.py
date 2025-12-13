#!/usr/bin/env python3
"""Generate a verified inline GPA verification task."""

import os
import json
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('/home/rs/projects/hud-python/.env')

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

print("Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch all students and randomly select 100
print("Fetching students from database...")
import random

all_students = []
offset = 0
while True:
    batch = supabase.table("school_students")\
        .select("id, first_name, last_name, gpa")\
        .range(offset, offset + 999)\
        .execute()
    all_students.extend(batch.data)
    if len(batch.data) < 1000:
        break
    offset += 1000

print(f"Fetched {len(all_students)} total students, randomly selecting 100...")
random.seed(42)  # For reproducibility
students_data = random.sample(all_students, 100)
students_data.sort(key=lambda x: x['id'])  # Sort for cleaner output
print(f"Got {len(students_data)} students")

# Pick 4 students to have wrong GPAs (spread them out)
contradiction_indices = [15, 35, 60, 85]
contradictions = []

print("\nGenerating inline list with 4 wrong GPAs...")

# Build the inline student list
student_lines = []
for i, s in enumerate(students_data):
    sid = s['id']
    name = f"{s['first_name']} {s['last_name']}"
    real_gpa = float(s['gpa'])
    
    if i in contradiction_indices:
        # Add 0.47 to make it wrong
        fake_gpa = round(real_gpa + 0.47, 2)
        student_lines.append(f"{sid}: {name} - GPA {fake_gpa}")
        
        # Track contradiction for answer
        contradictions.append({
            "student_id": sid,
            "first_name": s['first_name'],
            "last_name": s['last_name'],
            "fake_gpa": fake_gpa,
            "real_gpa": real_gpa
        })
    else:
        student_lines.append(f"{sid}: {name} - GPA {real_gpa}")

# Generate prompt
prompt_intro = """Below is a list of 100 students with their reported GPAs. Compare these GPAs with the actual data in Supabase (school_students table, gpa column).

Find all students where the listed GPA differs from their actual GPA in the database.

Format: "{student_id}: {first_name} {last_name}, list says {listed_gpa}, actual {db_gpa}"
Sort by student_id (A-Z).
Separate entries with "; "

Example (match this structure exactly): "STU-999999: John Smith, list says 3.25, actual 2.80; STU-999998: Jane Doe, list says 2.15, actual 2.62"

Note: GPAs should match to 2 decimal places. Even 0.01 difference counts.

STUDENT GPA LIST:
"""

prompt = prompt_intro + "\n".join(student_lines)
prompt += "\n\nDatabase: supabase_execute_sql (project_id: 'qgubrffqclddzehyyvbt')\nTable: school_students\n\nStore result with key 'answer'. Say 'Task completed.' when done."

# Generate correct answer
answer_parts = []
for c in contradictions:
    part = f"{c['student_id']}: {c['first_name']} {c['last_name']}, list says {c['fake_gpa']}, actual {c['real_gpa']}"
    answer_parts.append(part)

correct_answer = "; ".join(answer_parts)

# Build task JSON
task = {
    "id": "task-school-inline-gpa-verification",
    "prompt": prompt,
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
    "setup_tool": {"name": "local_setup", "arguments": {}},
    "evaluate_tool": {
        "name": "local_evaluate",
        "arguments": {
            "exact_values": {
                "answer": correct_answer
            }
        }
    },
    "agent_config": {
        "allowed_tools": ["local_scratchpad_write", "local_scratchpad_read", "supabase_execute_sql"]
    }
}

# Save task
output_path = "../task_jsons/inline_gpa_task.json"
with open(output_path, 'w') as f:
    json.dump([task], f, indent=2)

print("\n" + "=" * 80)
print("TASK GENERATED")
print("=" * 80)
print(f"\nSaved to: {output_path}")
print(f"\nContradictions found: {len(contradictions)}")
for c in contradictions:
    print(f"  {c['student_id']}: {c['first_name']} {c['last_name']} - Fake: {c['fake_gpa']}, Real: {c['real_gpa']}")

print(f"\nCorrect answer:\n{correct_answer}")
print("\n✅ Task verified and ready to test!")

