#!/usr/bin/env python3
"""Verify task-four-ands (strict filter - all 4 sections)"""

import re
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_all_sections(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = {}
    lines = content.split('\n')
    current_section = None
    current_ids = []
    
    for line in lines:
        if line.startswith('SECTION') and ':' in line:
            if current_section:
                sections[current_section] = set(current_ids)
            match = re.match(r'SECTION\s+\d+:\s*(.+)', line)
            if match:
                current_section = match.group(1).strip()
                current_ids = []
        elif current_section:
            ids = re.findall(r'EMP-\d{5}[A-Z]', line)
            current_ids.extend(ids)
    
    if current_section:
        sections[current_section] = set(current_ids)
    
    return sections

sections = parse_all_sections('/home/rs/projects/hud-python/environments/mcp-multitools/documents/employee_attributes.txt')

# The 4 sections
health = sections.get("Has health insurance", set())
k401 = sections.get("Enrolled in 401k", set())
wellness = sections.get("Participates in wellness program", set())
emergency = sections.get("Emergency response trained", set())

print(f"Has health insurance: {len(health)} employees")
print(f"Enrolled in 401k: {len(k401)} employees")
print(f"Participates in wellness program: {len(wellness)} employees")
print(f"Emergency response trained: {len(emergency)} employees")

# Intersection of ALL FOUR
intersection = health & k401 & wellness & emergency

print(f"\nIntersection (ALL 4): {len(intersection)} employees")
print(f"IDs: {sorted(intersection)}")

# Get department and position from database
employees = supabase.table("employees").select("*").execute()
departments = supabase.table("departments").select("*").execute()

emp_by_id = {e["id"]: e for e in employees.data}
dept_names = {d["id"]: d["name"] for d in departments.data}

print("\nFORMATTED ANSWER:")
results = []
for emp_id in sorted(intersection):
    emp = emp_by_id.get(emp_id)
    if emp:
        dept = dept_names.get(emp["department_id"], "Unknown")
        pos = emp["position"]
        results.append(f"{emp_id}: {dept}, {pos}")

answer = "; ".join(results)
print(answer)

print("\n" + "="*70)
print("EXPECTED ANSWER:")
expected = "EMP-10067F: Operations, Director of Operations; EMP-10292W: Engineering, Junior Software Engineer; EMP-10329H: Finance, Junior Accountant; EMP-10567L: Legal, Senior Counsel; EMP-10668I: Engineering, Junior Software Engineer; EMP-10696K: Finance, Senior Accountant"
print(expected)

print("\n" + "="*70)
if answer == expected:
    print("✅ MATCH!")
else:
    print("❌ MISMATCH!")

