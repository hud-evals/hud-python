#!/usr/bin/env python3
"""
FINAL verification of ALL employee_attributes tasks against Supabase.
"""

import os
import re
import json
from dotenv import load_dotenv
from supabase import create_client
from collections import defaultdict

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

print("Loading database...")
employees = fetch_all("employees")
departments = fetch_all("departments")
shifts = fetch_all("shifts")

emp_by_id = {e["id"]: e for e in employees}
dept_names = {d["id"]: d["name"] for d in departments}
shift_counts = defaultdict(int)
for s in shifts:
    shift_counts[s["employee_id"]] += 1

def parse_sections(filepath):
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

sections = parse_sections('/home/rs/projects/hud-python/environments/mcp-multitools/documents/employee_attributes.txt')

# Load tasks
with open('/home/rs/projects/hud-python/environments/mcp-multitools/task_jsons/final_tasks.json', 'r') as f:
    tasks = json.load(f)

print("="*80)
print("VERIFYING ALL EMPLOYEE_ATTRIBUTES TASKS")
print("="*80)

issues = []

# =============================================================================
# 1. task-count-avg-salary (Certification analysis)
# =============================================================================
print("\n" + "-"*80)
print("1. task-count-avg-salary")
print("-"*80)

cert_sections = {
    "Cloud certified": sections.get("Cloud certified", set()),
    "Remote work approved": sections.get("Remote work approved", set()),
    "First aid certified": sections.get("First aid certified", set()),
    "Certified Scrum Master": sections.get("Certified Scrum Master", set()),
    "Certified PMP": sections.get("Certified PMP", set()),
    "Completed leadership training": sections.get("Completed leadership training", set()),
}

results = []
for name, emp_ids in cert_sections.items():
    salaries = [emp_by_id[eid]["salary"] for eid in emp_ids if eid in emp_by_id]
    count = len(salaries)
    avg = sum(salaries) / count if count > 0 else 0
    results.append(f"{name}: {count} employees, ${avg:.2f} avg")
    print(f"  {name}: {count} employees, ${avg:.2f} avg")

calculated = "; ".join(results)

# Get expected
for t in tasks:
    if t.get("id") == "task-count-avg-salary":
        expected = t["evaluate_tool"]["arguments"]["exact_values"]["answer"]
        break

print(f"\nCalculated: {calculated}")
print(f"Expected:   {expected}")
if calculated == expected:
    print("✅ MATCH")
else:
    print("❌ MISMATCH")
    issues.append("task-count-avg-salary")

# =============================================================================
# 2. task-four-ands
# =============================================================================
print("\n" + "-"*80)
print("2. task-four-ands")
print("-"*80)

health = sections.get("Has health insurance", set())
k401 = sections.get("Enrolled in 401k", set())
wellness = sections.get("Participates in wellness program", set())
emergency = sections.get("Emergency response trained", set())

intersection = health & k401 & wellness & emergency
print(f"  Intersection: {len(intersection)} employees")

results = []
for emp_id in sorted(intersection):
    emp = emp_by_id.get(emp_id)
    if emp:
        dept = dept_names.get(emp["department_id"], "Unknown")
        results.append(f"{emp_id}: {dept}, {emp['position']}")

calculated = "; ".join(results)

for t in tasks:
    if t.get("id") == "task-four-ands":
        expected = t["evaluate_tool"]["arguments"]["exact_values"]["answer"]
        break

print(f"\nCalculated: {calculated}")
print(f"Expected:   {expected}")
if calculated == expected:
    print("✅ MATCH")
else:
    print("❌ MISMATCH")
    issues.append("task-four-ands")

# =============================================================================
# 3. task-shifts-count
# =============================================================================
print("\n" + "-"*80)
print("3. task-shifts-count")
print("-"*80)

weekend = sections.get("Available for weekend oncall", set())
emergency = sections.get("Emergency response trained", set())
intersection = weekend & emergency
print(f"  Intersection: {len(intersection)} employees")

results = []
for emp_id in sorted(intersection):
    emp = emp_by_id.get(emp_id)
    if emp:
        name = f"{emp['first_name']} {emp['last_name']}"
        count = shift_counts.get(emp_id, 0)
        results.append(f"{emp_id}: {name}, {count} shifts")

calculated = "; ".join(results)

for t in tasks:
    if t.get("id") == "task-shifts-count":
        expected = t["evaluate_tool"]["arguments"]["exact_values"]["answer"]
        break

print(f"\nCalculated: {calculated}")
print(f"Expected:   {expected}")
if calculated == expected:
    print("✅ MATCH")
else:
    print("❌ MISMATCH")
    issues.append("task-shifts-count")

# =============================================================================
# 4. task-language-safety-dept
# =============================================================================
print("\n" + "-"*80)
print("4. task-language-safety-dept")
print("-"*80)

spanish = sections.get("Speaks Spanish", set())
french = sections.get("Speaks French", set())
firstaid = sections.get("First aid certified", set())

candidates = (spanish | french) & firstaid
target_depts = {"Engineering", "Marketing", "Sales", "Customer Support"}

results = []
for emp_id in candidates:
    emp = emp_by_id.get(emp_id)
    if emp:
        dept = dept_names.get(emp["department_id"], "Unknown")
        if dept in target_depts:
            name = f"{emp['first_name']} {emp['last_name']}"
            results.append(name)

results.sort()
calculated = "; ".join(results)

for t in tasks:
    if t.get("id") == "task-language-safety-dept":
        expected = t["evaluate_tool"]["arguments"]["exact_values"]["answer"]
        break

print(f"\nCalculated: {calculated}")
print(f"Expected:   {expected}")
if calculated == expected:
    print("✅ MATCH")
else:
    print("❌ MISMATCH")
    issues.append("task-language-safety-dept")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

if issues:
    print(f"\n❌ {len(issues)} TASKS HAVE WRONG EXPECTED ANSWERS:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("\n✅ ALL TASKS VERIFIED CORRECT!")

