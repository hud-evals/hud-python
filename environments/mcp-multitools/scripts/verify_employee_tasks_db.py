#!/usr/bin/env python3
"""
Verify employee tasks by querying the actual Supabase database.
"""

import os
import re
from dotenv import load_dotenv
from supabase import create_client
from collections import defaultdict

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

print(f"Connecting to Supabase...")
print(f"URL: {SUPABASE_URL}")
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

# Fetch employees, departments, shifts
employees = fetch_all("employees")
departments = fetch_all("departments")
shifts = fetch_all("shifts")

print(f"Employees: {len(employees)}")
print(f"Departments: {len(departments)}")
print(f"Shifts: {len(shifts)}")

# Build lookups
dept_names = {d["id"]: d["name"] for d in departments}
emp_by_id = {e["id"]: e for e in employees}

# Count shifts per employee
shift_counts = defaultdict(int)
for s in shifts:
    shift_counts[s["employee_id"]] += 1

# =============================================================================
# Parse employee_attributes.txt
# =============================================================================

def parse_all_sections(filepath):
    """Parse all sections from employee_attributes.txt into a dict."""
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

# =============================================================================
# TASK 1: task-shifts-count
# =============================================================================
print("\n" + "=" * 70)
print("TASK 1: task-shifts-count")
print("=" * 70)

# Filter: "Available for weekend oncall" AND "Emergency response trained"
weekend = sections.get("Available for weekend oncall", set())
emergency = sections.get("Emergency response trained", set())
intersection = weekend & emergency

print(f"Available for weekend oncall: {len(weekend)}")
print(f"Emergency response trained: {len(emergency)}")
print(f"Intersection: {len(intersection)} employees")
print()

# Get shift counts for each employee in intersection
results = []
for emp_id in sorted(intersection):
    emp = emp_by_id.get(emp_id)
    if emp:
        name = f"{emp['first_name']} {emp['last_name']}"
        count = shift_counts.get(emp_id, 0)
        results.append((emp_id, name, count))
    else:
        print(f"WARNING: {emp_id} not found in database!")

print("CORRECT ANSWER (from database):")
answer_parts = []
for emp_id, name, count in results:
    answer_parts.append(f"{emp_id}: {name}, {count} shifts")
    print(f"  {emp_id}: {name}, {count} shifts")

print()
print("FORMATTED ANSWER:")
answer = "; ".join(answer_parts)
print(answer)

# =============================================================================
# TASK 2: task-language-safety-dept  
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2: task-language-safety-dept")
print("=" * 70)

# Filter: ("Speaks Spanish" OR "Speaks French") AND "First aid certified"
spanish = sections.get("Speaks Spanish", set())
french = sections.get("Speaks French", set())
firstaid = sections.get("First aid certified", set())

bilingual_firstaid = (spanish | french) & firstaid
print(f"Speaks Spanish: {len(spanish)}")
print(f"Speaks French: {len(french)}")
print(f"First aid certified: {len(firstaid)}")
print(f"(Spanish OR French) AND First aid: {len(bilingual_firstaid)}")
print()

# Filter by department: Engineering, Marketing, Sales, Customer Support
target_depts = {"Engineering", "Marketing", "Sales", "Customer Support"}

results2 = []
for emp_id in sorted(bilingual_firstaid):
    emp = emp_by_id.get(emp_id)
    if emp:
        dept_id = emp.get("department_id")
        dept_name = dept_names.get(dept_id, "Unknown")
        if dept_name in target_depts:
            name = f"{emp['first_name']} {emp['last_name']}"
            results2.append((name, dept_name, emp_id))
    else:
        print(f"WARNING: {emp_id} not found in database!")

# Sort alphabetically by name
results2.sort(key=lambda x: x[0])

print(f"After department filter: {len(results2)} employees")
print()
print("CORRECT ANSWER (from database):")
for name, dept, emp_id in results2:
    print(f"  {name} ({emp_id}): {dept}")

print()
print("FORMATTED ANSWER:")
answer2 = "; ".join([name for name, _, _ in results2])
print(answer2)

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

