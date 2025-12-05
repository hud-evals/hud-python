#!/usr/bin/env python3
"""
Verify specific employees and tasks.
"""

import os
import re
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
print("CHECKING FELIX FISCHER (EMP-10067F) SHIFTS")
print("=" * 70)

# Get all shifts for Felix Fischer
all_shifts = fetch_all("shifts")
felix_shifts = [s for s in all_shifts if s.get("employee_id") == "EMP-10067F"]

print(f"Total shifts in database: {len(all_shifts)}")
print(f"Felix Fischer (EMP-10067F) shifts: {len(felix_shifts)}")

if felix_shifts:
    print("\nFelix's shift details:")
    for s in felix_shifts[:10]:  # Show first 10
        print(f"  {s}")
else:
    print("\nNo shifts found for Felix Fischer")

# Also check the employee exists
employees = fetch_all("employees")
felix = [e for e in employees if e.get("id") == "EMP-10067F"]
if felix:
    print(f"\nFelix employee record: {felix[0]}")
else:
    print("\nWARNING: Felix Fischer not found in employees table!")

# =============================================================================
# Re-verify task-language-safety-dept
# =============================================================================
print("\n" + "=" * 70)
print("RE-VERIFYING task-language-safety-dept")
print("=" * 70)

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

# Filter: ("Speaks Spanish" OR "Speaks French") AND "First aid certified"
spanish = sections.get("Speaks Spanish", set())
french = sections.get("Speaks French", set())
firstaid = sections.get("First aid certified", set())

print(f"\nSpeaks Spanish ({len(spanish)} employees):")
print(f"  {sorted(spanish)}")
print(f"\nSpeaks French ({len(french)} employees):")
print(f"  {sorted(french)}")
print(f"\nFirst aid certified ({len(firstaid)} employees):")
print(f"  {sorted(firstaid)}")

bilingual = spanish | french
print(f"\n(Spanish OR French): {len(bilingual)} employees")

bilingual_firstaid = bilingual & firstaid
print(f"(Spanish OR French) AND First aid: {len(bilingual_firstaid)} employees")
print(f"  {sorted(bilingual_firstaid)}")

# Get department info from database
departments = fetch_all("departments")
dept_names = {d["id"]: d["name"] for d in departments}
emp_by_id = {e["id"]: e for e in employees}

# Filter by department
target_depts = {"Engineering", "Marketing", "Sales", "Customer Support"}

print(f"\nFiltering by departments: {target_depts}")
print()

results = []
for emp_id in sorted(bilingual_firstaid):
    emp = emp_by_id.get(emp_id)
    if emp:
        dept_id = emp.get("department_id")
        dept_name = dept_names.get(dept_id, "Unknown")
        name = f"{emp['first_name']} {emp['last_name']}"
        in_target = dept_name in target_depts
        status = "✓" if in_target else "✗"
        print(f"  {status} {emp_id}: {name} -> {dept_name}")
        if in_target:
            results.append((name, dept_name, emp_id))
    else:
        print(f"  ⚠ {emp_id}: NOT IN DATABASE")

results.sort(key=lambda x: x[0])

print(f"\nFinal count after department filter: {len(results)}")
print("\nFORMATTED ANSWER:")
answer = "; ".join([name for name, _, _ in results])
print(answer)

print("\n" + "=" * 70)
print("EXPECTED ANSWER FROM JSON:")
print("=" * 70)
expected = "Aaron Valdez; Alexander Petrov; Arthur Reyes; Bridget Yang; Charlotte Chen; Edward Zhang; Ethan Eriksson; James Chen; James Chen; Oscar Zhang; Peter Valdez; Quentin Wagner; Raymond Davis; Walter Ishikawa; William Vargas"
print(expected)

print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
if answer == expected:
    print("✅ ANSWERS MATCH EXACTLY!")
else:
    print("❌ ANSWERS DO NOT MATCH!")
    print(f"\nCalculated: {answer}")
    print(f"\nExpected:   {expected}")

