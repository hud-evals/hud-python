#!/usr/bin/env python3
"""Check specific employees for the 4 sections"""

import re

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

health = sections.get("Has health insurance", set())
k401 = sections.get("Enrolled in 401k", set())
wellness = sections.get("Participates in wellness program", set())
emergency = sections.get("Emergency response trained", set())

# Check employees from BOTH my answer and expected answer
employees_to_check = [
    # My calculated answer
    'EMP-10067F', 'EMP-10091D', 'EMP-10400A', 'EMP-10448W', 'EMP-10567L', 'EMP-10802M',
    # Expected answer
    'EMP-10292W', 'EMP-10329H', 'EMP-10668I', 'EMP-10696K'
]

print("Checking each employee in all 4 sections:")
print("-" * 80)
print(f"{'Employee':<12} {'Health':<8} {'401k':<8} {'Wellness':<10} {'Emergency':<10} {'ALL 4?'}")
print("-" * 80)

for emp in sorted(set(employees_to_check)):
    in_health = emp in health
    in_401k = emp in k401
    in_wellness = emp in wellness
    in_emergency = emp in emergency
    all_four = in_health and in_401k and in_wellness and in_emergency
    
    h = "✓" if in_health else "✗"
    k = "✓" if in_401k else "✗"
    w = "✓" if in_wellness else "✗"
    e = "✓" if in_emergency else "✗"
    a = "YES" if all_four else "NO"
    
    print(f"{emp:<12} {h:<8} {k:<8} {w:<10} {e:<10} {a}")

print()
print("CONCLUSION:")
print("My calculated intersection:", sorted(health & k401 & wellness & emergency))

