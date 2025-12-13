#!/usr/bin/env python3
"""Verify the department summary task."""

import os
from dotenv import load_dotenv
from supabase import create_client
from decimal import Decimal, ROUND_HALF_UP

load_dotenv('/home/rs/projects/hud-python/.env')

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("=" * 80)
print("Verifying Department Summary Task")
print("=" * 80)

# Fetch all data
print("\nFetching data...")
depts = supabase.table("departments").select("*").execute().data
employees = supabase.table("employees").select("*").execute().data
orders = supabase.table("orders").select("*").execute().data

# Fetch order items in batches
print("Fetching order items...")
all_order_items = []
offset = 0
while True:
    batch = supabase.table("order_items").select("*").range(offset, offset + 999).execute()
    all_order_items.extend(batch.data)
    if len(batch.data) < 1000:
        break
    offset += 1000

print(f"Departments: {len(depts)}")
print(f"Employees: {len(employees)}")
print(f"Orders: {len(orders)}")
print(f"Order Items: {len(all_order_items)}")

# Group employees by department
emp_by_dept = {}
for emp in employees:
    dept_id = emp['department_id']
    if dept_id not in emp_by_dept:
        emp_by_dept[dept_id] = []
    emp_by_dept[dept_id].append(emp)

# Map order items by order_id
items_by_order = {}
for item in all_order_items:
    order_id = item['order_id']
    if order_id not in items_by_order:
        items_by_order[order_id] = []
    items_by_order[order_id].append(item)

# Map employees by id for revenue calculation
emp_map = {e['id']: e for e in employees}

# Calculate stats for each department
results = []
for dept in sorted(depts, key=lambda d: d['name']):
    dept_id = dept['id']
    dept_name = dept['name']
    
    # 1. Employee count
    dept_employees = emp_by_dept.get(dept_id, [])
    emp_count = len(dept_employees)
    
    # 2. Total salary
    total_salary = sum(emp['salary'] for emp in dept_employees)
    
    # 3. Total revenue (from orders by sales reps in this dept)
    dept_emp_ids = {emp['id'] for emp in dept_employees}
    total_revenue = Decimal('0')
    
    for order in orders:
        # Skip cancelled orders
        if order['status'] == 'cancelled':
            continue
        
        # Check if sales rep belongs to this department
        sales_rep_id = order['sales_rep_id']
        if sales_rep_id not in dept_emp_ids:
            continue
        
        # Calculate order revenue
        order_items = items_by_order.get(order['id'], [])
        product_revenue = Decimal('0')
        for item in order_items:
            qty = Decimal(str(item['quantity']))
            price = Decimal(str(item['unit_price']))
            discount = Decimal(str(item['discount_percent']))
            product_revenue += qty * price * (Decimal('1') - discount / Decimal('100'))
        
        shipping = Decimal(str(order['shipping_cost']))
        order_revenue = product_revenue + shipping
        total_revenue += order_revenue
    
    # Round revenue to 2 decimal places (half up)
    total_revenue = total_revenue.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    results.append({
        "name": dept_name,
        "emp_count": emp_count,
        "total_salary": total_salary,
        "total_revenue": float(total_revenue)
    })

# Format answer
answer_parts = []
for r in results:
    part = f"{r['name']}: {r['emp_count']}, {r['total_salary']}, {r['total_revenue']:.2f}"
    answer_parts.append(part)

computed_answer = "; ".join(answer_parts)

print("\n" + "=" * 80)
print("COMPUTED ANSWER:")
print("=" * 80)
print(computed_answer)

print("\n" + "=" * 80)
print("EXPECTED FROM TASK:")
print("=" * 80)

expected = "Customer Support: 9, 683435, 0.00; Engineering: 11, 1222803, 0.00; Finance: 10, 985764, 0.00; Human Resources: 10, 808739, 0.00; Legal: 9, 1218585, 0.00; Marketing: 10, 988876, 0.00; Operations: 9, 790065, 0.00; Product Management: 9, 1236463, 0.00; Research & Development: 9, 1447542, 0.00; Sales: 11, 1340737, 2827914.86"

print(expected)

print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)

if computed_answer == expected:
    print("✅ TASK IS CORRECT!")
else:
    print("❌ MISMATCH DETECTED")
    print("\nDifferences:")
    computed_parts = computed_answer.split('; ')
    expected_parts = expected.split('; ')
    
    for i, (comp, exp) in enumerate(zip(computed_parts, expected_parts)):
        if comp != exp:
            print(f"\n  Department {i+1}:")
            print(f"    Computed: {comp}")
            print(f"    Expected: {exp}")

