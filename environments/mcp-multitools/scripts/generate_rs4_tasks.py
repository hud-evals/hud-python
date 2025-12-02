"""
Generate rs4_tasks.json with correct answers calculated from the database.
"""
import httpx
import os
import random
import json

# Load env
with open('.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ[key] = val

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_API_KEY")
headers = {"apikey": key, "Authorization": f"Bearer {key}"}


def fetch_all(table, select="*"):
    """Fetch all rows with pagination."""
    all_data = []
    offset = 0
    while True:
        resp = httpx.get(f"{url}/rest/v1/{table}?select={select}&limit=1000&offset={offset}", headers=headers)
        data = resp.json()
        all_data.extend(data)
        if len(data) < 1000:
            break
        offset += 1000
    return all_data


def calculate_batched_profit_answer():
    """Calculate the answer for task-batched-profit-300."""
    # Get all products
    products = fetch_all("products", "id,sku,cost")
    print(f"Total products: {len(products)}")
    
    # Get 300 random SKUs (fixed seed for reproducibility)
    random.seed(42)
    selected = random.sample(products, 300)
    skus = [p['sku'] for p in selected]
    
    # Split into 6 batches of 50
    batches = {}
    for i in range(6):
        batch = skus[i*50:(i+1)*50]
        batches[f"batch_{i+1}"] = ", ".join(batch)
    
    # Add formula
    batches["formula"] = "profit = quantity × (unit_price × (1 - discount_percent/100) - cost)"
    
    # Get all order_items
    order_items = fetch_all("order_items", "product_id,quantity,unit_price,discount_percent")
    print(f"Total order_items: {len(order_items)}")
    
    # Build lookup maps
    sku_to_cost = {p['sku']: float(p['cost']) for p in products}
    id_to_sku = {p['id']: p['sku'] for p in products}
    
    # Calculate profit
    total_profit = 0
    for item in order_items:
        sku = id_to_sku.get(item['product_id'])
        if sku and sku in skus:
            qty = item['quantity']
            price = float(item['unit_price'])
            discount = float(item['discount_percent'])
            cost = sku_to_cost[sku]
            profit = qty * (price * (1 - discount/100) - cost)
            total_profit += profit
    
    answer = f"{total_profit:.2f}"
    print(f"Batched profit answer: {answer}")
    
    return batches, answer


def main():
    print("=== Generating rs4_tasks.json ===\n")
    
    # Calculate batched profit task
    initial_data, batched_profit_answer = calculate_batched_profit_answer()
    
    # Build tasks
    tasks = [
        {
            "id": "task-department-summary",
            "prompt": "For EACH of the 10 departments in the database, calculate:\n1. Number of employees\n2. Total salary (sum of all employee salaries in that department)\n3. Total revenue generated (sum of order revenue where sales_rep_id belongs to that department)\n\nRevenue = product revenue + shipping cost\nProduct revenue formula: quantity × unit_price × (1 - discount_percent/100)\nThen add orders.shipping_cost to get total revenue per order.\n\nIMPORTANT: Exclude cancelled orders (status = 'cancelled') from revenue calculation.\n\nFormat your answer as a single string with departments sorted ALPHABETICALLY by name:\n`DeptName1: emp_count, total_salary, total_revenue; DeptName2: emp_count, total_salary, total_revenue; ...`\n\nRevenue must have exactly 2 decimal places (e.g., 0.00, 22525.34).\n\nExample format: `Engineering: 15, 1500000, 250000.50; Marketing: 12, 1200000, 0.00`\n\nYou have access to a Supabase database via supabase_execute_sql (project_id: 'qgubrffqclddzehyyvbt'). Available tables: departments, employees, orders, order_items.\n\nStore the formatted answer using local_scratchpad_write with key 'answer'. When finished, say 'Task completed.'",
            "mcp_config": {
                "local": {
                    "command": "docker",
                    "args": ["run", "--rm", "-i", 
                        "--env-file", "/home/rs/projects/hud-python/.env",
                        "mcp-multitools:latest"]
                },
                "supabase": {
                    "url": "https://mcp.supabase.com/mcp?read_only=true",
                    "headers": {
                        "Authorization": "Bearer sbp_6db19bff28d413a387bb9999cf989201887e5bdd"
                    }
                }
            },
            "setup_tool": {"name": "local_setup", "arguments": {}},
            "evaluate_tool": {
                "name": "local_evaluate",
                "arguments": {
                    "exact_values": {"answer": "Customer Support: 9, 683435, 0.00; Engineering: 11, 1222803, 0.00; Finance: 10, 985764, 0.00; Human Resources: 10, 808739, 0.00; Legal: 9, 1218585, 0.00; Marketing: 10, 988876, 0.00; Operations: 9, 790065, 0.00; Product Management: 9, 1236463, 0.00; Research & Development: 9, 1447542, 0.00; Sales: 11, 1340737, 2827914.86"}
                }
            },
            "agent_config": {
                "allowed_tools": ["local_scratchpad_write", "local_scratchpad_read", "supabase_execute_sql", "supabase_list_tables"]
            }
        },
        {
            "id": "task-batched-profit-300",
            "prompt": "The scratchpad contains 6 batches of product SKUs (batch_1 through batch_6), with 50 SKUs in each batch (300 total). There is also a 'formula' key with the profit calculation formula.\n\nYour task: Calculate the TOTAL PROFIT for all 300 products across all batches.\n\nSteps:\n1. Read each batch from the scratchpad (batch_1, batch_2, ..., batch_6)\n2. Read the formula from scratchpad\n3. Query the database for the required data\n4. Calculate total profit\n\nNOTE: The SQL query with 300 SKUs may be very long. You may need to run multiple queries (e.g., one per batch) and sum the results.\n\nYou have access to a Supabase database via supabase_execute_sql (project_id: 'qgubrffqclddzehyyvbt'). Tables: products, order_items.\n\nStore the total profit (rounded to 2 decimal places) using local_scratchpad_write with key 'answer'. When finished, say 'Task completed.'",
            "mcp_config": {
                "local": {
                    "command": "docker",
                    "args": ["run", "--rm", "-i", 
                        "--env-file", "/home/rs/projects/hud-python/.env",
                        "mcp-multitools:latest"]
                },
                "supabase": {
                    "url": "https://mcp.supabase.com/mcp?read_only=true",
                    "headers": {
                        "Authorization": "Bearer sbp_6db19bff28d413a387bb9999cf989201887e5bdd"
                    }
                }
            },
            "setup_tool": {
                "name": "local_setup",
                "arguments": {
                    "initial_data": initial_data
                }
            },
            "evaluate_tool": {
                "name": "local_evaluate",
                "arguments": {
                    "exact_values": {"answer": batched_profit_answer}
                }
            },
            "agent_config": {
                "allowed_tools": ["local_scratchpad_write", "local_scratchpad_read", "supabase_execute_sql", "supabase_list_tables"]
            }
        }
    ]
    
    # Write to file
    output_path = "environments/mcp-multitools/rs4_tasks.json"
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    
    print(f"\n=== Written to {output_path} ===")
    print(f"Tasks: {len(tasks)}")


if __name__ == "__main__":
    main()

