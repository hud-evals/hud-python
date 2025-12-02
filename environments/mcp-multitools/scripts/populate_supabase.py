#!/usr/bin/env python3
"""Populate Supabase 'people' table with 100 realistic records."""

import random
import httpx

# Supabase config
SUPABASE_URL = "https://qgubrffqclddzehyyvbt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFndWJyZmZxY2xkZHplaHl5dmJ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQxMTY3MjYsImV4cCI6MjA3OTY5MjcyNn0.LdB0gwRaHzOdVne11lcTdQNbZ0V-dMF5Dzc8Ys6BPZU"

# Realistic first names (diverse)
FIRST_NAMES = [
    "James", "Maria", "David", "Sarah", "Michael", "Emma", "Robert", "Olivia",
    "William", "Sophia", "Carlos", "Isabella", "Ahmed", "Mia", "Hiroshi", "Yuki",
    "Wei", "Aisha", "Pierre", "Fatima", "Ivan", "Priya", "Mohammed", "Ling",
    "Andreas", "Nadia", "Juan", "Elena", "Raj", "Amara", "Chen", "Ingrid",
    "Boris", "Keiko", "Dmitri", "Zara", "Kofi", "Mei", "Stefan", "Aaliya",
    "Erik", "Chioma", "Lars", "Sakura", "Marco", "Ananya", "Olga", "Tariq",
    "Hans", "Leila"
]

# Realistic last names (diverse)
LAST_NAMES = [
    "Smith", "Garcia", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
    "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Jackson",
    "Lee", "Patel", "Kim", "Nguyen", "Chen", "Wang", "Tanaka", "Suzuki", "Yamamoto",
    "Müller", "Schmidt", "Fischer", "Weber", "Johansson", "Andersson", "Petrov",
    "Ivanov", "Silva", "Santos", "Oliveira", "Jensen", "Nielsen", "Hansen",
    "Kowalski", "Novak", "Okonkwo", "Mensah", "Abubakar", "Hassan", "Ali",
    "Singh", "Kumar", "Sharma", "Gupta", "Russo"
]

def generate_person(id_num: int) -> dict:
    """Generate a realistic person with correlated height/weight."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    
    # Height: normal distribution (150-200 cm, mean 170)
    height = int(random.gauss(170, 12))
    height = max(150, min(200, height))  # clamp
    
    # Weight: correlated with height + some variation
    # BMI typically 18-30, use ~22 as baseline
    base_weight = (height / 100) ** 2 * 22
    weight = round(base_weight + random.gauss(0, 8), 1)
    weight = max(45.0, min(120.0, weight))  # clamp
    
    return {
        "id": id_num,
        "full_name": f"{first} {last}",
        "weight_kg": weight,
        "height_cm": height,
    }

def main():
    # Generate 100 people
    people = [generate_person(i + 1) for i in range(100)]
    
    # Insert into Supabase (table name with spaces needs URL encoding)
    table_name = "people heights and weights"
    url = f"{SUPABASE_URL}/rest/v1/{table_name.replace(' ', '%20')}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    
    print(f"Inserting {len(people)} records into 'people' table...")
    
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=people, headers=headers)
        
        if response.status_code in (200, 201):
            print("✓ Successfully inserted 100 records!")
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return
    
    # Verify by querying first 5
    print("\nFirst 5 records:")
    verify_url = f"{SUPABASE_URL}/rest/v1/{table_name.replace(' ', '%20')}?select=*&limit=5&order=id"
    response = httpx.get(verify_url, headers=headers)
    for p in response.json():
        print(f"  {p['id']:3d}. {p['full_name']:20s} - {p['weight_kg']:5.1f} kg, {p['height_cm']:3d} cm")

if __name__ == "__main__":
    main()

