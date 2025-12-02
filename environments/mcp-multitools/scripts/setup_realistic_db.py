#!/usr/bin/env python3
"""
Create a HIGHLY REALISTIC connected database for evaluation tasks.

Features:
- Realistic alphanumeric IDs (EMP-7284K, PRD-38291, etc.)
- 97 employees with proper manager hierarchy (managers hired BEFORE reports)
- 823 products with realistic price distributions
- Name collisions: 3 share surname, 2 share first name, 3 have SAME full name
- Dates that make sense (orders can't be delivered before shipped, etc.)
- Country-appropriate phone numbers
- Realistic email domains
- Status matches timeline (pending = recent, delivered = older)
"""

import argparse
import asyncio
import os
import random
import string
from datetime import datetime, timedelta
from dotenv import load_dotenv
import httpx

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_API_KEY") or os.getenv("SUPABASE_KEY")

# Fixed "today" for deterministic data
TODAY = datetime(2024, 11, 15)

# ============================================================
# SQL SCHEMA
# ============================================================

CREATE_TABLES_SQL = """
-- Drop all tables
DROP TABLE IF EXISTS meeting_attendees CASCADE;
DROP TABLE IF EXISTS meetings CASCADE;
DROP TABLE IF EXISTS shifts CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS suppliers CASCADE;
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS departments CASCADE;

-- DEPARTMENTS
CREATE TABLE departments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    budget INTEGER NOT NULL,
    floor_number INTEGER,
    head_id TEXT
);
ALTER TABLE departments DISABLE ROW LEVEL SECURITY;

-- EMPLOYEES (with manager hierarchy)
CREATE TABLE employees (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    department_id TEXT REFERENCES departments(id),
    position TEXT NOT NULL,
    salary INTEGER NOT NULL,
    hire_date DATE NOT NULL,
    birth_date DATE,
    manager_id TEXT REFERENCES employees(id),
    is_active BOOLEAN DEFAULT TRUE
);
ALTER TABLE employees DISABLE ROW LEVEL SECURITY;

-- SUPPLIERS
CREATE TABLE suppliers (
    id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    contact_name TEXT,
    contact_email TEXT,
    phone TEXT,
    country TEXT NOT NULL,
    city TEXT,
    rating DECIMAL(2,1) CHECK (rating >= 1 AND rating <= 5),
    payment_terms INTEGER DEFAULT 30
);
ALTER TABLE suppliers DISABLE ROW LEVEL SECURITY;

-- PRODUCTS (with creator employee)
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    sku TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL,
    subcategory TEXT,
    price DECIMAL(10,2) NOT NULL,
    cost DECIMAL(10,2) NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    supplier_id TEXT REFERENCES suppliers(id),
    added_by TEXT REFERENCES employees(id),
    discontinued BOOLEAN DEFAULT FALSE
);
ALTER TABLE products DISABLE ROW LEVEL SECURITY;

-- CUSTOMERS (with account manager)
CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    contact_first_name TEXT NOT NULL,
    contact_last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    address TEXT,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    postal_code TEXT,
    account_manager_id TEXT REFERENCES employees(id),
    credit_limit INTEGER DEFAULT 10000,
    is_active BOOLEAN DEFAULT TRUE
);
ALTER TABLE customers DISABLE ROW LEVEL SECURITY;

-- ORDERS
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(id),
    sales_rep_id TEXT REFERENCES employees(id),
    order_date DATE NOT NULL,
    required_date DATE,
    shipped_date DATE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled', 'returned')),
    shipping_method TEXT,
    shipping_cost DECIMAL(8,2) DEFAULT 0,
    notes TEXT
);
ALTER TABLE orders DISABLE ROW LEVEL SECURITY;

-- ORDER ITEMS
CREATE TABLE order_items (
    id TEXT PRIMARY KEY,
    order_id TEXT REFERENCES orders(id),
    product_id TEXT REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    discount_percent DECIMAL(5,2) DEFAULT 0
);
ALTER TABLE order_items DISABLE ROW LEVEL SECURITY;

-- SHIFTS (employee schedules)
CREATE TABLE shifts (
    id TEXT PRIMARY KEY,
    employee_id TEXT REFERENCES employees(id),
    shift_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    break_minutes INTEGER DEFAULT 30,
    location TEXT NOT NULL,
    notes TEXT
);
ALTER TABLE shifts DISABLE ROW LEVEL SECURITY;

-- MEETINGS
CREATE TABLE meetings (
    id TEXT PRIMARY KEY,
    organizer_id TEXT REFERENCES employees(id),
    title TEXT NOT NULL,
    description TEXT,
    scheduled_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    room TEXT,
    is_virtual BOOLEAN DEFAULT FALSE,
    meeting_link TEXT
);
ALTER TABLE meetings DISABLE ROW LEVEL SECURITY;

-- MEETING ATTENDEES
CREATE TABLE meeting_attendees (
    id TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id),
    employee_id TEXT REFERENCES employees(id),
    response_status TEXT CHECK (response_status IN ('accepted', 'declined', 'tentative', 'pending')),
    is_required BOOLEAN DEFAULT TRUE
);
ALTER TABLE meeting_attendees DISABLE ROW LEVEL SECURITY;

-- Indexes
CREATE INDEX idx_emp_manager ON employees(manager_id);
CREATE INDEX idx_emp_dept ON employees(department_id);
CREATE INDEX idx_prod_supplier ON products(supplier_id);
CREATE INDEX idx_prod_addedby ON products(added_by);
CREATE INDEX idx_prod_category ON products(category);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_salesrep ON orders(sales_rep_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orderitems_order ON order_items(order_id);
CREATE INDEX idx_orderitems_product ON order_items(product_id);
CREATE INDEX idx_shifts_emp ON shifts(employee_id);
CREATE INDEX idx_shifts_date ON shifts(shift_date);
CREATE INDEX idx_meetings_org ON meetings(organizer_id);
CREATE INDEX idx_meetings_date ON meetings(scheduled_date);
"""

# ============================================================
# REALISTIC DATA POOLS
# ============================================================

# Unique first names pool
FIRST_NAMES_POOL = [
    "Alexander", "Benjamin", "Charlotte", "Daniel", "Eleanor", "Felix", "Georgia", "Harrison",
    "Isabella", "James", "Katherine", "Leonardo", "Margaret", "Nicholas", "Olivia", "Patrick",
    "Quinn", "Rebecca", "Sebastian", "Theresa", "Ulrich", "Victoria", "William", "Xavier",
    "Yolanda", "Zachary", "Adrian", "Beatrice", "Cameron", "Diana", "Ethan", "Florence",
    "Gabriel", "Helena", "Isaac", "Julia", "Kenneth", "Laura", "Marcus", "Natalie",
    "Oscar", "Penelope", "Raymond", "Sophia", "Thomas", "Uma", "Vincent", "Wendy",
    "Xander", "Yasmine", "Aaron", "Bridget", "Connor", "Delilah", "Emmanuel", "Fiona",
    "Gregory", "Hannah", "Ivan", "Josephine", "Kyle", "Lydia", "Maxwell", "Nicole",
    "Oliver", "Patricia", "Quentin", "Rosa", "Samuel", "Tiffany", "Ursula", "Victor",
    "Walter", "Ximena", "Yusuf", "Zoe", "Arthur", "Bianca", "Christian", "Danielle",
    "Edward", "Francesca", "George", "Heather", "Ignacio", "Jennifer", "Kevin", "Linda",
    "Martin", "Naomi", "Peter", "Rachel", "Stephen", "Teresa"
]

LAST_NAMES_POOL = [
    "Anderson", "Blackwood", "Chen", "Dimitrov", "Edwards", "Fischer", "Gonzalez", "Harrison",
    "Ivanovic", "Jensen", "Kim", "Larsson", "Morrison", "Nakamura", "O'Brien", "Petrov",
    "Quinn", "Rodriguez", "Svensson", "Tanaka", "Underwood", "Volkov", "Wagner", "Xu",
    "Yamamoto", "Zimmerman", "Adler", "Bergman", "Castillo", "Dubois", "Eriksson", "Fernandez",
    "Garcia", "Hoffman", "Ishikawa", "Johansson", "Kowalski", "Lopez", "Mueller", "Nielsen",
    "Olsen", "Park", "Reyes", "Schmidt", "Torres", "Ueda", "Vargas", "Williams",
    "Xiao", "Yoshida", "Zhang", "Alvarez", "Brown", "Cruz", "Davis", "Evans",
    "Foster", "Green", "Harris", "Jackson", "King", "Lee", "Martinez", "Nguyen",
    "Ortiz", "Patel", "Richardson", "Singh", "Thompson", "Uribe", "Valdez", "White",
    "Yang", "Zhu", "Baker", "Clark", "Diaz", "Ellis", "Franklin", "Gray"
]

DEPARTMENTS_DATA = [
    ("DEPT-ENG", "Engineering", 2850000, 3),
    ("DEPT-SAL", "Sales", 1920000, 2),
    ("DEPT-MKT", "Marketing", 1050000, 2),
    ("DEPT-HR", "Human Resources", 680000, 1),
    ("DEPT-FIN", "Finance", 1240000, 4),
    ("DEPT-OPS", "Operations", 1580000, 1),
    ("DEPT-SUP", "Customer Support", 890000, 2),
    ("DEPT-RND", "Research & Development", 3450000, 5),
    ("DEPT-LEG", "Legal", 970000, 4),
    ("DEPT-PRD", "Product Management", 1380000, 3),
]

POSITIONS_BY_LEVEL = {
    "Engineering": [
        ("Junior Software Engineer", 75000, 95000),
        ("Software Engineer", 95000, 130000),
        ("Senior Software Engineer", 130000, 165000),
        ("Staff Engineer", 160000, 195000),
        ("Engineering Manager", 150000, 190000),
        ("Director of Engineering", 185000, 240000),
    ],
    "Sales": [
        ("Sales Development Rep", 55000, 75000),
        ("Account Executive", 70000, 110000),
        ("Senior Account Executive", 100000, 145000),
        ("Sales Manager", 120000, 160000),
        ("Regional Sales Director", 150000, 200000),
    ],
    "Marketing": [
        ("Marketing Coordinator", 50000, 70000),
        ("Marketing Specialist", 65000, 90000),
        ("Senior Marketing Manager", 95000, 130000),
        ("Director of Marketing", 130000, 175000),
    ],
    "Human Resources": [
        ("HR Coordinator", 48000, 62000),
        ("HR Specialist", 58000, 78000),
        ("HR Manager", 80000, 105000),
        ("Director of HR", 110000, 145000),
    ],
    "Finance": [
        ("Junior Accountant", 55000, 72000),
        ("Financial Analyst", 70000, 95000),
        ("Senior Accountant", 85000, 115000),
        ("Finance Manager", 110000, 145000),
        ("Controller", 140000, 185000),
    ],
    "Operations": [
        ("Operations Coordinator", 48000, 65000),
        ("Operations Analyst", 62000, 85000),
        ("Operations Manager", 85000, 120000),
        ("Director of Operations", 125000, 165000),
    ],
    "Customer Support": [
        ("Support Specialist", 42000, 58000),
        ("Senior Support Specialist", 55000, 72000),
        ("Support Team Lead", 68000, 88000),
        ("Customer Success Manager", 80000, 110000),
        ("Director of Support", 105000, 140000),
    ],
    "Research & Development": [
        ("Research Associate", 72000, 95000),
        ("Research Scientist", 95000, 130000),
        ("Senior Research Scientist", 125000, 165000),
        ("Principal Scientist", 160000, 210000),
        ("R&D Director", 180000, 250000),
    ],
    "Legal": [
        ("Paralegal", 55000, 75000),
        ("Associate Counsel", 95000, 130000),
        ("Senior Counsel", 130000, 175000),
        ("General Counsel", 175000, 250000),
    ],
    "Product Management": [
        ("Associate Product Manager", 80000, 105000),
        ("Product Manager", 110000, 145000),
        ("Senior Product Manager", 140000, 180000),
        ("Director of Product", 170000, 220000),
    ],
}

# Map dept ID to name
DEPT_ID_TO_NAME = {d[0]: d[1] for d in DEPARTMENTS_DATA}

PHONE_PREFIXES = {
    "USA": "+1", "Canada": "+1", "UK": "+44", "Germany": "+49", "France": "+33",
    "Japan": "+81", "Australia": "+61", "Brazil": "+55", "India": "+91", "Mexico": "+52",
    "Spain": "+34", "Italy": "+39", "Netherlands": "+31", "Sweden": "+46", "Singapore": "+65",
    "South Korea": "+82", "China": "+86"
}

CITIES_BY_COUNTRY = {
    "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle", "Boston", "Denver", "San Francisco", "Miami"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
    "UK": ["London", "Manchester", "Birmingham", "Edinburgh", "Bristol"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"],
    "France": ["Paris", "Lyon", "Marseille", "Nice", "Toulouse"],
    "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
    "Brazil": ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
    "Mexico": ["Mexico City", "Guadalajara", "Monterrey", "Cancún"],
    "Spain": ["Madrid", "Barcelona", "Valencia", "Seville"],
    "Italy": ["Rome", "Milan", "Florence", "Venice", "Naples"],
    "Netherlands": ["Amsterdam", "Rotterdam", "The Hague", "Utrecht"],
    "Sweden": ["Stockholm", "Gothenburg", "Malmö", "Uppsala"],
    "Singapore": ["Singapore"],
    "South Korea": ["Seoul", "Busan", "Incheon"],
    "China": ["Shanghai", "Beijing", "Shenzhen", "Guangzhou"],
}

PRODUCT_CATALOG = {
    "Electronics": {
        "Smartphones": [("iPhone 15 Pro", 999, 1199), ("Galaxy S24 Ultra", 899, 1099), ("Pixel 8 Pro", 799, 999), ("OnePlus 12", 699, 849)],
        "Laptops": [("MacBook Pro 14", 1599, 1999), ("ThinkPad X1 Carbon", 1299, 1699), ("Dell XPS 15", 1199, 1549), ("Surface Laptop 5", 999, 1399)],
        "Tablets": [("iPad Pro 12.9", 999, 1299), ("Galaxy Tab S9", 699, 899), ("Surface Pro 9", 899, 1199)],
        "Audio": [("AirPods Pro 2", 199, 279), ("Sony WH-1000XM5", 299, 399), ("Bose QC45", 279, 349), ("Sonos One", 179, 229)],
        "Cameras": [("Sony A7 IV", 2199, 2699), ("Canon EOS R6", 1999, 2499), ("Nikon Z6 III", 1799, 2299)],
        "Wearables": [("Apple Watch Ultra 2", 699, 849), ("Galaxy Watch 6", 329, 429), ("Fitbit Sense 2", 249, 329)],
    },
    "Clothing": {
        "Men's Casual": [("Oxford Button-Down Shirt", 45, 79), ("Slim Fit Chinos", 55, 89), ("Merino Wool Sweater", 89, 149), ("Canvas Sneakers", 65, 99)],
        "Men's Formal": [("Tailored Wool Suit", 399, 699), ("Dress Shirt", 65, 119), ("Leather Oxford Shoes", 179, 299), ("Silk Tie", 55, 95)],
        "Women's Casual": [("Cashmere Cardigan", 129, 219), ("High-Rise Jeans", 79, 129), ("Cotton Blouse", 49, 89), ("Ballet Flats", 69, 119)],
        "Women's Formal": [("Wool Blazer", 199, 349), ("Silk Dress", 179, 329), ("Leather Pumps", 149, 269)],
        "Sportswear": [("Running Shoes", 99, 179), ("Yoga Pants", 59, 99), ("Performance T-Shirt", 35, 65), ("Windbreaker Jacket", 89, 149)],
    },
    "Home & Garden": {
        "Furniture": [("Ergonomic Office Chair", 299, 549), ("Standing Desk", 449, 799), ("Leather Sofa", 999, 1899), ("Dining Table Set", 699, 1299)],
        "Kitchen": [("Espresso Machine", 299, 599), ("Stand Mixer", 249, 449), ("Chef's Knife Set", 149, 349), ("Dutch Oven", 89, 179)],
        "Bedding": [("Memory Foam Mattress", 699, 1299), ("Egyptian Cotton Sheets", 129, 249), ("Down Comforter", 199, 349)],
        "Lighting": [("LED Floor Lamp", 79, 149), ("Smart Bulb Pack", 39, 79), ("Chandelier", 199, 449)],
        "Outdoor": [("Patio Furniture Set", 599, 1199), ("Gas Grill", 449, 899), ("Garden Tool Set", 79, 159)],
    },
    "Books": {
        "Technical": [("Clean Code", 35, 55), ("Design Patterns", 45, 65), ("System Design Interview", 30, 50), ("Cracking the Coding Interview", 35, 55)],
        "Business": [("Good to Great", 20, 35), ("Zero to One", 22, 38), ("The Lean Startup", 25, 42), ("Thinking Fast and Slow", 18, 32)],
        "Fiction": [("The Great Gatsby", 12, 22), ("1984", 14, 24), ("To Kill a Mockingbird", 13, 23), ("Pride and Prejudice", 11, 21)],
        "Science": [("A Brief History of Time", 18, 32), ("Sapiens", 22, 38), ("The Selfish Gene", 16, 28)],
    },
    "Sports & Outdoors": {
        "Fitness": [("Adjustable Dumbbells", 249, 449), ("Exercise Bike", 399, 799), ("Yoga Mat Premium", 49, 89), ("Resistance Bands Set", 29, 59)],
        "Outdoor Gear": [("Hiking Backpack 50L", 129, 229), ("4-Person Tent", 199, 349), ("Sleeping Bag", 79, 149), ("Trekking Poles", 59, 109)],
        "Sports Equipment": [("Tennis Racket Pro", 149, 279), ("Golf Club Set", 499, 999), ("Basketball Official", 29, 59), ("Soccer Ball", 25, 55)],
    },
    "Beauty & Personal Care": {
        "Skincare": [("Vitamin C Serum", 28, 58), ("Retinol Cream", 35, 69), ("Sunscreen SPF 50", 18, 38), ("Hyaluronic Acid", 24, 48)],
        "Haircare": [("Argan Oil Treatment", 22, 42), ("Sulfate-Free Shampoo", 18, 34), ("Hair Mask", 28, 52)],
        "Fragrances": [("Eau de Parfum 100ml", 79, 159), ("Cologne 50ml", 55, 109), ("Body Mist", 25, 49)],
    },
}

SUPPLIER_DATA = [
    ("TechSource Global", "USA", "Michael Chen"),
    ("EuroDistributors GmbH", "Germany", "Hans Mueller"),
    ("Pacific Rim Trading", "Japan", "Yuki Tanaka"),
    ("Nordic Supply Co", "Sweden", "Erik Larsson"),
    ("MegaStock Industries", "USA", "Sarah Johnson"),
    ("Atlas Manufacturing", "UK", "James Wilson"),
    ("Prime Wholesale Ltd", "Canada", "Marie Dubois"),
    ("FastShip Logistics", "Netherlands", "Jan van Berg"),
    ("Reliable Partners", "Australia", "David Brown"),
    ("Global Goods Inc", "USA", "Robert Smith"),
    ("Quality First Trading", "Germany", "Anna Schmidt"),
    ("Eastern Imports", "China", "Wei Zhang"),
    ("Southern Cross Supply", "Australia", "Emily Davis"),
    ("Continental Distributors", "France", "Pierre Martin"),
    ("Summit Sourcing", "India", "Priya Patel"),
    ("Vertex Distribution", "UK", "William Taylor"),
    ("CoreStock Solutions", "USA", "Jennifer Lee"),
    ("Alpine Trading", "Switzerland", "Thomas Berger"),
    ("Seaside Imports", "Spain", "Carlos Garcia"),
    ("Northern Lights Supply", "Canada", "Michelle Thompson"),
    ("Pinnacle Products", "Singapore", "David Tan"),
    ("Sterling Supplies", "UK", "Elizabeth Moore"),
    ("Frontier Wholesale", "USA", "Christopher Anderson"),
]

COMPANY_NAMES = [
    "Acme Corporation", "TechStart Innovations", "Global Solutions Ltd", "DataFlow Systems", "CloudNine Technologies",
    "InnovateCo Industries", "SwiftBiz Enterprises", "NextGen Digital", "BlueSky Partners", "Quantum Labs Inc",
    "Alpha Dynamics Corp", "Beta Industries LLC", "Gamma Technologies", "Delta Services Group", "Epsilon Holdings",
    "Zeta Ventures", "Theta Systems Inc", "Iota Networks", "Kappa Solutions", "Lambda Corporation",
    "Omega Enterprises", "Sigma Partners", "Phoenix Rising Inc", "Stellar Dynamics", "Horizon Tech Group",
    "Vanguard Industries", "Pinnacle Corp", "Summit Enterprises", "Vertex Solutions", "Apex Group Inc",
    "Core Systems Ltd", "Frontier Technologies", "Legacy Partners", "Prime Industries", "Elite Consulting Group",
    "Titan Corporation", "Atlas Enterprises", "Mercury Systems", "Jupiter Holdings", "Neptune Trading Co",
    "Orion Logistics", "Polaris Investments", "Aurora Industries", "Cosmos Technologies", "Nova Solutions",
    "Zenith Corporation", "Nexus Partners", "Velocity Dynamics", "Momentum Group", "Catalyst Innovations",
    "Synergy Systems", "Fusion Technologies", "Matrix Solutions", "Vector Industries", "Prism Holdings",
]

SHIPPING_METHODS = ["Standard Ground", "Express 2-Day", "Overnight", "Economy", "Freight"]

MEETING_ROOMS = ["Board Room", "Conference Room A", "Conference Room B", "Meeting Room 101", "Meeting Room 102", 
                 "Innovation Lab", "Training Room", "Executive Suite", "Huddle Space 1", "Huddle Space 2"]

MEETING_TITLES = [
    "Weekly Team Standup", "Sprint Planning", "Sprint Retrospective", "Quarterly Business Review",
    "Product Roadmap Discussion", "Budget Planning Meeting", "Client Presentation", "Training Session",
    "One-on-One", "Department All-Hands", "Project Kickoff", "Design Review", "Code Review Session",
    "Sales Pipeline Review", "Marketing Campaign Planning", "HR Policy Update", "Security Training",
    "New Hire Orientation", "Performance Review", "Strategy Session"
]

LOCATIONS = ["Main Office - Floor 1", "Main Office - Floor 2", "Main Office - Floor 3", "Main Office - Floor 4",
             "Main Office - Floor 5", "Remote", "Branch Office - Downtown", "Branch Office - Westside", "Warehouse A"]


def gen_id(prefix: str, num: int, add_letter: bool = True) -> str:
    """Generate realistic ID like EMP-10284K or PRD-38291."""
    if add_letter:
        letter = chr(65 + (num % 26))  # A-Z based on number
        return f"{prefix}-{num:05d}{letter}"
    return f"{prefix}-{num:05d}"


def gen_phone(country: str) -> str:
    """Generate country-appropriate phone number."""
    prefix = PHONE_PREFIXES.get(country, "+1")
    random.seed(hash(country + str(random.random())))
    if prefix == "+1":
        return f"{prefix}-{random.randint(201,989)}-{random.randint(201,989)}-{random.randint(1001,9999)}"
    elif prefix == "+44":
        return f"{prefix} {random.randint(20,79)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
    elif prefix == "+81":
        return f"{prefix}-{random.randint(3,90)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    else:
        return f"{prefix} {random.randint(100,999)} {random.randint(100,999)} {random.randint(1000,9999)}"


def generate_departments() -> list[dict]:
    return [{"id": d[0], "name": d[1], "budget": d[2], "floor_number": d[3], "head_id": None} for d in DEPARTMENTS_DATA]


def generate_employees(n: int = 97) -> list[dict]:
    """Generate employees with realistic hierarchy and name collisions."""
    random.seed(42424)
    employees = []
    used_emails = set()
    
    # Track for hierarchy: managers must be hired before their reports
    # We'll create employees in order of seniority
    
    # First, create the structure with name collisions
    # 3x "James Chen" (same full name, different departments)
    # 3x Morrison surname
    # 2x Alexander first name
    
    employee_specs = []
    emp_num_counter = 10000
    
    # Level 1: Directors (hired 6-10 years ago) - 10 people, one per dept
    for i, (dept_id, dept_name, _, _) in enumerate(DEPARTMENTS_DATA):
        positions = POSITIONS_BY_LEVEL.get(dept_name, POSITIONS_BY_LEVEL["Engineering"])
        director_pos = [p for p in positions if "Director" in p[0] or "General Counsel" in p[0] or "Controller" in p[0]]
        if not director_pos:
            director_pos = [positions[-1]]  # Highest level
        pos_name, sal_min, sal_max = director_pos[0]
        
        emp_num_counter += random.randint(5, 15)
        days_employed = random.randint(2200, 3650)  # 6-10 years
        
        employee_specs.append({
            "emp_num": emp_num_counter,
            "first_name": FIRST_NAMES_POOL[i],
            "last_name": LAST_NAMES_POOL[i],
            "dept_id": dept_id,
            "dept_name": dept_name,
            "position": pos_name,
            "salary": random.randint(sal_min, sal_max),
            "days_employed": days_employed,
            "level": 1,
            "manager_idx": None
        })
    
    # Level 2: Managers (hired 3-6 years ago) - 15 people
    for i in range(15):
        dept_idx = i % len(DEPARTMENTS_DATA)
        dept_id, dept_name, _, _ = DEPARTMENTS_DATA[dept_idx]
        positions = POSITIONS_BY_LEVEL.get(dept_name, POSITIONS_BY_LEVEL["Engineering"])
        manager_pos = [p for p in positions if "Manager" in p[0] or "Lead" in p[0] or "Senior" in p[0]]
        if not manager_pos:
            manager_pos = [positions[len(positions)//2]]
        pos_name, sal_min, sal_max = random.choice(manager_pos)
        
        emp_num_counter += random.randint(7, 18)
        days_employed = random.randint(1100, 2200)  # 3-6 years
        
        # Name collision: make employee 10,11,12 be "James Chen"
        if i == 0:
            first_name, last_name = "James", "Chen"
        elif i == 1:
            first_name, last_name = "James", "Chen"
        elif i == 2:
            first_name, last_name = "James", "Chen"
        elif i == 3:
            first_name, last_name = "Daniel", "Morrison"
        elif i == 4:
            first_name, last_name = "Rebecca", "Morrison"
        else:
            first_name = FIRST_NAMES_POOL[(i + 10) % len(FIRST_NAMES_POOL)]
            last_name = LAST_NAMES_POOL[(i * 3 + 10) % len(LAST_NAMES_POOL)]
        
        employee_specs.append({
            "emp_num": emp_num_counter,
            "first_name": first_name,
            "last_name": last_name,
            "dept_id": dept_id,
            "dept_name": dept_name,
            "position": pos_name,
            "salary": random.randint(sal_min, sal_max),
            "days_employed": days_employed,
            "level": 2,
            "manager_idx": dept_idx  # Reports to director of same dept
        })
    
    # Level 3: Senior ICs and regular employees (hired 0-3 years ago)
    remaining = n - len(employee_specs)
    for i in range(remaining):
        dept_idx = i % len(DEPARTMENTS_DATA)
        dept_id, dept_name, _, _ = DEPARTMENTS_DATA[dept_idx]
        positions = POSITIONS_BY_LEVEL.get(dept_name, POSITIONS_BY_LEVEL["Engineering"])
        # Pick from lower/mid level positions
        pos_name, sal_min, sal_max = positions[i % max(len(positions)-1, 1)]
        
        emp_num_counter += random.randint(3, 12)
        days_employed = random.randint(30, 1100)  # 0-3 years
        
        # More name collisions
        if i == 5:
            first_name, last_name = "Thomas", "Morrison"
        elif i == 6:
            first_name, last_name = "Alexander", "Petrov"
        elif i == 7:
            first_name, last_name = "Alexander", "Yamamoto"
        else:
            first_name = FIRST_NAMES_POOL[(i + 30) % len(FIRST_NAMES_POOL)]
            last_name = LAST_NAMES_POOL[(i * 2 + 30) % len(LAST_NAMES_POOL)]
        
        # Find a manager from level 2 in same dept
        potential_managers = [j for j, e in enumerate(employee_specs) 
                            if e["level"] == 2 and e["dept_id"] == dept_id]
        manager_idx = random.choice(potential_managers) if potential_managers else dept_idx
        
        employee_specs.append({
            "emp_num": emp_num_counter,
            "first_name": first_name,
            "last_name": last_name,
            "dept_id": dept_id,
            "dept_name": dept_name,
            "position": pos_name,
            "salary": random.randint(sal_min, sal_max),
            "days_employed": days_employed,
            "level": 3,
            "manager_idx": manager_idx
        })
    
    # Now create actual employee records
    for i, spec in enumerate(employee_specs):
        # Generate unique email
        base_email = f"{spec['first_name'].lower()}.{spec['last_name'].lower()}"
        email = f"{base_email}@company.com"
        if email in used_emails:
            email = f"{base_email}.{spec['emp_num']}@company.com"
        used_emails.add(email)
        
        hire_date = TODAY - timedelta(days=spec["days_employed"])
        
        # Age at hire should match position level
        if spec["level"] == 1:  # Directors: 38-58 years old at hire
            age_at_hire = random.randint(38, 58)
        elif spec["level"] == 2:  # Managers: 30-50 years old at hire
            age_at_hire = random.randint(30, 50)
        else:  # Regular employees: 22-45 years old at hire
            if "Junior" in spec["position"] or "Associate" in spec["position"] or "Coordinator" in spec["position"]:
                age_at_hire = random.randint(22, 32)
            elif "Senior" in spec["position"]:
                age_at_hire = random.randint(28, 45)
            else:
                age_at_hire = random.randint(24, 42)
        
        birth_year = hire_date.year - age_at_hire
        birth_date = datetime(birth_year, random.randint(1,12), random.randint(1,28))
        
        manager_id = None
        if spec["manager_idx"] is not None and spec["manager_idx"] < len(employee_specs):
            # Ensure employee is not their own manager!
            if spec["manager_idx"] != i:
                manager_id = gen_id("EMP", employee_specs[spec["manager_idx"]]["emp_num"])
        
        employees.append({
            "id": gen_id("EMP", spec["emp_num"]),
            "first_name": spec["first_name"],
            "last_name": spec["last_name"],
            "email": email,
            "phone": gen_phone("USA"),
            "department_id": spec["dept_id"],
            "position": spec["position"],
            "salary": spec["salary"],
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "birth_date": birth_date.strftime("%Y-%m-%d"),
            "manager_id": manager_id,
            "is_active": i % 19 != 0  # ~5% inactive
        })
    
    return employees


def generate_suppliers(n: int = 23) -> list[dict]:
    random.seed(88888)
    suppliers = []
    
    for i in range(min(n, len(SUPPLIER_DATA))):
        name, country, contact = SUPPLIER_DATA[i]
        city = random.choice(CITIES_BY_COUNTRY.get(country, ["Unknown"]))
        sup_num = 5000 + i * 11 + random.randint(1, 8)
        
        # Suppliers active for 1-8 years (all in the past)
        years_active = 1 + (i * 3) % 8
        active_since = TODAY - timedelta(days=years_active * 365 + random.randint(0, 180))
        
        suppliers.append({
            "id": gen_id("SUP", sup_num),
            "company_name": name,
            "contact_name": contact,
            "contact_email": f"{contact.lower().replace(' ', '.')}@{name.lower().replace(' ', '')[:12]}.com",
            "phone": gen_phone(country),
            "country": country,
            "city": city,
            "rating": round(3.2 + (sup_num % 18) / 10, 1),
            "payment_terms": [15, 30, 30, 45, 60][i % 5]
        })
    
    return suppliers


def generate_products(employees: list, suppliers: list, n: int = 823) -> list[dict]:
    """Generate products with realistic prices and assignments."""
    random.seed(99999)
    products = []
    sku_counter = 100000
    prod_num = 20000
    
    # Flatten the catalog
    all_products = []
    for category, subcats in PRODUCT_CATALOG.items():
        for subcat, items in subcats.items():
            for name, cost_low, cost_high in items:
                all_products.append((category, subcat, name, cost_low, cost_high))
    
    for i in range(n):
        # Cycle through base products, adding variations
        base_idx = i % len(all_products)
        category, subcat, base_name, cost_low, cost_high = all_products[base_idx]
        
        # Add variation to names for duplicates
        variation = i // len(all_products)
        if variation == 0:
            name = base_name
        elif variation == 1:
            name = f"{base_name} - Black"
        elif variation == 2:
            name = f"{base_name} - White"
        elif variation == 3:
            name = f"{base_name} v2"
        elif variation == 4:
            name = f"{base_name} Pro"
        else:
            name = f"{base_name} Edition {variation}"
        
        # Realistic cost and price (with margin 30-60%)
        cost = cost_low + (i * 17 % (cost_high - cost_low + 1))
        margin = 1.3 + (i % 31) / 100  # 30-60% margin
        price = round(cost * margin, 2)
        cost = round(cost * 0.98, 2)  # Slight variation
        
        prod_num += random.randint(2, 8)
        sku_counter += 1
        
        # Product was added 1-30 months ago
        months_ago = 1 + (i % 30)
        product_added_date = TODAY - timedelta(days=months_ago * 30)
        
        # Find an employee who was hired BEFORE this product was added
        # and works in operations/product/engineering
        eligible_employees = [e for e in employees 
                            if datetime.strptime(e["hire_date"], "%Y-%m-%d") < product_added_date
                            and e["department_id"] in ["DEPT-OPS", "DEPT-PRD", "DEPT-ENG"]
                            and e["is_active"]]
        if not eligible_employees:
            eligible_employees = [e for e in employees 
                                if datetime.strptime(e["hire_date"], "%Y-%m-%d") < product_added_date]
        if not eligible_employees:
            eligible_employees = employees  # Fallback for very old products
        
        added_by = eligible_employees[i % len(eligible_employees)]["id"]
        
        products.append({
            "id": gen_id("PRD", prod_num),
            "sku": f"SKU-{sku_counter}",
            "name": name,
            "description": f"High-quality {name.lower()} from our {category.lower()} collection.",
            "category": category,
            "subcategory": subcat,
            "price": price,
            "cost": cost,
            "stock_quantity": max(0, 50 + (prod_num % 400) - 100),  # Some at 0
            "reorder_level": 10 + (i % 40),
            "supplier_id": suppliers[i % len(suppliers)]["id"],
            "added_by": added_by,
            "discontinued": i % 47 == 0  # ~2% discontinued
        })
    
    return products


def generate_customers(employees: list, n: int = 156) -> list[dict]:
    random.seed(11111)
    customers = []
    countries = list(CITIES_BY_COUNTRY.keys())
    
    # Get sales/support employees for account managers
    sales_support = [e for e in employees if e["department_id"] in ["DEPT-SAL", "DEPT-SUP"]]
    if not sales_support:
        sales_support = employees[:10]
    
    for i in range(n):
        cust_num = 30000 + i * 7 + random.randint(1, 5)
        country = countries[i % len(countries)]
        city = CITIES_BY_COUNTRY[country][i % len(CITIES_BY_COUNTRY[country])]
        
        company = COMPANY_NAMES[i % len(COMPANY_NAMES)]
        if i >= len(COMPANY_NAMES):
            company = f"{company} {country}"
        
        contact_first = FIRST_NAMES_POOL[(i * 5 + 7) % len(FIRST_NAMES_POOL)]
        contact_last = LAST_NAMES_POOL[(i * 3 + 11) % len(LAST_NAMES_POOL)]
        
        clean_company = company.lower().replace(" ", "").replace("&", "").replace(".", "")[:12]
        
        customers.append({
            "id": gen_id("CUS", cust_num),
            "company_name": company,
            "contact_first_name": contact_first,
            "contact_last_name": contact_last,
            "email": f"contact{cust_num}@{clean_company}.com",
            "phone": gen_phone(country),
            "address": f"{random.randint(1,9999)} {['Main', 'Oak', 'Park', 'Lake', 'River', 'Hill'][i%6]} Street",
            "city": city,
            "country": country,
            "postal_code": f"{random.randint(10000, 99999)}",
            "account_manager_id": sales_support[i % len(sales_support)]["id"],
            "credit_limit": 5000 + (cust_num % 50) * 1000,
            "is_active": i % 23 != 0
        })
    
    return customers


def generate_orders(customers: list, employees: list, n: int = 512) -> list[dict]:
    """Generate orders with REALISTIC status based on date."""
    random.seed(22222)
    orders = []
    
    # Only ACTIVE sales employees can handle orders
    sales_emps = [e for e in employees if e["department_id"] == "DEPT-SAL" and e["is_active"]]
    if not sales_emps:
        sales_emps = [e for e in employees if e["is_active"]][:5]
    
    # Only ACTIVE customers place orders
    active_customers = [c for c in customers if c["is_active"]]
    
    for i in range(n):
        order_num = 100000 + i * 3 + random.randint(0, 2)
        days_ago = i % 400  # Orders span ~13 months
        order_date = TODAY - timedelta(days=days_ago)
        
        # Status MUST make sense with date
        if days_ago <= 3:
            status = "pending"
            shipped_date = None
            required_date = order_date + timedelta(days=random.randint(5, 14))
        elif days_ago <= 7:
            status = random.choice(["pending", "confirmed", "processing"])
            shipped_date = None
            required_date = order_date + timedelta(days=random.randint(5, 14))
        elif days_ago <= 14:
            status = random.choice(["processing", "shipped"])
            shipped_date = (order_date + timedelta(days=random.randint(2, 5))).strftime("%Y-%m-%d") if status == "shipped" else None
            required_date = order_date + timedelta(days=random.randint(7, 21))
        elif days_ago <= 30:
            status = random.choice(["shipped", "delivered", "delivered"])
            shipped_date = (order_date + timedelta(days=random.randint(1, 4))).strftime("%Y-%m-%d")
            required_date = order_date + timedelta(days=random.randint(7, 21))
        else:
            # Older orders are mostly delivered, some cancelled/returned
            status_weights = ["delivered"] * 8 + ["cancelled"] + ["returned"]
            status = random.choice(status_weights)
            if status in ["delivered", "returned"]:
                shipped_date = (order_date + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d")
            else:
                shipped_date = None
            required_date = order_date + timedelta(days=random.randint(7, 21))
        
        customer = active_customers[i % len(active_customers)]
        
        # Find a sales rep who was already hired on the order date
        eligible_reps = [e for e in sales_emps 
                        if datetime.strptime(e["hire_date"], "%Y-%m-%d") < order_date]
        if not eligible_reps:
            eligible_reps = sales_emps  # Fallback
        sales_rep = eligible_reps[i % len(eligible_reps)]
        
        orders.append({
            "id": gen_id("ORD", order_num),
            "customer_id": customer["id"],
            "sales_rep_id": sales_rep["id"],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "required_date": required_date.strftime("%Y-%m-%d"),
            "shipped_date": shipped_date,
            "status": status,
            "shipping_method": SHIPPING_METHODS[i % len(SHIPPING_METHODS)],
            "shipping_cost": round(5 + (order_num % 45) + random.random() * 10, 2),
            "notes": None if i % 7 != 0 else "Customer requested gift wrapping"
        })
    
    return orders


def generate_order_items(orders: list, products: list) -> list[dict]:
    random.seed(33333)
    items = []
    item_counter = 500000
    
    active_products = [p for p in products if not p.get("discontinued", False)]
    
    for order in orders:
        # 1-8 items per order, weighted toward 2-4
        num_items = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[5, 20, 25, 20, 15, 8, 5, 2])[0]
        
        used_products = set()
        for j in range(num_items):
            # Pick a product not already in this order
            attempts = 0
            while attempts < 10:
                product = active_products[(hash(order["id"]) + j * 17 + attempts) % len(active_products)]
                if product["id"] not in used_products:
                    break
                attempts += 1
            used_products.add(product["id"])
            
            quantity = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                      weights=[30, 25, 15, 10, 8, 5, 3, 2, 1, 1])[0]
            
            # Discount: most have none, some have small, few have large
            discount = random.choices([0, 5, 10, 15, 20, 25], weights=[60, 15, 10, 8, 5, 2])[0]
            
            items.append({
                "id": gen_id("ITM", item_counter, add_letter=False),
                "order_id": order["id"],
                "product_id": product["id"],
                "quantity": quantity,
                "unit_price": product["price"],
                "discount_percent": discount
            })
            item_counter += 1
    
    return items


def generate_shifts(employees: list, days: int = 45) -> list[dict]:
    random.seed(44444)
    shifts = []
    shift_counter = 800000
    
    # Only non-executive employees have shifts
    shift_employees = [e for e in employees if "Director" not in e["position"] and "General Counsel" not in e["position"]]
    
    for emp in shift_employees[:60]:  # Limit to 60 employees
        emp_hire_date = datetime.strptime(emp["hire_date"], "%Y-%m-%d")
        
        for d in range(days):
            # ~70% chance of working on weekdays, 20% on weekends
            shift_date = TODAY - timedelta(days=d)
            
            # Employee must be hired before the shift date!
            if shift_date < emp_hire_date:
                continue
                
            is_weekend = shift_date.weekday() >= 5
            
            if (is_weekend and random.random() < 0.2) or (not is_weekend and random.random() < 0.7):
                # Vary start times realistically
                start_hours = [7, 8, 8, 9, 9, 9, 10, 14, 14]  # Most 8-9am
                start_hour = random.choice(start_hours)
                shift_length = random.choice([4, 6, 8, 8, 8, 10])  # Most 8 hours
                
                shifts.append({
                    "id": gen_id("SHF", shift_counter, add_letter=False),
                    "employee_id": emp["id"],
                    "shift_date": shift_date.strftime("%Y-%m-%d"),
                    "start_time": f"{start_hour:02d}:00:00",
                    "end_time": f"{min(start_hour + shift_length, 23):02d}:00:00",
                    "break_minutes": 30 if shift_length >= 6 else 0,
                    "location": LOCATIONS[(hash(emp["id"]) + d) % len(LOCATIONS)],
                    "notes": None if d % 11 != 0 else "Coverage for team member on PTO"
                })
                shift_counter += 1
    
    return shifts


def generate_meetings(employees: list, n: int = 89) -> list[dict]:
    random.seed(55555)
    meetings = []
    
    # Only ACTIVE managers and above organize meetings
    organizers = [e for e in employees 
                  if e["is_active"] and any(x in e["position"] for x in ["Manager", "Director", "Lead", "Senior"])]
    if len(organizers) < 10:
        organizers = [e for e in employees if e["is_active"]][:20]
    
    for i in range(n):
        meeting_num = 900000 + i * 5 + random.randint(1, 4)
        
        days_from_now = random.randint(-30, 30)  # Past and future meetings
        meeting_date = TODAY + timedelta(days=days_from_now)
        
        # Find an organizer who was hired before the meeting date
        eligible_organizers = [o for o in organizers 
                              if datetime.strptime(o["hire_date"], "%Y-%m-%d") < meeting_date]
        if not eligible_organizers:
            eligible_organizers = organizers
        organizer = eligible_organizers[i % len(eligible_organizers)]
        
        # Business hours
        start_hour = random.choice([8, 9, 9, 10, 10, 11, 13, 14, 14, 15, 16])
        duration = random.choice([30, 30, 45, 60, 60, 90, 120])
        end_hour = start_hour + duration // 60
        end_min = duration % 60
        
        is_virtual = random.random() < 0.4  # 40% virtual
        
        meetings.append({
            "id": gen_id("MTG", meeting_num, add_letter=False),
            "organizer_id": organizer["id"],
            "title": MEETING_TITLES[i % len(MEETING_TITLES)],
            "description": f"Regular {MEETING_TITLES[i % len(MEETING_TITLES)].lower()} for the team.",
            "scheduled_date": meeting_date.strftime("%Y-%m-%d"),
            "start_time": f"{start_hour:02d}:00:00",
            "end_time": f"{end_hour:02d}:{end_min:02d}:00",
            "room": None if is_virtual else MEETING_ROOMS[i % len(MEETING_ROOMS)],
            "is_virtual": is_virtual,
            "meeting_link": f"https://meet.company.com/{meeting_num}" if is_virtual else None
        })
    
    return meetings


def generate_meeting_attendees(meetings: list, employees: list) -> list[dict]:
    random.seed(66666)
    attendees = []
    att_counter = 950000
    
    for meeting in meetings:
        # Determine if meeting is in the past
        meeting_date = datetime.strptime(meeting["scheduled_date"], "%Y-%m-%d")
        is_past = meeting_date < TODAY
        # 3-12 attendees, weighted toward 4-6
        num_attendees = random.choices(range(3, 13), weights=[5, 15, 20, 20, 15, 10, 6, 4, 3, 2])[0]
        
        # Organizer is always an attendee
        organizer = next((e for e in employees if e["id"] == meeting["organizer_id"]), None)
        meeting_employees = [organizer] if organizer else []
        
        # Add random attendees from same department + some from others
        # Only include employees hired BEFORE the meeting date
        if organizer:
            same_dept = [e for e in employees 
                        if e["department_id"] == organizer["department_id"] 
                        and e["id"] != organizer["id"]
                        and datetime.strptime(e["hire_date"], "%Y-%m-%d") < meeting_date]
            other_dept = [e for e in employees 
                         if e["department_id"] != organizer["department_id"]
                         and datetime.strptime(e["hire_date"], "%Y-%m-%d") < meeting_date]
            
            # 70% from same dept, 30% from other depts
            same_count = min(int(num_attendees * 0.7), len(same_dept))
            other_count = min(num_attendees - same_count - 1, len(other_dept))
            
            random.shuffle(same_dept)
            random.shuffle(other_dept)
            meeting_employees.extend(same_dept[:same_count])
            meeting_employees.extend(other_dept[:other_count])
        else:
            eligible = [e for e in employees if datetime.strptime(e["hire_date"], "%Y-%m-%d") < meeting_date]
            random.shuffle(eligible)
            meeting_employees = eligible[:num_attendees]
        
        for emp in meeting_employees:
            # Response status depends on whether meeting is past or future
            if is_past:
                # Past meetings: no "pending" or "tentative" - people either attended or didn't
                status = random.choices(
                    ["accepted", "declined"],
                    weights=[85, 15]
                )[0]
            else:
                # Future meetings: can have pending/tentative
                status = random.choices(
                    ["accepted", "declined", "tentative", "pending"],
                    weights=[60, 8, 12, 20]
                )[0]
            
            attendees.append({
                "id": gen_id("ATT", att_counter, add_letter=False),
                "meeting_id": meeting["id"],
                "employee_id": emp["id"],
                "response_status": status,
                "is_required": emp["id"] == meeting["organizer_id"] or random.random() < 0.6
            })
            att_counter += 1
    
    return attendees


def calculate_statistics(employees, customers, products, orders, order_items, suppliers, shifts, meetings, meeting_attendees):
    """Calculate statistics for creating evaluation tasks."""
    print("\n" + "=" * 80)
    print("📋 TASK STATISTICS - Use these exact values for evaluation!")
    print("=" * 80)
    
    emp_map = {e["id"]: e for e in employees}
    cust_map = {c["id"]: c for c in customers}
    prod_map = {p["id"]: p for p in products}
    sup_map = {s["id"]: s for s in suppliers}
    
    # === NAME COLLISION QUERIES ===
    print("\n🔤 NAME COLLISION QUERIES:")
    james_chens = [e for e in employees if e["first_name"] == "James" and e["last_name"] == "Chen"]
    print(f"1. Employees named 'James Chen': {len(james_chens)}")
    for jc in james_chens:
        print(f"   - {jc['id']}: {jc['position']} in {DEPT_ID_TO_NAME.get(jc['department_id'], jc['department_id'])}")
    
    morrisons = [e for e in employees if e["last_name"] == "Morrison"]
    print(f"2. Employees with surname 'Morrison': {len(morrisons)}")
    print(f"   Names: {[e['first_name'] + ' ' + e['last_name'] for e in morrisons]}")
    
    alexanders = [e for e in employees if e["first_name"] == "Alexander"]
    print(f"3. Employees with first name 'Alexander': {len(alexanders)}")
    print(f"   Full names: {[e['first_name'] + ' ' + e['last_name'] for e in alexanders]}")
    
    # === SINGLE TABLE QUERIES ===
    print("\n📊 SINGLE TABLE QUERIES:")
    
    highest_paid = max(employees, key=lambda x: x["salary"])
    print(f"4. Highest paid: {highest_paid['first_name']} {highest_paid['last_name']} (${highest_paid['salary']:,})")
    print(f"   Position: {highest_paid['position']}, ID: {highest_paid['id']}")
    
    lowest_paid_active = min([e for e in employees if e["is_active"]], key=lambda x: x["salary"])
    print(f"5. Lowest paid (active): {lowest_paid_active['first_name']} {lowest_paid_active['last_name']} (${lowest_paid_active['salary']:,})")
    
    most_expensive = max(products, key=lambda x: x["price"])
    print(f"6. Most expensive product: {most_expensive['name']} (${most_expensive['price']})")
    print(f"   ID: {most_expensive['id']}, SKU: {most_expensive['sku']}")
    
    cheapest = min(products, key=lambda x: x["price"])
    print(f"7. Cheapest product: {cheapest['name']} (${cheapest['price']})")
    
    highest_margin = max(products, key=lambda x: x["price"] - x["cost"])
    margin = highest_margin["price"] - highest_margin["cost"]
    print(f"8. Highest margin product: {highest_margin['name']} (${margin:.2f} margin)")
    
    # === CROSS-TABLE QUERIES ===
    print("\n🔗 CROSS-TABLE QUERIES:")
    
    # Order totals
    order_totals = {}
    for item in order_items:
        oid = item["order_id"]
        subtotal = item["quantity"] * float(item["unit_price"]) * (1 - float(item["discount_percent"])/100)
        order_totals[oid] = order_totals.get(oid, 0) + subtotal
    
    largest_order_id = max(order_totals.items(), key=lambda x: x[1])[0]
    largest_order = next(o for o in orders if o["id"] == largest_order_id)
    handling_emp = emp_map[largest_order["sales_rep_id"]]
    cust = cust_map[largest_order["customer_id"]]
    print(f"9. Largest order: ${order_totals[largest_order_id]:,.2f}")
    print(f"   Order ID: {largest_order_id}")
    print(f"   Customer: {cust['company_name']}")
    print(f"   Sales rep: {handling_emp['first_name']} {handling_emp['last_name']}")
    
    # Top customer by total spend
    customer_spend = {}
    for order in orders:
        cid = order["customer_id"]
        customer_spend[cid] = customer_spend.get(cid, 0) + order_totals.get(order["id"], 0)
    top_cust_id = max(customer_spend.items(), key=lambda x: x[1])[0]
    top_cust = cust_map[top_cust_id]
    print(f"10. Top customer (total spend): {top_cust['company_name']} (${customer_spend[top_cust_id]:,.2f})")
    
    # Employee with most orders
    emp_orders = {}
    for o in orders:
        emp_orders[o["sales_rep_id"]] = emp_orders.get(o["sales_rep_id"], 0) + 1
    top_sales_id = max(emp_orders.items(), key=lambda x: x[1])[0]
    top_sales = emp_map[top_sales_id]
    print(f"11. Sales rep with most orders: {top_sales['first_name']} {top_sales['last_name']} ({emp_orders[top_sales_id]} orders)")
    
    # Manager with most reports
    reports = {}
    for e in employees:
        if e["manager_id"]:
            reports[e["manager_id"]] = reports.get(e["manager_id"], 0) + 1
    if reports:
        top_mgr_id = max(reports.items(), key=lambda x: x[1])[0]
        top_mgr = emp_map[top_mgr_id]
        print(f"12. Manager with most reports: {top_mgr['first_name']} {top_mgr['last_name']} ({reports[top_mgr_id]} direct reports)")
    
    # Product added by most employees... no, product ordered most
    prod_qty = {}
    for item in order_items:
        prod_qty[item["product_id"]] = prod_qty.get(item["product_id"], 0) + item["quantity"]
    top_prod_id = max(prod_qty.items(), key=lambda x: x[1])[0]
    top_prod = prod_map[top_prod_id]
    print(f"13. Most ordered product: {top_prod['name']} ({prod_qty[top_prod_id]} units)")
    
    # Best supplier rating
    best_supplier = max(suppliers, key=lambda x: float(x["rating"]))
    print(f"14. Best rated supplier: {best_supplier['company_name']} (rating: {best_supplier['rating']})")
    
    # === AGGREGATION QUERIES ===
    print("\n📈 AGGREGATION QUERIES:")
    
    total_revenue = sum(order_totals.values())
    print(f"15. Total revenue (all orders): ${total_revenue:,.2f}")
    
    delivered = [o for o in orders if o["status"] == "delivered"]
    delivered_rev = sum(order_totals.get(o["id"], 0) for o in delivered)
    print(f"16. Delivered orders revenue: ${delivered_rev:,.2f} ({len(delivered)} orders)")
    
    pending = [o for o in orders if o["status"] == "pending"]
    print(f"17. Pending orders: {len(pending)}")
    
    total_salary = sum(e["salary"] for e in employees if e["is_active"])
    print(f"18. Total active employee salaries: ${total_salary:,}")
    
    eng_emps = [e for e in employees if e["department_id"] == "DEPT-ENG" and e["is_active"]]
    avg_eng = sum(e["salary"] for e in eng_emps) / len(eng_emps) if eng_emps else 0
    print(f"19. Average Engineering salary: ${avg_eng:,.2f} ({len(eng_emps)} employees)")
    
    total_stock_value = sum(p["stock_quantity"] * float(p["cost"]) for p in products)
    print(f"20. Total inventory value (at cost): ${total_stock_value:,.2f}")
    
    electronics = [p for p in products if p["category"] == "Electronics"]
    elec_revenue = sum(
        item["quantity"] * float(item["unit_price"]) * (1 - float(item["discount_percent"])/100)
        for item in order_items if item["product_id"] in [p["id"] for p in electronics]
    )
    print(f"21. Electronics revenue: ${elec_revenue:,.2f}")
    
    print("\n" + "=" * 80)


async def insert_data(client: httpx.AsyncClient, table: str, data: list[dict]) -> bool:
    if not data:
        print(f"  ⚠ {table}: no data to insert")
        return True
        
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    try:
        # Clear existing - delete all rows where id is not null (all rows)
        await client.delete(url, headers=headers, params={"id": "not.is.null"})
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            resp = await client.post(url, headers=headers, json=batch)
            if resp.status_code not in (200, 201):
                print(f"  ✗ {table}: {resp.status_code} - {resp.text[:200]}")
                return False
        
        print(f"  ✓ {table}: {len(data)} rows")
        return True
    except Exception as e:
        print(f"  ✗ {table}: {e}")
        return False


async def populate():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Set SUPABASE_URL and SUPABASE_API_KEY in .env")
        return
    
    print("Generating realistic data...")
    departments = generate_departments()
    employees = generate_employees(97)
    suppliers = generate_suppliers(23)
    products = generate_products(employees, suppliers, 823)
    customers = generate_customers(employees, 156)
    orders = generate_orders(customers, employees, 512)
    order_items = generate_order_items(orders, products)
    shifts = generate_shifts(employees, 45)
    meetings = generate_meetings(employees, 89)
    meeting_attendees = generate_meeting_attendees(meetings, employees)
    
    print(f"\n  📁 {len(departments)} departments")
    print(f"  👥 {len(employees)} employees")
    print(f"  🏭 {len(suppliers)} suppliers")
    print(f"  📦 {len(products)} products")
    print(f"  🏢 {len(customers)} customers")
    print(f"  🛒 {len(orders)} orders")
    print(f"  📝 {len(order_items)} order items")
    print(f"  ⏰ {len(shifts)} shifts")
    print(f"  📅 {len(meetings)} meetings")
    print(f"  👋 {len(meeting_attendees)} meeting attendees")
    
    print("\nInserting into Supabase...")
    async with httpx.AsyncClient(timeout=120) as client:
        # Order matters due to foreign keys!
        await insert_data(client, "departments", departments)
        await insert_data(client, "employees", employees)
        await insert_data(client, "suppliers", suppliers)
        await insert_data(client, "products", products)
        await insert_data(client, "customers", customers)
        await insert_data(client, "orders", orders)
        await insert_data(client, "order_items", order_items)
        await insert_data(client, "shifts", shifts)
        await insert_data(client, "meetings", meetings)
        await insert_data(client, "meeting_attendees", meeting_attendees)
    
    calculate_statistics(employees, customers, products, orders, order_items, suppliers, shifts, meetings, meeting_attendees)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", action="store_true", help="Print CREATE TABLE SQL")
    parser.add_argument("--populate", action="store_true", help="Populate tables with data")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    args = parser.parse_args()
    
    if args.sql:
        print(CREATE_TABLES_SQL)
    elif args.populate:
        asyncio.run(populate())
    elif args.stats:
        print("Generating data for statistics...")
        departments = generate_departments()
        employees = generate_employees(97)
        suppliers = generate_suppliers(23)
        products = generate_products(employees, suppliers, 823)
        customers = generate_customers(employees, 156)
        orders = generate_orders(customers, employees, 512)
        order_items = generate_order_items(orders, products)
        shifts = generate_shifts(employees, 45)
        meetings = generate_meetings(employees, 89)
        meeting_attendees = generate_meeting_attendees(meetings, employees)
        calculate_statistics(employees, customers, products, orders, order_items, suppliers, shifts, meetings, meeting_attendees)
    else:
        print("Realistic Database Setup")
        print("\n1. uv run python scripts/setup_realistic_db.py --sql")
        print("2. Paste SQL in Supabase Dashboard → SQL Editor → Run")
        print("3. uv run python scripts/setup_realistic_db.py --populate")


if __name__ == "__main__":
    main()
