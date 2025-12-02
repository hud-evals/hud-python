#!/usr/bin/env python3
"""
Setup script for the School database with extreme normalization.

Design principle: Simple questions require 5+ table joins.
Example: "What city is student X's chess club in?"
  → student → club_memberships → clubs → club_school_assignments 
    → schools → city_school_assignments → cities

Usage:
    python scripts/setup_school_db.py --populate
    python scripts/setup_school_db.py --verify
    python scripts/setup_school_db.py --query-examples
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from dotenv import load_dotenv

# Load environment from hud-python
load_dotenv(os.path.join(os.path.dirname(__file__), '../../hud-python/.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_URL and SUPABASE_API_KEY in .env")
    sys.exit(1)

import httpx

# ============================================
# CONFIGURATION
# ============================================

NUM_CITIES = 8
NUM_SCHOOLS = 15
NUM_CLUBS = 80  # ~5-6 clubs per school
NUM_STUDENTS = 5000  # ~330 students per school (realistic)
NUM_TEACHERS = 400  # ~27 teachers per school (~12:1 student ratio)
NUM_ROOMS = 300  # ~20 rooms per school
NUM_EVENTS = 150  # ~10 events per school

# Realistic data pools
CITY_DATA = [
    ("Boston", 685000, "USA", "EST"),
    ("Cambridge", 118000, "USA", "EST"),
    ("Newton", 88000, "USA", "EST"),
    ("Brookline", 59000, "USA", "EST"),
    ("Somerville", 81000, "USA", "EST"),
    ("Quincy", 101000, "USA", "EST"),
    ("Medford", 57000, "USA", "EST"),
    ("Malden", 66000, "USA", "EST"),
]

SCHOOL_NAMES = [
    "Lincoln High School", "Washington Academy", "Jefferson Prep",
    "Roosevelt Secondary", "Kennedy High", "Adams Institute",
    "Franklin Academy", "Hamilton High", "Madison Prep",
    "Monroe Secondary", "Jackson High School", "Harrison Academy",
    "Tyler Institute", "Polk Prep", "Taylor High",
]

CLUB_TYPES = {
    "sports": ["Basketball Club", "Soccer Club", "Tennis Club", "Swimming Club", 
               "Track & Field", "Volleyball Club", "Baseball Club", "Chess Club"],
    "academic": ["Math Club", "Science Olympiad", "Debate Team", "Model UN",
                 "Robotics Club", "Computer Science Club", "Literature Club", "History Club"],
    "arts": ["Drama Club", "Art Club", "Photography Club", "Film Club",
             "Dance Club", "Music Club", "Orchestra", "Band"],
    "social": ["Student Council", "Yearbook Committee", "Newspaper Club", "Environmental Club",
               "Community Service", "Cultural Club", "Language Club", "Cooking Club"],
}

FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
    "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Abigail", "Michael",
    "Emily", "Daniel", "Elizabeth", "Jacob", "Sofia", "Logan", "Avery", "Jackson",
    "Ella", "Sebastian", "Scarlett", "Aiden", "Grace", "Matthew", "Chloe", "Samuel",
    "Victoria", "David", "Riley", "Joseph", "Aria", "Carter", "Lily", "Owen",
    "Aubrey", "Wyatt", "Zoey", "John", "Hannah", "Jack", "Lillian", "Luke",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
]

SUBJECTS = ["Mathematics", "English", "Science", "History", "Art", "Music", 
            "Physical Education", "Computer Science", "Foreign Language", "Social Studies"]

MEETING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

ROOM_TYPES = ["classroom", "gym", "lab", "auditorium", "office", "library", "cafeteria"]

EVENT_TYPES = ["competition", "performance", "meeting", "fundraiser", "workshop", "conference"]

MEMBER_ROLES = ["member", "president", "vice_president", "secretary", "treasurer"]

PARTICIPANT_ROLES = ["participant", "organizer", "volunteer", "performer", "competitor"]


def random_date_between(start: date, end: date) -> date:
    """Generate a random date between start and end."""
    if start > end:
        start, end = end, start
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def generate_id(prefix: str, num: int) -> str:
    """Generate a formatted ID."""
    return f"{prefix}-{num:06d}"


# ============================================
# DATA GENERATION
# ============================================

def generate_cities() -> list[dict]:
    """Generate city data."""
    cities = []
    for i, (name, pop, country, tz) in enumerate(CITY_DATA[:NUM_CITIES], 1):
        cities.append({
            "id": generate_id("CTY", i),
            "name": name,
            "population": pop,
            "country": country,
            "timezone": tz,
        })
    return cities


def generate_schools() -> list[dict]:
    """Generate school data (NO city reference here!)."""
    schools = []
    for i in range(1, NUM_SCHOOLS + 1):
        schools.append({
            "id": generate_id("SCH", i),
            "name": SCHOOL_NAMES[i - 1] if i <= len(SCHOOL_NAMES) else f"School {i}",
            "square_meters": random.randint(5000, 50000),
            "floors": random.randint(1, 5),
            "rating": round(random.uniform(2.5, 5.0), 2),
            "founded_year": random.randint(1950, 2010),
        })
    return schools


def generate_clubs() -> list[dict]:
    """Generate club data (NO school reference here!)."""
    clubs = []
    club_id = 1
    
    # Generate clubs from all types
    for club_type, club_names in CLUB_TYPES.items():
        for name in club_names:
            if club_id > NUM_CLUBS:
                break
            clubs.append({
                "id": generate_id("CLB", club_id),
                "name": name,
                "club_type": club_type,
                "max_members": random.randint(10, 50),
                "meeting_day": random.choice(MEETING_DAYS),
                "parent_club_id": None,  # Will set some later
            })
            club_id += 1
    
    # Create some sub-clubs (parent-child relationships)
    for club in clubs[10:15]:
        parent = random.choice(clubs[:10])
        club["parent_club_id"] = parent["id"]
    
    return clubs


def generate_students() -> list[dict]:
    """Generate student data (NO school/club reference here!)."""
    students = []
    for i in range(1, NUM_STUDENTS + 1):
        birth_year = random.randint(2005, 2012)
        students.append({
            "id": generate_id("STU", i),
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "birth_date": f"{birth_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "gender": random.choice(["M", "F"]),
            "gpa": round(random.uniform(2.0, 4.0), 2),
        })
    return students


def generate_teachers() -> list[dict]:
    """Generate teacher data (NO school reference here!)."""
    teachers = []
    for i in range(1, NUM_TEACHERS + 1):
        teachers.append({
            "id": generate_id("TCH", i),
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "subject": random.choice(SUBJECTS),
            "hire_date": f"{random.randint(2000, 2023)}-{random.randint(1,12):02d}-01",
            "salary": random.randint(45000, 95000),
        })
    return teachers


def generate_rooms() -> list[dict]:
    """Generate room data (NO school reference here!)."""
    rooms = []
    for i in range(1, NUM_ROOMS + 1):
        room_type = random.choice(ROOM_TYPES)
        rooms.append({
            "id": generate_id("ROM", i),
            "room_number": f"{random.randint(1,5)}{random.randint(0,9)}{random.randint(0,9)}",
            "capacity": random.randint(20, 200) if room_type in ["gym", "auditorium", "cafeteria"] else random.randint(15, 40),
            "room_type": room_type,
            "has_projector": random.choice([True, False]),
        })
    return rooms


def generate_events() -> list[dict]:
    """Generate event data (NO school/club reference here!)."""
    events = []
    event_names = [
        "Science Fair", "Art Exhibition", "Music Concert", "Sports Day",
        "Math Competition", "Debate Tournament", "Drama Performance", "Dance Recital",
        "Book Fair", "Career Day", "Cultural Festival", "Talent Show",
        "Robotics Competition", "Chess Tournament", "Poetry Slam", "Film Festival",
        "Community Service Day", "Environmental Awareness Week", "Language Festival",
        "Photography Exhibition", "Coding Hackathon", "Model UN Conference",
    ]
    
    for i in range(1, NUM_EVENTS + 1):
        events.append({
            "id": generate_id("EVT", i),
            "name": event_names[i - 1] if i <= len(event_names) else f"Event {i}",
            "event_type": random.choice(EVENT_TYPES),
            "budget": round(random.uniform(100, 5000), 2),
            "is_recurring": random.choice([True, False]),
        })
    return events


# ============================================
# JUNCTION TABLE GENERATION
# ============================================

def generate_city_school_assignments(cities: list, schools: list) -> list[dict]:
    """Assign schools to cities via junction table."""
    assignments = []
    for i, school in enumerate(schools, 1):
        city = random.choice(cities)
        assignments.append({
            "id": generate_id("CSA", i),
            "city_id": city["id"],
            "school_id": school["id"],
            "established_date": f"{school['founded_year']}-09-01",
            "closed_date": None,
            "is_primary_location": True,
        })
    return assignments


def generate_room_school_assignments(rooms: list, schools: list) -> list[dict]:
    """Assign rooms to schools via junction table."""
    assignments = []
    rooms_per_school = len(rooms) // len(schools)
    
    for i, room in enumerate(rooms, 1):
        school = schools[i % len(schools)]
        assignments.append({
            "id": generate_id("RSA", i),
            "room_id": room["id"],
            "school_id": school["id"],
            "floor_number": random.randint(1, 4),
            "wing": random.choice(["North", "South", "East", "West", "Main"]),
        })
    return assignments


def generate_club_school_assignments(clubs: list, schools: list, rooms: list, room_assignments: list) -> list[dict]:
    """Assign clubs to schools via junction table."""
    assignments = []
    
    # Build room-to-school mapping
    room_to_school = {ra["room_id"]: ra["school_id"] for ra in room_assignments}
    school_rooms = defaultdict(list)
    for ra in room_assignments:
        school_rooms[ra["school_id"]].append(ra["room_id"])
    
    for i, club in enumerate(clubs, 1):
        school = random.choice(schools)
        # Assign a room from this school
        available_rooms = school_rooms.get(school["id"], [])
        room_id = random.choice(available_rooms) if available_rooms else None
        
        assignments.append({
            "id": generate_id("CLA", i),
            "club_id": club["id"],
            "school_id": school["id"],
            "start_date": f"{random.randint(2015, 2023)}-09-01",
            "end_date": None,
            "assigned_room_id": room_id,
            "budget_allocation": round(random.uniform(500, 5000), 2),
        })
    return assignments


def generate_student_enrollments(students: list, schools: list) -> list[dict]:
    """Enroll students in schools via junction table."""
    enrollments = []
    for i, student in enumerate(students, 1):
        school = random.choice(schools)
        birth_year = int(student["birth_date"][:4])
        enrollment_year = birth_year + 5  # Start school at age 5-6
        
        enrollments.append({
            "id": generate_id("ENR", i),
            "student_id": student["id"],
            "school_id": school["id"],
            "enrollment_date": f"{enrollment_year}-09-01",
            "graduation_date": None,  # Still enrolled
            "grade_level": min(12, 2024 - enrollment_year),
            "is_transfer": random.random() < 0.1,  # 10% are transfers
        })
    return enrollments


def generate_club_memberships(students: list, clubs: list, club_assignments: list) -> list[dict]:
    """Create club memberships via junction table."""
    memberships = []
    membership_id = 1
    
    # Realistic distribution: ~40% of students join clubs
    # 60% in 0 clubs, 25% in 1 club, 10% in 2 clubs, 5% in 3 clubs
    for student in students:
        num_clubs = random.choices([0, 1, 2, 3], weights=[60, 25, 10, 5])[0]
        if num_clubs == 0:
            continue
        joined_clubs = random.sample(clubs, min(num_clubs, len(clubs)))
        
        for club in joined_clubs:
            # Determine role (mostly members, few officers)
            if random.random() < 0.1:
                role = random.choice(["president", "vice_president", "secretary", "treasurer"])
            else:
                role = "member"
            
            memberships.append({
                "id": generate_id("MEM", membership_id),
                "student_id": student["id"],
                "club_id": club["id"],
                "role": role,
                "join_date": f"{random.randint(2020, 2024)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "leave_date": None if random.random() > 0.1 else f"2024-{random.randint(1,11):02d}-{random.randint(1,28):02d}",
                "is_active": random.random() > 0.1,
            })
            membership_id += 1
    
    return memberships


def generate_teacher_school_assignments(teachers: list, schools: list) -> list[dict]:
    """Assign teachers to schools via junction table."""
    assignments = []
    for i, teacher in enumerate(teachers, 1):
        school = random.choice(schools)
        assignments.append({
            "id": generate_id("TSA", i),
            "teacher_id": teacher["id"],
            "school_id": school["id"],
            "start_date": teacher["hire_date"],
            "end_date": None,
            "department": teacher["subject"],
            "is_department_head": random.random() < 0.15,
        })
    return assignments


def generate_club_advisors(teachers: list, clubs: list, teacher_assignments: list, club_assignments: list) -> list[dict]:
    """Assign teachers as club advisors via junction table."""
    advisors = []
    advisor_id = 1
    
    # Build teacher-to-school mapping
    teacher_schools = {ta["teacher_id"]: ta["school_id"] for ta in teacher_assignments}
    club_schools = {ca["club_id"]: ca["school_id"] for ca in club_assignments}
    
    for club in clubs:
        club_school = club_schools.get(club["id"])
        if not club_school:
            continue
        
        # Find teachers at this school
        school_teachers = [t for t in teachers if teacher_schools.get(t["id"]) == club_school]
        if not school_teachers:
            continue
        
        # Assign 1-2 advisors
        num_advisors = random.randint(1, 2)
        for j, teacher in enumerate(random.sample(school_teachers, min(num_advisors, len(school_teachers)))):
            advisors.append({
                "id": generate_id("ADV", advisor_id),
                "teacher_id": teacher["id"],
                "club_id": club["id"],
                "start_date": f"{random.randint(2018, 2023)}-09-01",
                "end_date": None,
                "is_primary_advisor": j == 0,
            })
            advisor_id += 1
    
    return advisors


def generate_event_locations(events: list, schools: list, rooms: list, room_assignments: list) -> list[dict]:
    """Assign events to schools/rooms via junction table."""
    locations = []
    
    # Build room-to-school mapping
    school_rooms = defaultdict(list)
    for ra in room_assignments:
        school_rooms[ra["school_id"]].append(ra["room_id"])
    
    for i, event in enumerate(events, 1):
        school = random.choice(schools)
        available_rooms = school_rooms.get(school["id"], [])
        room_id = random.choice(available_rooms) if available_rooms else None
        
        event_date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        start_hour = random.randint(8, 16)
        
        locations.append({
            "id": generate_id("ELO", i),
            "event_id": event["id"],
            "school_id": school["id"],
            "room_id": room_id,
            "event_date": event_date,
            "start_time": f"{start_hour:02d}:00:00",
            "end_time": f"{start_hour + random.randint(1, 4):02d}:00:00",
        })
    
    return locations


def generate_event_participants(events: list, students: list) -> list[dict]:
    """Create event participation records via junction table."""
    participants = []
    part_id = 1
    
    for event in events:
        # Each event has 5-30 participants
        num_participants = random.randint(5, 30)
        event_students = random.sample(students, min(num_participants, len(students)))
        
        for student in event_students:
            participants.append({
                "id": generate_id("EPT", part_id),
                "event_id": event["id"],
                "student_id": student["id"],
                "role": random.choice(PARTICIPANT_ROLES),
                "registration_date": f"2024-{random.randint(1,11):02d}-{random.randint(1,28):02d}",
                "attended": random.random() > 0.1,
            })
            part_id += 1
    
    return participants


def generate_event_organizers(events: list, clubs: list) -> list[dict]:
    """Assign clubs as event organizers via junction table."""
    organizers = []
    org_id = 1
    
    for event in events:
        # Each event organized by 1-3 clubs
        num_clubs = random.randint(1, 3)
        organizing_clubs = random.sample(clubs, min(num_clubs, len(clubs)))
        
        # Distribute contribution percentages
        contributions = [random.randint(20, 60) for _ in organizing_clubs]
        total = sum(contributions)
        contributions = [round(c / total * 100, 2) for c in contributions]
        
        for club, contrib in zip(organizing_clubs, contributions):
            organizers.append({
                "id": generate_id("EOG", org_id),
                "event_id": event["id"],
                "club_id": club["id"],
                "contribution_percent": contrib,
            })
            org_id += 1
    
    return organizers


def generate_student_scores(students: list) -> list[dict]:
    """Generate test scores for students."""
    scores = []
    score_id = 1
    
    semesters = ["Fall 2023", "Spring 2024", "Fall 2024"]
    
    for student in students:
        # Each student has 3-8 test scores
        num_scores = random.randint(3, 8)
        for _ in range(num_scores):
            max_score = random.choice([100, 50, 200])
            scores.append({
                "id": generate_id("SCR", score_id),
                "student_id": student["id"],
                "subject": random.choice(SUBJECTS),
                "score": round(random.uniform(max_score * 0.5, max_score), 2),
                "max_score": max_score,
                "test_date": f"2024-{random.randint(1,11):02d}-{random.randint(1,28):02d}",
                "semester": random.choice(semesters),
            })
            score_id += 1
    
    return scores


def generate_student_attendance(students: list, enrollments: list) -> list[dict]:
    """Generate attendance records for students."""
    attendance = []
    att_id = 1
    
    # Build student-to-school mapping
    student_schools = {e["student_id"]: e["school_id"] for e in enrollments}
    
    # Generate attendance for a few random days
    dates = [f"2024-{m:02d}-{d:02d}" for m in range(9, 12) for d in range(1, 29, 7)]
    
    for student in students:
        school_id = student_schools.get(student["id"])
        if not school_id:
            continue
        
        for att_date in random.sample(dates, min(5, len(dates))):
            status = random.choices(
                ["present", "absent", "late", "excused"],
                weights=[0.85, 0.05, 0.07, 0.03]
            )[0]
            
            attendance.append({
                "id": generate_id("ATT", att_id),
                "student_id": student["id"],
                "school_id": school_id,
                "attendance_date": att_date,
                "status": status,
                "notes": None if status == "present" else "Recorded by system",
            })
            att_id += 1
    
    return attendance


def generate_club_meetings(clubs: list, club_assignments: list, room_assignments: list) -> list[dict]:
    """Generate club meeting records."""
    meetings = []
    meeting_id = 1
    
    # Build school rooms mapping
    school_rooms = defaultdict(list)
    for ra in room_assignments:
        school_rooms[ra["school_id"]].append(ra["room_id"])
    
    club_schools = {ca["club_id"]: ca["school_id"] for ca in club_assignments}
    
    for club in clubs:
        school_id = club_schools.get(club["id"])
        if not school_id:
            continue
        
        available_rooms = school_rooms.get(school_id, [])
        
        # Each club has 5-15 meetings recorded
        num_meetings = random.randint(5, 15)
        for _ in range(num_meetings):
            meetings.append({
                "id": generate_id("MTG", meeting_id),
                "club_id": club["id"],
                "meeting_date": f"2024-{random.randint(1,11):02d}-{random.randint(1,28):02d}",
                "room_id": random.choice(available_rooms) if available_rooms else None,
                "duration_minutes": random.choice([30, 45, 60, 90, 120]),
                "attendee_count": random.randint(5, club["max_members"]),
                "notes": None,
            })
            meeting_id += 1
    
    return meetings


# ============================================
# DATABASE OPERATIONS
# ============================================

def insert_data(table: str, data: list[dict]) -> None:
    """Insert data into Supabase table."""
    if not data:
        return
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    
    # Insert in batches
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=headers,
            json=batch,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            print(f"  ERROR inserting into {table}: {resp.status_code} - {resp.text[:200]}")
            return
    
    print(f"  ✓ Inserted {len(data)} rows into {table}")


def clear_table(table: str) -> None:
    """Clear all data from a table."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "return=minimal",
    }
    
    resp = httpx.delete(
        f"{SUPABASE_URL}/rest/v1/{table}?id=neq.IMPOSSIBLE_VALUE",
        headers=headers,
        timeout=30,
    )
    if resp.status_code in (200, 204):
        print(f"  ✓ Cleared {table}")
    else:
        print(f"  Warning: Could not clear {table}: {resp.status_code}")


def populate_database() -> None:
    """Generate and insert all data."""
    print("\n" + "=" * 60)
    print("POPULATING SCHOOL DATABASE")
    print("=" * 60)
    
    # Clear existing data (in reverse dependency order)
    print("\nClearing existing data...")
    tables_to_clear = [
        "school_club_meetings",
        "school_student_attendance",
        "school_student_scores",
        "school_event_organizers",
        "school_event_participants",
        "school_event_locations",
        "school_club_advisors",
        "school_teacher_school_assignments",
        "school_club_memberships",
        "school_student_enrollments",
        "school_club_school_assignments",
        "school_room_school_assignments",
        "school_city_school_assignments",
        "school_events",
        "school_rooms",
        "school_teachers",
        "school_students",
        "school_clubs",
        "school_schools",
        "school_cities",
    ]
    for table in tables_to_clear:
        clear_table(table)
    
    # Generate core entities
    print("\nGenerating core entities...")
    cities = generate_cities()
    schools = generate_schools()
    clubs = generate_clubs()
    students = generate_students()
    teachers = generate_teachers()
    rooms = generate_rooms()
    events = generate_events()
    
    # Generate junction tables
    print("Generating junction tables...")
    city_school_assignments = generate_city_school_assignments(cities, schools)
    room_school_assignments = generate_room_school_assignments(rooms, schools)
    club_school_assignments = generate_club_school_assignments(clubs, schools, rooms, room_school_assignments)
    student_enrollments = generate_student_enrollments(students, schools)
    club_memberships = generate_club_memberships(students, clubs, club_school_assignments)
    teacher_school_assignments = generate_teacher_school_assignments(teachers, schools)
    club_advisors = generate_club_advisors(teachers, clubs, teacher_school_assignments, club_school_assignments)
    event_locations = generate_event_locations(events, schools, rooms, room_school_assignments)
    event_participants = generate_event_participants(events, students)
    event_organizers = generate_event_organizers(events, clubs)
    
    # Generate attribute tables
    print("Generating attribute tables...")
    student_scores = generate_student_scores(students)
    student_attendance = generate_student_attendance(students, student_enrollments)
    club_meetings = generate_club_meetings(clubs, club_school_assignments, room_school_assignments)
    
    # Insert data (in dependency order)
    print("\nInserting data...")
    insert_data("school_cities", cities)
    insert_data("school_schools", schools)
    insert_data("school_clubs", clubs)
    insert_data("school_students", students)
    insert_data("school_teachers", teachers)
    insert_data("school_rooms", rooms)
    insert_data("school_events", events)
    
    insert_data("school_city_school_assignments", city_school_assignments)
    insert_data("school_room_school_assignments", room_school_assignments)
    insert_data("school_club_school_assignments", club_school_assignments)
    insert_data("school_student_enrollments", student_enrollments)
    insert_data("school_club_memberships", club_memberships)
    insert_data("school_teacher_school_assignments", teacher_school_assignments)
    insert_data("school_club_advisors", club_advisors)
    insert_data("school_event_locations", event_locations)
    insert_data("school_event_participants", event_participants)
    insert_data("school_event_organizers", event_organizers)
    
    insert_data("school_student_scores", student_scores)
    insert_data("school_student_attendance", student_attendance)
    insert_data("school_club_meetings", club_meetings)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cities: {len(cities)}")
    print(f"Schools: {len(schools)}")
    print(f"Clubs: {len(clubs)}")
    print(f"Students: {len(students)}")
    print(f"Teachers: {len(teachers)}")
    print(f"Rooms: {len(rooms)}")
    print(f"Events: {len(events)}")
    print(f"Club memberships: {len(club_memberships)}")
    print(f"Event participants: {len(event_participants)}")
    print(f"Student scores: {len(student_scores)}")
    print(f"Attendance records: {len(student_attendance)}")
    print(f"Club meetings: {len(club_meetings)}")


def show_query_examples() -> None:
    """Show example complex queries."""
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLEX QUERIES")
    print("=" * 60)
    
    print("""
1. What city is student 'Emma Smith' in?
   (Requires: students → enrollments → schools → city_assignments → cities)
   
   SELECT c.name 
   FROM school_cities c
   JOIN school_city_school_assignments csa ON c.id = csa.city_id
   JOIN school_schools s ON s.id = csa.school_id
   JOIN school_student_enrollments se ON s.id = se.school_id
   JOIN school_students st ON st.id = se.student_id
   WHERE st.first_name = 'Emma' AND st.last_name = 'Smith';

2. What is the average GPA of students in the 'Chess Club' at 'Lincoln High School'?
   (Requires: clubs → club_school_assignments → schools, 
              clubs → memberships → students)
   
   SELECT AVG(st.gpa)
   FROM school_students st
   JOIN school_club_memberships cm ON st.id = cm.student_id
   JOIN school_clubs cl ON cl.id = cm.club_id
   JOIN school_club_school_assignments csa ON cl.id = csa.club_id
   JOIN school_schools s ON s.id = csa.school_id
   WHERE cl.name = 'Chess Club' AND s.name = 'Lincoln High School';

3. Which teacher advises the club with the most members at schools in 'Boston'?
   (Requires: cities → city_school_assignments → schools 
              → club_school_assignments → clubs → advisors → teachers
              + clubs → memberships for counting)

4. What room hosts the event with the highest budget?
   (Requires: events → event_locations → rooms)
   
5. List students who attend events organized by clubs they are members of.
   (Requires: students → memberships → clubs → event_organizers → events 
              → event_participants → students)
""")


def main():
    parser = argparse.ArgumentParser(description="Setup School Database")
    parser.add_argument("--populate", action="store_true", help="Populate database with data")
    parser.add_argument("--query-examples", action="store_true", help="Show example queries")
    
    args = parser.parse_args()
    
    if args.populate:
        populate_database()
    elif args.query_examples:
        show_query_examples()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

