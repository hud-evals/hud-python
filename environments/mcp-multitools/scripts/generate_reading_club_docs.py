#!/usr/bin/env python3
"""
Generate interconnected reading club documents with consistent data.
No duplicates, no contradictions - all data flows from single source of truth.

IMPORTANT: Literary preferences point to EXTERNAL book files, no embedded excerpts.
Model must navigate to actual book files to retrieve content.
"""

import random
import json
from datetime import datetime, timedelta

random.seed(42)  # Reproducible generation

# =============================================================================
# SINGLE SOURCE OF TRUTH - All data generated from here
# =============================================================================

FIRST_NAMES = [
    "Alice", "Benjamin", "Charlotte", "Daniel", "Eleanor", "Frederick", "Grace",
    "Henry", "Isabella", "James", "Katherine", "Leonard", "Margaret", "Nathan",
    "Olivia", "Patrick", "Quinn", "Rebecca", "Samuel", "Theresa", "Ulysses",
    "Victoria", "William", "Xena", "Yasmine", "Zachary", "Abigail", "Bernard",
    "Cecilia", "Douglas", "Elizabeth", "Francis", "Gertrude", "Harold", "Irene",
    "Jonathan", "Lillian", "Marcus", "Nora", "Oscar", "Penelope", "Raymond",
    "Sophia", "Theodore", "Uma", "Vincent", "Winifred", "Xavier", "Yvonne", "Zelda"
]

LAST_NAMES = [
    "Anderson", "Baker", "Campbell", "Davidson", "Edwards", "Fletcher", "Graham",
    "Harrison", "Irving", "Jenkins", "Knight", "Lambert", "Morrison", "Nelson",
    "O'Connor", "Patterson", "Quincy", "Richardson", "Stewart", "Thompson",
    "Underwood", "Vaughn", "Whitfield", "Xander", "Young", "Zimmerman", "Abbott",
    "Bennett", "Crawford", "Donovan", "Ellsworth", "Foster", "Griffith", "Hawkins",
    "Ingram", "Jacobs", "Keller", "Lawson", "Mitchell", "Newman", "Owens"
]

MEMBERSHIP_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
FAVORITE_GENRES = ["Classic Fiction", "Philosophy", "Adventure", "Gothic", "Fantasy"]

# Books with chapter references ONLY (no embedded text - that's in external files)
BOOKS = {
    "Pride and Prejudice": {
        "author": "Jane Austen",
        "doc_id": "pride_and_prejudice",
        "chapters": [1, 2, 3, 10, 20, 34, 45, 58]  # Available chapters to reference
    },
    "Moby Dick": {
        "author": "Herman Melville",
        "doc_id": "moby_dick",
        "chapters": [1, 2, 3, 10, 36, 42, 87, 133, 135]
    },
    "The Art of War": {
        "author": "Sun Tzu",
        "doc_id": "art_of_war",
        "chapters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    }
}

DISCUSSION_TOPICS = [
    "Character Development", "Themes of Mortality", "Social Commentary",
    "Narrative Structure", "Symbolism", "Historical Context", "Author's Intent",
    "Moral Dilemmas", "Literary Techniques", "Comparative Analysis"
]

# =============================================================================
# GENERATE MEMBERS (Single source of truth)
# =============================================================================

def generate_members(count=60):
    """Generate unique members with no duplicates."""
    members = []
    used_names = set()
    
    for i in range(count):
        while True:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                used_names.add(full_name)
                break
        
        member_id = f"RC-{1000 + i:04d}"
        join_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
        tier = random.choice(MEMBERSHIP_TIERS)
        genre = random.choice(FAVORITE_GENRES)
        books_read = random.randint(5, 50)
        
        members.append({
            "id": member_id,
            "first_name": first,
            "last_name": last,
            "full_name": full_name,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "tier": tier,
            "favorite_genre": genre,
            "books_read": books_read
        })
    
    return members

# =============================================================================
# GENERATE BOOK PREFERENCES (References only, no text)
# =============================================================================

def generate_book_preferences(members):
    """Generate which book/chapter each member marked as favorite (references only)."""
    preferences = {}
    
    for member in members:
        num_favorites = random.randint(1, 3)
        member_prefs = []
        
        used_combos = set()
        for _ in range(num_favorites):
            while True:
                book = random.choice(list(BOOKS.keys()))
                chapter = random.choice(BOOKS[book]["chapters"])
                combo = (book, chapter)
                if combo not in used_combos:
                    used_combos.add(combo)
                    member_prefs.append({
                        "book": book,
                        "chapter": chapter,
                        "rating": random.randint(7, 10)
                    })
                    break
        
        preferences[member["id"]] = member_prefs
    
    return preferences

# =============================================================================
# GENERATE DISCUSSION GROUPS
# =============================================================================

def generate_discussion_groups(members):
    """Generate discussion groups with member assignments."""
    groups = []
    member_ids = [m["id"] for m in members]
    
    group_names = [
        "The Page Turners", "Literary Lions", "Bookworm Brigade",
        "Chapter Chasers", "Novel Navigators", "Prose Pioneers",
        "Fiction Fanatics", "Classic Crusaders"
    ]
    
    for i, name in enumerate(group_names):
        group_size = random.randint(6, 12)
        group_members = random.sample(member_ids, min(group_size, len(member_ids)))
        focus_book = random.choice(list(BOOKS.keys()))
        
        groups.append({
            "group_id": f"DG-{i+1:02d}",
            "name": name,
            "focus_book": focus_book,
            "members": group_members,
            "meeting_day": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
            "discussion_topic": random.choice(DISCUSSION_TOPICS)
        })
    
    return groups

# =============================================================================
# GENERATE MEETING RECORDS
# =============================================================================

def generate_meetings(groups, members):
    """Generate meeting attendance records."""
    meetings = []
    member_lookup = {m["id"]: m for m in members}
    
    for group in groups:
        num_meetings = random.randint(3, 5)
        
        for meeting_num in range(1, num_meetings + 1):
            attendance_rate = random.uniform(0.6, 0.95)
            attendees = [mid for mid in group["members"] 
                        if random.random() < attendance_rate]
            
            meeting_date = datetime(2024, 1, 1) + timedelta(days=meeting_num * 14)
            
            focus_book = group["focus_book"]
            chapters_discussed = random.sample(
                BOOKS[focus_book]["chapters"],
                min(2, len(BOOKS[focus_book]["chapters"]))
            )
            
            meetings.append({
                "meeting_id": f"MTG-{group['group_id']}-{meeting_num:02d}",
                "group_id": group["group_id"],
                "group_name": group["name"],
                "date": meeting_date.strftime("%Y-%m-%d"),
                "book_discussed": focus_book,
                "chapters_discussed": chapters_discussed,
                "attendees": attendees,
                "topic": group["discussion_topic"]
            })
    
    return meetings

# =============================================================================
# WRITE DOCUMENT 1: MEMBER DIRECTORY
# =============================================================================

def write_member_directory(members, filepath):
    """Write the member directory document."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("RIVERSIDE READING CLUB - MEMBER DIRECTORY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("This directory lists all active club members organized by membership tier.")
    lines.append("Each member entry includes: ID, Name, Join Date, Favorite Genre, Books Read.")
    lines.append("Member IDs follow format: RC-XXXX")
    lines.append("")
    lines.append("SECTIONS:")
    
    tier_counts = {}
    for m in members:
        tier_counts[m["tier"]] = tier_counts.get(m["tier"], 0) + 1
    
    for tier in MEMBERSHIP_TIERS:
        lines.append(f"  {tier} Members: {tier_counts.get(tier, 0)} members")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    for tier in MEMBERSHIP_TIERS:
        tier_members = [m for m in members if m["tier"] == tier]
        
        lines.append(f"SECTION: {tier.upper()} MEMBERS")
        lines.append("-" * 50)
        lines.append("")
        
        for m in tier_members:
            lines.append(f"  {m['id']}")
            lines.append(f"    Name: {m['full_name']}")
            lines.append(f"    Joined: {m['join_date']}")
            lines.append(f"    Favorite Genre: {m['favorite_genre']}")
            lines.append(f"    Books Read: {m['books_read']}")
            lines.append("")
        
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Written: {filepath} ({len(lines)} lines)")

# =============================================================================
# WRITE DOCUMENT 2: DISCUSSION GROUPS & MEETINGS
# =============================================================================

def write_discussion_document(groups, meetings, members, filepath):
    """Write the discussion groups and meetings document."""
    lines = []
    member_lookup = {m["id"]: m for m in members}
    
    lines.append("=" * 70)
    lines.append("RIVERSIDE READING CLUB - DISCUSSION GROUPS & MEETINGS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("This document contains discussion group information and meeting records.")
    lines.append("Part 1: Discussion group definitions with assigned members")
    lines.append("Part 2: Meeting attendance records with chapters discussed")
    lines.append("")
    lines.append("CONTENTS:")
    lines.append(f"  Discussion Groups: {len(groups)}")
    lines.append(f"  Meeting Records: {len(meetings)}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    # Part 1: Discussion Groups
    lines.append("PART 1: DISCUSSION GROUPS")
    lines.append("=" * 50)
    lines.append("")
    
    for group in groups:
        lines.append(f"GROUP: {group['name']} ({group['group_id']})")
        lines.append("-" * 40)
        lines.append(f"  Focus Book: {group['focus_book']}")
        lines.append(f"  Meeting Day: {group['meeting_day']}")
        lines.append(f"  Current Topic: {group['discussion_topic']}")
        lines.append(f"  Members ({len(group['members'])}):")
        for mid in group['members']:
            member = member_lookup[mid]
            lines.append(f"    - {mid}: {member['full_name']}")
        lines.append("")
    
    lines.append("")
    
    # Part 2: Meeting Records
    lines.append("PART 2: MEETING RECORDS")
    lines.append("=" * 50)
    lines.append("")
    
    for meeting in meetings:
        lines.append(f"MEETING: {meeting['meeting_id']}")
        lines.append("-" * 40)
        lines.append(f"  Group: {meeting['group_name']}")
        lines.append(f"  Date: {meeting['date']}")
        lines.append(f"  Book: {meeting['book_discussed']}")
        lines.append(f"  Chapters Discussed: {', '.join(map(str, meeting['chapters_discussed']))}")
        lines.append(f"  Topic: {meeting['topic']}")
        lines.append(f"  Attendees ({len(meeting['attendees'])}):")
        for mid in meeting['attendees']:
            member = member_lookup[mid]
            lines.append(f"    - {member['full_name']} ({mid})")
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Written: {filepath} ({len(lines)} lines)")

# =============================================================================
# WRITE DOCUMENT 3: READING PREFERENCES (NO EMBEDDED TEXT - REFERENCES ONLY)
# =============================================================================

def write_reading_preferences(preferences, members, filepath):
    """Write member reading preferences - references to external book files only."""
    lines = []
    member_lookup = {m["id"]: m for m in members}
    
    lines.append("=" * 70)
    lines.append("RIVERSIDE READING CLUB - MEMBER READING PREFERENCES")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("This document records each member's favorite book chapters and ratings.")
    lines.append("Preferences are organized by member ID.")
    lines.append("")
    lines.append("IMPORTANT: Full book texts are available in separate documents:")
    lines.append("  - 'pride_and_prejudice' (Pride and Prejudice by Jane Austen)")
    lines.append("  - 'moby_dick' (Moby Dick by Herman Melville)")
    lines.append("  - 'art_of_war' (The Art of War by Sun Tzu)")
    lines.append("")
    lines.append("To read a favorite chapter, look up the referenced book and chapter")
    lines.append("in the corresponding document listed above.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    # Group by member
    for member_id in sorted(preferences.keys()):
        prefs = preferences[member_id]
        member = member_lookup[member_id]
        
        lines.append(f"MEMBER: {member['full_name']} ({member_id})")
        lines.append("-" * 40)
        lines.append(f"  Tier: {member['tier']}")
        lines.append(f"  Favorite Chapters:")
        
        for pref in prefs:
            book_info = BOOKS[pref['book']]
            lines.append(f"    - {pref['book']}, Chapter {pref['chapter']}")
            lines.append(f"      Rating: {pref['rating']}/10")
            lines.append(f"      (Full text in document: '{book_info['doc_id']}')")
        
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Written: {filepath} ({len(lines)} lines)")

# =============================================================================
# SAVE DATA FOR TASK GENERATION
# =============================================================================

def save_ground_truth(members, preferences, groups, meetings, filepath):
    """Save all data for task generation and verification."""
    data = {
        "members": members,
        "preferences": preferences,
        "groups": groups,
        "meetings": meetings,
        "books": {book: {"author": info["author"], "doc_id": info["doc_id"], "chapters": info["chapters"]} 
                 for book, info in BOOKS.items()}
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Written: {filepath}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Generating Reading Club Documents...")
    print("=" * 50)
    
    # Generate all data from single source
    members = generate_members(60)
    preferences = generate_book_preferences(members)
    groups = generate_discussion_groups(members)
    meetings = generate_meetings(groups, members)
    
    # Write documents
    write_member_directory(
        members, 
        "documents/reading_club_members.txt"
    )
    
    write_discussion_document(
        groups, meetings, members,
        "documents/reading_club_discussions.txt"
    )
    
    write_reading_preferences(
        preferences, members,
        "documents/reading_club_preferences.txt"  # Renamed from excerpts
    )
    
    # Save ground truth for task generation
    save_ground_truth(
        members, preferences, groups, meetings,
        "scripts/reading_club_data.json"
    )
    
    print("=" * 50)
    print("Done! Documents point to external book files:")
    print("  - pride_and_prejudice")
    print("  - moby_dick") 
    print("  - art_of_war")

if __name__ == "__main__":
    main()
