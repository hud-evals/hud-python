#!/usr/bin/env python3
"""
Populate football database with REALISTIC interconnected data.

Key feature: Match events (goals, cards) are stored by JERSEY NUMBER, not player name.
To find who scored, you must:
1. Get the match date
2. Look up jersey_number + team_id in football_jersey_assignments for that date
3. Get the player from football_players

NO existing tables are touched - only football_* tables are created/populated.
"""

import argparse
import asyncio
import os
import random
from datetime import datetime, timedelta, date
from decimal import Decimal
from dotenv import load_dotenv
import httpx

# Load .env from hud-python directory
load_dotenv("/home/rs/projects/hud-python/.env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_API_KEY") or os.getenv("SUPABASE_KEY")

# Fixed dates for deterministic data
CURRENT_DATE = date(2024, 11, 15)
SEASON_2024_START = date(2024, 8, 10)
SEASON_2024_END = date(2025, 5, 25)
SEASON_2023_START = date(2023, 8, 12)
SEASON_2023_END = date(2024, 5, 19)

# ============================================================
# DATA POOLS
# ============================================================

TEAMS_DATA = [
    ("TEAM-MNU", "Manchester United", "MNU", "Manchester", "England", "Old Trafford", 74310, 1878, "#DA291C", "#FFE500"),
    ("TEAM-MCI", "Manchester City", "MCI", "Manchester", "England", "Etihad Stadium", 53400, 1880, "#6CABDD", "#FFFFFF"),
    ("TEAM-LIV", "Liverpool", "LIV", "Liverpool", "England", "Anfield", 61276, 1892, "#C8102E", "#FFFFFF"),
    ("TEAM-CHE", "Chelsea", "CHE", "London", "England", "Stamford Bridge", 40341, 1905, "#034694", "#FFFFFF"),
    ("TEAM-ARS", "Arsenal", "ARS", "London", "England", "Emirates Stadium", 60704, 1886, "#EF0107", "#FFFFFF"),
    ("TEAM-TOT", "Tottenham Hotspur", "TOT", "London", "England", "Tottenham Stadium", 62850, 1882, "#132257", "#FFFFFF"),
    ("TEAM-NEW", "Newcastle United", "NEW", "Newcastle", "England", "St James Park", 52305, 1892, "#241F20", "#FFFFFF"),
    ("TEAM-WHU", "West Ham United", "WHU", "London", "England", "London Stadium", 62500, 1895, "#7A263A", "#1BB1E7"),
    ("TEAM-AVL", "Aston Villa", "AVL", "Birmingham", "England", "Villa Park", 42749, 1874, "#670E36", "#95BFE5"),
    ("TEAM-BRI", "Brighton", "BRI", "Brighton", "England", "Amex Stadium", 31800, 1901, "#0057B8", "#FFFFFF"),
    ("TEAM-EVE", "Everton", "EVE", "Liverpool", "England", "Goodison Park", 39414, 1878, "#003399", "#FFFFFF"),
    ("TEAM-LEI", "Leicester City", "LEI", "Leicester", "England", "King Power Stadium", 32261, 1884, "#003090", "#FDBE11"),
    ("TEAM-CRY", "Crystal Palace", "CRY", "London", "England", "Selhurst Park", 25486, 1905, "#1B458F", "#C4122E"),
    ("TEAM-WOL", "Wolverhampton", "WOL", "Wolverhampton", "England", "Molineux Stadium", 32050, 1877, "#FDB913", "#231F20"),
    ("TEAM-FUL", "Fulham", "FUL", "London", "England", "Craven Cottage", 25700, 1879, "#FFFFFF", "#000000"),
    ("TEAM-BOU", "Bournemouth", "BOU", "Bournemouth", "England", "Vitality Stadium", 11364, 1899, "#DA291C", "#000000"),
    ("TEAM-NFO", "Nottingham Forest", "NFO", "Nottingham", "England", "City Ground", 30445, 1865, "#DD0000", "#FFFFFF"),
    ("TEAM-BRE", "Brentford", "BRE", "London", "England", "Gtech Stadium", 17250, 1889, "#E30613", "#FFB81C"),
    ("TEAM-IPS", "Ipswich Town", "IPS", "Ipswich", "England", "Portman Road", 30311, 1878, "#0033A0", "#FFFFFF"),
    ("TEAM-SOU", "Southampton", "SOU", "Southampton", "England", "St Marys Stadium", 32384, 1885, "#D71920", "#FFFFFF"),
]

FIRST_NAMES = [
    "James", "Marcus", "Bruno", "Mason", "Bukayo", "Declan", "Cole", "Mohamed", "Erling", "Kevin",
    "Phil", "Bernardo", "Rodri", "Kyle", "Ruben", "Virgil", "Trent", "Andy", "Darwin", "Luis",
    "Enzo", "Moises", "Noni", "Nicolas", "Raheem", "Son", "James", "Pedro", "Brennan", "Dominic",
    "Alexander", "Ollie", "Ivan", "Joao", "Matheus", "Hwang", "Raul", "Daniel", "Emiliano", "Youri",
    "Anthony", "Casemiro", "Lisandro", "Alejandro", "Rasmus", "Andre", "Luke", "Harry", "Diogo",
    "Gabriel", "Martin", "Ben", "William", "Leandro", "Kaoru", "Josko", "Sandro", "Bryan", "Jean-Philippe",
    "Eberechi", "Michael", "Tyrick", "Marc", "Joachim", "Danny", "Jamie", "Adam", "Chris", "Conor",
    "Callum", "Morgan", "Antonee", "Pervis", "Carlos", "Miguel", "Omari", "Romeo", "Kobbie", "Amad"
]

LAST_NAMES = [
    "Fernandes", "Rashford", "Mount", "Saka", "Rice", "Palmer", "Salah", "Haaland", "De Bruyne", "Foden",
    "Silva", "Rodriguez", "Walker", "Dias", "Van Dijk", "Alexander-Arnold", "Robertson", "Nunez", "Diaz",
    "Fernandez", "Caicedo", "Madueke", "Jackson", "Sterling", "Heung-min", "Maddison", "Porro", "Johnson",
    "Solanke", "Isak", "Watkins", "Toney", "Palhinha", "Nunes", "Hee-chan", "Jimenez", "Podence", "Martinez",
    "Tielemans", "Gordon", "Mainoo", "Garnacho", "Hojlund", "Onana", "Shaw", "Kane", "Jota",
    "Jesus", "Odegaard", "White", "Saliba", "Trossard", "Mitoma", "Gvardiol", "Tonali", "Mbeumo", "Mateta",
    "Eze", "Olise", "Mitchell", "Guehi", "Andersen", "Ings", "Vardy", "Lallana", "Wood", "Gallagher",
    "Wilson", "Robinson", "Estupinan", "Almiron", "Guimaraes", "Hutchinson", "Lavia", "Mainoo", "Diallo"
]

NATIONALITIES = [
    "England", "Portugal", "Brazil", "Argentina", "France", "Spain", "Germany", "Belgium", 
    "Netherlands", "Norway", "South Korea", "Japan", "Senegal", "Egypt", "Colombia", "Uruguay",
    "Ecuador", "Ghana", "Ivory Coast", "Mali", "Morocco", "Algeria", "Nigeria", "Scotland",
    "Wales", "Ireland", "USA", "Canada", "Mexico", "Australia", "Croatia", "Serbia", "Ukraine"
]

POSITIONS = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "CF", "ST"]

def gen_id(prefix: str, num: int) -> str:
    return f"{prefix}-{num:06d}"

def random_date_between(start: date, end: date) -> date:
    delta = (end - start).days
    random_days = random.randint(0, delta)
    return start + timedelta(days=random_days)

# ============================================================
# DATA GENERATION
# ============================================================

def generate_teams() -> list[dict]:
    return [{
        "id": t[0], "name": t[1], "short_name": t[2], "city": t[3], "country": t[4],
        "stadium": t[5], "stadium_capacity": t[6], "founded_year": t[7],
        "primary_color": t[8], "secondary_color": t[9]
    } for t in TEAMS_DATA]


def generate_seasons() -> list[dict]:
    return [
        {"id": "SEASON-2324", "name": "2023-24 Season", "start_date": SEASON_2023_START.isoformat(), 
         "end_date": SEASON_2023_END.isoformat(), "is_current": False},
        {"id": "SEASON-2425", "name": "2024-25 Season", "start_date": SEASON_2024_START.isoformat(), 
         "end_date": SEASON_2024_END.isoformat(), "is_current": True},
    ]


def generate_players(n_per_team: int = 25) -> list[dict]:
    """Generate players with realistic career profiles.
    
    Career profiles:
    - STABLE (55%): One team, one jersey, loyal servant
    - JERSEY_HOPPER (10%): Same team, 2-4 jersey changes over career
    - CLUB_HOPPER (15%): 2-3 teams, jersey changes with each move
    - RETIRED (10%): Contract ended, no longer playing
    - YOUTH (10%): Recent academy graduate, might change numbers
    """
    random.seed(12345)
    players = []
    player_num = 100000
    
    used_names = set()
    
    # Career profile distribution
    PROFILES = ["STABLE"] * 55 + ["JERSEY_HOPPER"] * 10 + ["CLUB_HOPPER"] * 15 + ["RETIRED"] * 10 + ["YOUTH"] * 10
    
    for team_idx, team in enumerate(TEAMS_DATA):
        team_id = team[0]
        
        for i in range(n_per_team):
            player_num += 1
            
            # Assign career profile
            profile = random.choice(PROFILES)
            
            # Generate unique name
            attempts = 0
            while attempts < 100:
                first = random.choice(FIRST_NAMES)
                last = random.choice(LAST_NAMES)
                full_name = f"{first} {last}"
                if full_name not in used_names or attempts > 50:
                    used_names.add(full_name)
                    break
                attempts += 1
            
            # Position distribution: 3 GK, 8 DEF, 8 MID, 6 FWD per team
            if i < 3:
                position = "GK"
            elif i < 11:
                position = random.choice(["CB", "CB", "LB", "RB", "LWB", "RWB"])
            elif i < 19:
                position = random.choice(["CDM", "CM", "CM", "CAM", "LM", "RM"])
            else:
                position = random.choice(["LW", "RW", "CF", "ST", "ST"])
            
            # Age varies by profile
            if profile == "YOUTH":
                age = random.randint(17, 21)
            elif profile == "RETIRED":
                age = random.randint(34, 40)
            elif position == "GK":
                age = random.randint(22, 38)
            elif position in ["CB", "LB", "RB", "LWB", "RWB"]:
                age = random.randint(20, 35)
            elif position in ["CDM", "CM", "CAM", "LM", "RM"]:
                age = random.randint(19, 34)
            else:
                age = random.randint(18, 33)
            
            birth_year = CURRENT_DATE.year - age
            birth_date = date(birth_year, random.randint(1, 12), random.randint(1, 28))
            
            # Height based on position
            if position == "GK":
                height = random.randint(185, 200)
            elif position in ["CB"]:
                height = random.randint(180, 195)
            elif position in ["ST", "CF"]:
                height = random.randint(175, 195)
            else:
                height = random.randint(168, 188)
            
            players.append({
                "id": gen_id("PLY", player_num),
                "first_name": first,
                "last_name": last,
                "nationality": random.choice(NATIONALITIES),
                "birth_date": birth_date.isoformat(),
                "height_cm": height,
                "preferred_foot": random.choices(["right", "left", "both"], weights=[70, 25, 5])[0],
                "primary_position": position,
                "secondary_position": random.choice([p for p in POSITIONS if p != position]) if random.random() < 0.4 else None,
                "_career_profile": profile,  # Internal use only
                "_current_team_idx": team_idx  # Internal use only
            })
    
    return players


def generate_contracts(players: list, teams: list) -> list[dict]:
    """Generate player contracts based on career profiles.
    
    - STABLE: One long-term contract
    - JERSEY_HOPPER: One contract (same team)
    - CLUB_HOPPER: 2-3 contracts with different teams
    - RETIRED: Contract ended in the past
    - YOUTH: Recent short contract
    """
    random.seed(23456)
    contracts = []
    contract_num = 200000
    
    for player in players:
        profile = player.get("_career_profile", "STABLE")
        current_team_idx = player.get("_current_team_idx", 0)
        
        if profile == "STABLE":
            # Long-serving player, 3-8 years at the club
            contract_num += 1
            years_ago = random.randint(3, 8)
            contract_start = CURRENT_DATE - timedelta(days=years_ago * 365 + random.randint(0, 180))
            contract_years = random.randint(3, 6)
            contract_end = contract_start + timedelta(days=contract_years * 365)
            if contract_end < CURRENT_DATE:
                # Renewed
                contract_end = CURRENT_DATE + timedelta(days=random.randint(180, 730))
            
            contracts.append({
                "id": gen_id("CTR", contract_num),
                "player_id": player["id"],
                "team_id": teams[current_team_idx]["id"],
                "start_date": contract_start.isoformat(),
                "end_date": contract_end.isoformat(),
                "salary_weekly": random.randint(80000, 250000),
                "contract_type": "permanent",
                "_profile": profile
            })
            
        elif profile == "JERSEY_HOPPER":
            # Same team, long tenure
            contract_num += 1
            years_ago = random.randint(4, 7)
            contract_start = CURRENT_DATE - timedelta(days=years_ago * 365)
            contract_end = CURRENT_DATE + timedelta(days=random.randint(365, 1095))
            
            contracts.append({
                "id": gen_id("CTR", contract_num),
                "player_id": player["id"],
                "team_id": teams[current_team_idx]["id"],
                "start_date": contract_start.isoformat(),
                "end_date": contract_end.isoformat(),
                "salary_weekly": random.randint(100000, 300000),
                "contract_type": "permanent",
                "_profile": profile
            })
            
        elif profile == "CLUB_HOPPER":
            # Multiple teams - 2-3 contracts
            num_clubs = random.randint(2, 3)
            available_teams = [i for i in range(len(teams)) if i != current_team_idx]
            past_teams = random.sample(available_teams, min(num_clubs - 1, len(available_teams)))
            
            # Start from 5-7 years ago
            career_start = CURRENT_DATE - timedelta(days=random.randint(5, 7) * 365)
            current_date_ptr = career_start
            
            # Past contracts
            for past_team_idx in past_teams:
                contract_num += 1
                contract_length = random.randint(1, 3) * 365
                contract_end = current_date_ptr + timedelta(days=contract_length)
                
                contracts.append({
                    "id": gen_id("CTR", contract_num),
                    "player_id": player["id"],
                    "team_id": teams[past_team_idx]["id"],
                    "start_date": current_date_ptr.isoformat(),
                    "end_date": contract_end.isoformat(),
                    "salary_weekly": random.randint(50000, 200000),
                    "contract_type": "permanent",
                    "_profile": profile,
                    "_is_past": True
                })
                current_date_ptr = contract_end + timedelta(days=random.randint(1, 30))
            
            # Current contract
            contract_num += 1
            contracts.append({
                "id": gen_id("CTR", contract_num),
                "player_id": player["id"],
                "team_id": teams[current_team_idx]["id"],
                "start_date": current_date_ptr.isoformat(),
                "end_date": (CURRENT_DATE + timedelta(days=random.randint(365, 1095))).isoformat(),
                "salary_weekly": random.randint(120000, 350000),
                "contract_type": "permanent",
                "_profile": profile,
                "_is_past": False
            })
            
        elif profile == "RETIRED":
            # Contract ended, player retired
            contract_num += 1
            years_ago = random.randint(5, 10)
            contract_start = CURRENT_DATE - timedelta(days=years_ago * 365)
            # Retired 6 months to 2 years ago
            contract_end = CURRENT_DATE - timedelta(days=random.randint(180, 730))
            
            contracts.append({
                "id": gen_id("CTR", contract_num),
                "player_id": player["id"],
                "team_id": teams[current_team_idx]["id"],
                "start_date": contract_start.isoformat(),
                "end_date": contract_end.isoformat(),
                "salary_weekly": random.randint(60000, 180000),
                "contract_type": "permanent",
                "_profile": profile
            })
            
        elif profile == "YOUTH":
            # Recent academy graduate
            contract_num += 1
            years_ago = random.randint(0, 2)
            contract_start = CURRENT_DATE - timedelta(days=years_ago * 365 + random.randint(0, 180))
            contract_years = random.randint(2, 4)
            contract_end = contract_start + timedelta(days=contract_years * 365)
            
            contracts.append({
                "id": gen_id("CTR", contract_num),
                "player_id": player["id"],
                "team_id": teams[current_team_idx]["id"],
                "start_date": contract_start.isoformat(),
                "end_date": contract_end.isoformat(),
                "salary_weekly": random.randint(10000, 80000),
                "contract_type": "permanent",
                "_profile": profile
            })
    
    return contracts


def generate_jersey_assignments(players: list, contracts: list, teams: list) -> list[dict]:
    """Generate jersey number assignments based on career profiles.
    
    - STABLE: Never changes jersey number
    - JERSEY_HOPPER: 2-4 jersey changes on same team
    - CLUB_HOPPER: One jersey per contract (new number at each club)
    - RETIRED: One jersey (historical)
    - YOUTH: May change jersey 0-2 times (developing player)
    """
    random.seed(34567)
    assignments = []
    assignment_num = 300000
    
    # Group contracts by player
    player_contracts_map = {}
    for c in contracts:
        pid = c["player_id"]
        if pid not in player_contracts_map:
            player_contracts_map[pid] = []
        player_contracts_map[pid].append(c)
    
    # Sort each player's contracts by start date
    for pid in player_contracts_map:
        player_contracts_map[pid].sort(key=lambda x: x["start_date"])
    
    # Track all jersey assignments per team with date ranges
    team_jersey_history = {t["id"]: [] for t in teams}
    
    def is_number_available(team_id: str, jersey_num: int, start_date: str, end_date: str) -> bool:
        end_date = end_date or "9999-12-31"
        for num, s, e in team_jersey_history[team_id]:
            if num != jersey_num:
                continue
            e = e or "9999-12-31"
            if not (end_date < s or e < start_date):
                return False
        return True
    
    def find_available_jersey(team_id: str, position: str, start_date: str, end_date: str, exclude: list = None) -> int:
        exclude = exclude or []
        
        # Position-based preferences
        if position == "GK":
            preferred = [1, 13, 25, 31, 33, 40]
        elif position in ["CB", "LB", "RB", "LWB", "RWB"]:
            preferred = list(range(2, 7)) + [12, 15, 16, 21, 22, 23]
        elif position in ["CDM", "CM", "CAM", "LM", "RM"]:
            preferred = [6, 7, 8, 10, 11, 14, 17, 18, 20, 24]
        else:
            preferred = [7, 9, 10, 11, 14, 17, 18, 19, 20, 29]
        
        # Try preferred numbers first
        for num in preferred:
            if num not in exclude and is_number_available(team_id, num, start_date, end_date):
                return num
        
        # Then try any number
        for num in range(1, 100):
            if num not in exclude and is_number_available(team_id, num, start_date, end_date):
                return num
        
        return random.randint(50, 99)  # Fallback
    
    def add_assignment(player_id: str, team_id: str, jersey: int, start: str, end: str):
        nonlocal assignment_num
        assignment_num += 1
        assignments.append({
            "id": gen_id("JRS", assignment_num),
            "player_id": player_id,
            "team_id": team_id,
            "jersey_number": jersey,
            "start_date": start,
            "end_date": end
        })
        team_jersey_history[team_id].append((jersey, start, end))
    
    for player in players:
        profile = player.get("_career_profile", "STABLE")
        position = player["primary_position"]
        player_id = player["id"]
        player_contracts = player_contracts_map.get(player_id, [])
        
        if not player_contracts:
            continue
        
        if profile == "STABLE" or profile == "RETIRED":
            # Single jersey for entire career at one club
            c = player_contracts[0]  # Should be only one
            jersey = find_available_jersey(c["team_id"], position, c["start_date"], c["end_date"])
            add_assignment(player_id, c["team_id"], jersey, c["start_date"], c["end_date"])
            
        elif profile == "JERSEY_HOPPER":
            # Same team, but 2-4 jersey changes
            c = player_contracts[0]
            team_id = c["team_id"]
            contract_start = date.fromisoformat(c["start_date"])
            contract_end = date.fromisoformat(c["end_date"]) if c["end_date"] else CURRENT_DATE + timedelta(days=365)
            
            num_changes = random.randint(2, 4)
            tenure_days = (contract_end - contract_start).days
            if tenure_days < num_changes * 180:
                num_changes = max(1, tenure_days // 180)
            
            segment_days = tenure_days // num_changes
            used_numbers = []
            current_start = contract_start
            
            for i in range(num_changes):
                if i == num_changes - 1:
                    seg_end = c["end_date"]
                else:
                    seg_end = (current_start + timedelta(days=segment_days)).isoformat()
                
                jersey = find_available_jersey(team_id, position, current_start.isoformat(), seg_end, used_numbers)
                add_assignment(player_id, team_id, jersey, current_start.isoformat(), seg_end)
                used_numbers.append(jersey)
                
                if seg_end:
                    current_start = date.fromisoformat(seg_end) + timedelta(days=1)
            
        elif profile == "CLUB_HOPPER":
            # One jersey per contract (different teams)
            for c in player_contracts:
                jersey = find_available_jersey(c["team_id"], position, c["start_date"], c["end_date"])
                add_assignment(player_id, c["team_id"], jersey, c["start_date"], c["end_date"])
            
        elif profile == "YOUTH":
            # May change jersey 0-2 times as they develop
            c = player_contracts[0]
            team_id = c["team_id"]
            contract_start = date.fromisoformat(c["start_date"])
            contract_end = date.fromisoformat(c["end_date"]) if c["end_date"] else CURRENT_DATE + timedelta(days=365)
            
            tenure_days = (contract_end - contract_start).days
            
            # Young players might start with high numbers, then get better ones
            if tenure_days > 365 and random.random() < 0.5:
                # Started with high number, earned lower
                mid_point = random_date_between(contract_start + timedelta(days=180), contract_end - timedelta(days=90))
                
                # First assignment - high number (youth/reserve)
                high_nums = list(range(30, 60))
                first_jersey = None
                for num in high_nums:
                    if is_number_available(team_id, num, c["start_date"], mid_point.isoformat()):
                        first_jersey = num
                        break
                if first_jersey is None:
                    first_jersey = find_available_jersey(team_id, position, c["start_date"], mid_point.isoformat())
                
                add_assignment(player_id, team_id, first_jersey, c["start_date"], mid_point.isoformat())
                
                # Second assignment - earned a real number
                second_jersey = find_available_jersey(team_id, position, (mid_point + timedelta(days=1)).isoformat(), c["end_date"], [first_jersey])
                add_assignment(player_id, team_id, second_jersey, (mid_point + timedelta(days=1)).isoformat(), c["end_date"])
            else:
                # Single assignment
                jersey = find_available_jersey(team_id, position, c["start_date"], c["end_date"])
                add_assignment(player_id, team_id, jersey, c["start_date"], c["end_date"])
    
    return assignments


def generate_matches(teams: list, seasons: list) -> list[dict]:
    """Generate matches - each team plays each other twice per season.
    
    IMPORTANT: Ensures no team plays twice on the same day!
    """
    random.seed(45678)
    matches = []
    match_num = 400000
    
    for season in seasons:
        season_start = date.fromisoformat(season["start_date"])
        season_end = date.fromisoformat(season["end_date"])
        
        # Only generate completed matches (before current date)
        effective_end = min(season_end, CURRENT_DATE)
        days_in_season = (effective_end - season_start).days
        if days_in_season <= 0:
            continue
        
        # Track which teams play on which days to avoid conflicts
        team_match_days = {t["id"]: set() for t in teams}
        
        # Generate all fixtures first (home and away)
        fixtures = []
        for i, home_team in enumerate(teams):
            for j, away_team in enumerate(teams):
                if i != j:
                    fixtures.append((home_team, away_team))
        
        # Shuffle fixtures and assign dates ensuring no team plays twice per day
        random.shuffle(fixtures)
        
        for home_team, away_team in fixtures:
            match_num += 1
            
            # Find a day where neither team is playing
            attempts = 0
            while attempts < 100:
                match_day = random.randint(0, days_in_season)
                match_date = season_start + timedelta(days=match_day)
                
                if match_date > CURRENT_DATE:
                    attempts += 1
                    continue
                    
                date_str = match_date.isoformat()
                if date_str not in team_match_days[home_team["id"]] and \
                   date_str not in team_match_days[away_team["id"]]:
                    break
                attempts += 1
            
            if attempts >= 100:
                continue  # Skip this fixture if can't find a valid date
            
            # Mark both teams as playing on this day
            team_match_days[home_team["id"]].add(date_str)
            team_match_days[away_team["id"]].add(date_str)
            
            # Realistic scores
            home_advantage = 0.3
            home_attack = random.gauss(1.5 + home_advantage, 0.5)
            away_attack = random.gauss(1.2, 0.5)
            
            home_score = max(0, int(random.gauss(home_attack, 1.0)))
            away_score = max(0, int(random.gauss(away_attack, 1.0)))
            
            # Cap at reasonable scores
            home_score = min(home_score, 7)
            away_score = min(away_score, 6)
            
            kickoff_hours = random.choice([12, 14, 15, 15, 15, 17, 17, 19, 20, 20])
            
            matches.append({
                "id": gen_id("MTH", match_num),
                "season_id": season["id"],
                "home_team_id": home_team["id"],
                "away_team_id": away_team["id"],
                "match_date": date_str,
                "kickoff_time": f"{kickoff_hours:02d}:00:00",
                "venue": home_team["stadium"],
                "home_score": home_score,
                "away_score": away_score,
                "attendance": int(home_team["stadium_capacity"] * random.uniform(0.85, 1.0)),
                "status": "completed"
            })
    
    return matches


def generate_lineups_and_events(matches: list, players: list, jerseys: list, teams: list) -> tuple[list, list]:
    """Generate lineups and match events.
    
    CRITICAL: Events are stored by JERSEY NUMBER, not player ID!
    """
    random.seed(56789)
    lineups = []
    events = []
    lineup_num = 500000
    event_num = 600000
    
    # Build lookup: team_id -> list of (player_id, jersey_num, start_date, end_date)
    team_players = {t["id"]: [] for t in teams}
    for j in jerseys:
        team_players[j["team_id"]].append({
            "player_id": j["player_id"],
            "jersey_number": j["jersey_number"],
            "start_date": j["start_date"],
            "end_date": j["end_date"]
        })
    
    # Build player info lookup
    player_info = {p["id"]: p for p in players}
    
    def get_jersey_for_player_on_date(team_id: str, player_id: str, match_date: str) -> int:
        """Find what jersey number a player wore on a specific date."""
        for jp in team_players[team_id]:
            if jp["player_id"] != player_id:
                continue
            start = jp["start_date"]
            end = jp["end_date"]
            if start <= match_date and (end is None or end >= match_date):
                return jp["jersey_number"]
        return None
    
    def get_available_players_for_team(team_id: str, match_date: str) -> list:
        """Get all players available for a team on a date with their jersey numbers.
        
        Note: Retired players (contract ended) won't appear since they won't have
        a valid jersey assignment. JERSEY_HOPPER players with multiple jersey
        assignments will only match one (the active one on that date).
        """
        available = []
        seen_players = set()  # Deduplicate by player_id
        
        for jp in team_players[team_id]:
            if jp["player_id"] in seen_players:
                continue
                
            start = jp["start_date"]
            end = jp["end_date"]
            if start <= match_date and (end is None or end >= match_date):
                player = player_info.get(jp["player_id"])
                if player:
                    available.append({
                        "player_id": jp["player_id"],
                        "jersey_number": jp["jersey_number"],
                        "position": player["primary_position"]
                    })
                    seen_players.add(jp["player_id"])
        return available
    
    for match in matches:
        match_date = match["match_date"]
        home_team = match["home_team_id"]
        away_team = match["away_team_id"]
        home_score = match["home_score"]
        away_score = match["away_score"]
        
        # Get available players for each team
        home_available = get_available_players_for_team(home_team, match_date)
        away_available = get_available_players_for_team(away_team, match_date)
        
        if len(home_available) < 11 or len(away_available) < 11:
            continue  # Skip matches without enough players
        
        # Sort by position to build lineup: 1 GK, 4 DEF, 4 MID, 2 FWD typical
        def sort_by_position(p):
            pos_order = {"GK": 0, "CB": 1, "LB": 1, "RB": 1, "LWB": 1, "RWB": 1, 
                        "CDM": 2, "CM": 2, "CAM": 2, "LM": 2, "RM": 2,
                        "LW": 3, "RW": 3, "CF": 3, "ST": 3}
            return pos_order.get(p["position"], 2)
        
        home_available.sort(key=sort_by_position)
        away_available.sort(key=sort_by_position)
        
        # Pick 11 starters + 3-5 subs
        home_starters = home_available[:11]
        home_subs = home_available[11:16] if len(home_available) > 11 else []
        away_starters = away_available[:11]
        away_subs = away_available[11:16] if len(away_available) > 11 else []
        
        # Generate lineups
        for starter in home_starters:
            lineup_num += 1
            lineups.append({
                "id": gen_id("LNP", lineup_num),
                "match_id": match["id"],
                "team_id": home_team,
                "player_id": starter["player_id"],
                "jersey_number": starter["jersey_number"],
                "position_played": starter["position"],
                "is_starter": True,
                "subbed_in_minute": None,
                "subbed_out_minute": random.choice([None, None, None, 70, 75, 80, 85]) if random.random() < 0.3 else None,
                "rating": round(random.uniform(5.5, 9.0), 1)
            })
        
        for starter in away_starters:
            lineup_num += 1
            lineups.append({
                "id": gen_id("LNP", lineup_num),
                "match_id": match["id"],
                "team_id": away_team,
                "player_id": starter["player_id"],
                "jersey_number": starter["jersey_number"],
                "position_played": starter["position"],
                "is_starter": True,
                "subbed_in_minute": None,
                "subbed_out_minute": random.choice([None, None, None, 70, 75, 80, 85]) if random.random() < 0.3 else None,
                "rating": round(random.uniform(5.5, 9.0), 1)
            })
        
        # Generate GOAL events - stored by JERSEY NUMBER!
        home_scorers = [s for s in home_starters if s["position"] not in ["GK"]]
        away_scorers = [s for s in away_starters if s["position"] not in ["GK"]]
        
        for _ in range(home_score):
            if not home_scorers:
                continue
            event_num += 1
            scorer = random.choice(home_scorers)
            assister = random.choice([s for s in home_scorers if s != scorer]) if random.random() < 0.7 else None
            
            events.append({
                "id": gen_id("EVT", event_num),
                "match_id": match["id"],
                "team_id": home_team,
                "jersey_number": scorer["jersey_number"],  # KEY: Jersey number, not player!
                "event_type": "goal",
                "minute": random.randint(1, 90),
                "added_time": random.choice([0, 0, 0, 0, 1, 2, 3]) if random.random() < 0.1 else 0,
                "assist_jersey_number": assister["jersey_number"] if assister else None,
                "description": None
            })
        
        for _ in range(away_score):
            if not away_scorers:
                continue
            event_num += 1
            scorer = random.choice(away_scorers)
            assister = random.choice([s for s in away_scorers if s != scorer]) if random.random() < 0.7 else None
            
            events.append({
                "id": gen_id("EVT", event_num),
                "match_id": match["id"],
                "team_id": away_team,
                "jersey_number": scorer["jersey_number"],
                "event_type": "goal",
                "minute": random.randint(1, 90),
                "added_time": 0,
                "assist_jersey_number": assister["jersey_number"] if assister else None,
                "description": None
            })
        
        # Generate some cards (yellow/red)
        all_players = home_starters + away_starters
        num_yellows = random.choices([0, 1, 2, 3, 4, 5], weights=[10, 25, 30, 20, 10, 5])[0]
        num_reds = random.choices([0, 1], weights=[95, 5])[0]
        
        for _ in range(num_yellows):
            event_num += 1
            player = random.choice(all_players)
            team = home_team if player in home_starters else away_team
            events.append({
                "id": gen_id("EVT", event_num),
                "match_id": match["id"],
                "team_id": team,
                "jersey_number": player["jersey_number"],
                "event_type": "yellow_card",
                "minute": random.randint(1, 90),
                "added_time": 0,
                "assist_jersey_number": None,
                "description": random.choice(["Foul", "Tactical foul", "Dissent", "Time wasting", "Simulation"])
            })
        
        for _ in range(num_reds):
            event_num += 1
            player = random.choice(all_players)
            team = home_team if player in home_starters else away_team
            events.append({
                "id": gen_id("EVT", event_num),
                "match_id": match["id"],
                "team_id": team,
                "jersey_number": player["jersey_number"],
                "event_type": "red_card",
                "minute": random.randint(20, 90),
                "added_time": 0,
                "assist_jersey_number": None,
                "description": random.choice(["Violent conduct", "Serious foul play", "Denying obvious goal-scoring opportunity"])
            })
    
    return lineups, events


def generate_player_stats(lineups: list, matches: list) -> list[dict]:
    """Generate per-match statistics for players."""
    random.seed(67890)
    stats = []
    stat_num = 700000
    
    match_lookup = {m["id"]: m for m in matches}
    
    for lineup in lineups:
        if not lineup["is_starter"]:
            continue
            
        stat_num += 1
        match = match_lookup.get(lineup["match_id"])
        if not match:
            continue
        
        minutes = 90
        if lineup["subbed_out_minute"]:
            minutes = lineup["subbed_out_minute"]
        if lineup["subbed_in_minute"]:
            minutes = 90 - lineup["subbed_in_minute"]
        
        position = lineup["position_played"]
        
        # Stats vary by position
        if position == "GK":
            stats.append({
                "id": gen_id("STS", stat_num),
                "match_id": lineup["match_id"],
                "player_id": lineup["player_id"],
                "minutes_played": minutes,
                "goals": 0,
                "assists": 0,
                "shots": 0,
                "shots_on_target": 0,
                "passes_completed": random.randint(15, 40),
                "passes_attempted": random.randint(20, 50),
                "tackles": 0,
                "interceptions": random.randint(0, 2),
                "fouls_committed": random.randint(0, 1),
                "fouls_suffered": random.randint(0, 1),
                "saves": random.randint(0, 8),
                "distance_covered_km": round(random.uniform(4.0, 6.5), 2)
            })
        elif position in ["CB", "LB", "RB", "LWB", "RWB"]:
            stats.append({
                "id": gen_id("STS", stat_num),
                "match_id": lineup["match_id"],
                "player_id": lineup["player_id"],
                "minutes_played": minutes,
                "goals": 0 if random.random() > 0.05 else 1,
                "assists": 0 if random.random() > 0.1 else 1,
                "shots": random.randint(0, 2),
                "shots_on_target": random.randint(0, 1),
                "passes_completed": random.randint(30, 70),
                "passes_attempted": random.randint(40, 85),
                "tackles": random.randint(1, 6),
                "interceptions": random.randint(1, 5),
                "fouls_committed": random.randint(0, 3),
                "fouls_suffered": random.randint(0, 2),
                "saves": 0,
                "distance_covered_km": round(random.uniform(9.0, 12.0), 2)
            })
        elif position in ["CDM", "CM", "CAM", "LM", "RM"]:
            stats.append({
                "id": gen_id("STS", stat_num),
                "match_id": lineup["match_id"],
                "player_id": lineup["player_id"],
                "minutes_played": minutes,
                "goals": 0 if random.random() > 0.15 else random.randint(1, 2),
                "assists": 0 if random.random() > 0.2 else 1,
                "shots": random.randint(0, 4),
                "shots_on_target": random.randint(0, 2),
                "passes_completed": random.randint(35, 80),
                "passes_attempted": random.randint(45, 95),
                "tackles": random.randint(1, 5),
                "interceptions": random.randint(0, 4),
                "fouls_committed": random.randint(0, 3),
                "fouls_suffered": random.randint(1, 4),
                "saves": 0,
                "distance_covered_km": round(random.uniform(10.0, 13.5), 2)
            })
        else:  # Forwards
            stats.append({
                "id": gen_id("STS", stat_num),
                "match_id": lineup["match_id"],
                "player_id": lineup["player_id"],
                "minutes_played": minutes,
                "goals": 0 if random.random() > 0.25 else random.randint(1, 3),
                "assists": 0 if random.random() > 0.15 else 1,
                "shots": random.randint(1, 6),
                "shots_on_target": random.randint(0, 4),
                "passes_completed": random.randint(15, 35),
                "passes_attempted": random.randint(20, 45),
                "tackles": random.randint(0, 2),
                "interceptions": random.randint(0, 2),
                "fouls_committed": random.randint(0, 2),
                "fouls_suffered": random.randint(1, 5),
                "saves": 0,
                "distance_covered_km": round(random.uniform(9.0, 12.0), 2)
            })
    
    return stats


def generate_weights(players: list, days: int = 90) -> list[dict]:
    """Generate daily weight records for players."""
    random.seed(78901)
    weights = []
    weight_num = 800000
    
    for player in players[:100]:  # Limit to first 100 players for manageable data
        # Base weight based on height and position
        height = player.get("height_cm", 180)
        position = player["primary_position"]
        
        if position == "GK":
            base_weight = height * 0.42 + random.uniform(-5, 5)
        elif position in ["CB"]:
            base_weight = height * 0.44 + random.uniform(-3, 5)
        elif position in ["ST", "CF"]:
            base_weight = height * 0.42 + random.uniform(-3, 5)
        else:
            base_weight = height * 0.40 + random.uniform(-3, 3)
        
        # Generate daily weights with small variations
        for d in range(days):
            weight_num += 1
            record_date = CURRENT_DATE - timedelta(days=d)
            
            # Small daily fluctuation
            daily_weight = base_weight + random.gauss(0, 0.5)
            body_fat = random.uniform(8, 15) if random.random() < 0.3 else None
            
            weights.append({
                "id": gen_id("WGT", weight_num),
                "player_id": player["id"],
                "recorded_date": record_date.isoformat(),
                "weight_kg": round(daily_weight, 2),
                "body_fat_percent": round(body_fat, 1) if body_fat else None,
                "notes": None if random.random() > 0.05 else "Post-match recovery"
            })
    
    return weights


def generate_diets(players: list, days: int = 60) -> list[dict]:
    """Generate daily diet records for players."""
    random.seed(89012)
    diets = []
    diet_num = 900000
    
    meal_plans = ["training", "match_day", "recovery", "rest", "travel"]
    
    for player in players[:100]:
        for d in range(days):
            diet_num += 1
            diet_date = CURRENT_DATE - timedelta(days=d)
            
            # Calories vary by day type
            day_type = random.choice(meal_plans)
            if day_type == "match_day":
                calories = random.randint(3200, 4000)
                carbs = random.randint(400, 500)
            elif day_type == "training":
                calories = random.randint(2800, 3500)
                carbs = random.randint(300, 400)
            elif day_type == "recovery":
                calories = random.randint(2500, 3200)
                carbs = random.randint(250, 350)
            else:
                calories = random.randint(2200, 2800)
                carbs = random.randint(200, 300)
            
            protein = random.randint(150, 220)
            fat = random.randint(60, 100)
            water = round(random.uniform(2.5, 4.5), 1)
            
            diets.append({
                "id": gen_id("DIT", diet_num),
                "player_id": player["id"],
                "diet_date": diet_date.isoformat(),
                "total_calories": calories,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "water_liters": water,
                "meal_plan": day_type
            })
    
    return diets


def generate_transfers(players: list, contracts: list, teams: list) -> list[dict]:
    """Generate transfer history based on actual contract changes.
    
    Transfers are derived from CLUB_HOPPER players who have multiple contracts.
    """
    random.seed(90123)
    transfers = []
    transfer_num = 950000
    
    # Group contracts by player
    player_contracts_map = {}
    for c in contracts:
        pid = c["player_id"]
        if pid not in player_contracts_map:
            player_contracts_map[pid] = []
        player_contracts_map[pid].append(c)
    
    # Sort each player's contracts by start date
    for pid in player_contracts_map:
        player_contracts_map[pid].sort(key=lambda x: x["start_date"])
    
    player_lookup = {p["id"]: p for p in players}
    
    for player_id, player_ctrs in player_contracts_map.items():
        if len(player_ctrs) < 2:
            continue  # No transfer if only one contract
        
        player = player_lookup.get(player_id)
        if not player:
            continue
        
        position = player["primary_position"]
        
        # Generate transfer for each contract change
        for i in range(1, len(player_ctrs)):
            transfer_num += 1
            from_contract = player_ctrs[i-1]
            to_contract = player_ctrs[i]
            
            # Transfer fee based on position and randomness
            if position in ["ST", "CF", "LW", "RW", "CAM"]:
                fee = random.randint(5000, 100000) * 1000  # £5M - £100M
            elif position in ["CM", "CDM"]:
                fee = random.randint(3000, 70000) * 1000
            elif position in ["CB", "LB", "RB"]:
                fee = random.randint(2000, 50000) * 1000
            else:  # GK
                fee = random.randint(1000, 40000) * 1000
            
            transfer_type = random.choices(
                ["permanent", "loan", "free"],
                weights=[60, 25, 15]
            )[0]
            
            if transfer_type == "free":
                fee = 0
            elif transfer_type == "loan":
                fee = fee // 10
            
            transfers.append({
                "id": gen_id("TRF", transfer_num),
                "player_id": player_id,
                "from_team_id": from_contract["team_id"],
                "to_team_id": to_contract["team_id"],
                "transfer_date": to_contract["start_date"],
                "transfer_fee": fee,
                "transfer_type": transfer_type
            })
    
    return transfers


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def validate_data_consistency(jerseys: list, events: list, matches: list, lineups: list) -> list[str]:
    """Validate data for contradictions and return list of issues."""
    issues = []
    
    # Build lookup: (team_id, jersey_num, date) -> player_id
    def get_player_for_jersey(team_id: str, jersey_num: int, match_date: str) -> str:
        for j in jerseys:
            if j["team_id"] != team_id or j["jersey_number"] != jersey_num:
                continue
            start = j["start_date"]
            end = j["end_date"]
            if start <= match_date and (end is None or end >= match_date):
                return j["player_id"]
        return None
    
    # Build match date lookup
    match_dates = {m["id"]: m["match_date"] for m in matches}
    
    # Check 1: Every event's jersey number should exist for that team on that date
    for event in events:
        match_date = match_dates.get(event["match_id"])
        if not match_date:
            issues.append(f"Event {event['id']} references non-existent match {event['match_id']}")
            continue
        
        player = get_player_for_jersey(event["team_id"], event["jersey_number"], match_date)
        if not player:
            issues.append(f"Event {event['id']}: No player found with jersey #{event['jersey_number']} for team {event['team_id']} on {match_date}")
    
    # Check 2: No duplicate jersey numbers on same team at same time
    # Group jerseys by team
    team_jerseys = {}
    for j in jerseys:
        key = j["team_id"]
        if key not in team_jerseys:
            team_jerseys[key] = []
        team_jerseys[key].append(j)
    
    for team_id, team_js in team_jerseys.items():
        # Check each pair for overlap
        for i, j1 in enumerate(team_js):
            for j2 in team_js[i+1:]:
                if j1["jersey_number"] != j2["jersey_number"]:
                    continue
                # Same jersey number - check if date ranges overlap
                s1, e1 = j1["start_date"], j1["end_date"] or "9999-12-31"
                s2, e2 = j2["start_date"], j2["end_date"] or "9999-12-31"
                if not (e1 < s2 or e2 < s1):  # Ranges overlap
                    issues.append(f"Jersey #{j1['jersey_number']} conflict on {team_id}: {j1['player_id']} ({s1}-{e1}) and {j2['player_id']} ({s2}-{e2})")
    
    # Check 3: Lineup jersey numbers match jersey assignments
    for lineup in lineups[:100]:  # Sample check
        match_date = match_dates.get(lineup["match_id"])
        if not match_date:
            continue
        player = get_player_for_jersey(lineup["team_id"], lineup["jersey_number"], match_date)
        if player != lineup["player_id"]:
            issues.append(f"Lineup {lineup['id']}: Jersey #{lineup['jersey_number']} assigned to {player}, but lineup says {lineup['player_id']}")
    
    return issues


def clean_for_insert(data: list[dict]) -> list[dict]:
    """Remove internal fields (prefixed with _) before inserting to Supabase."""
    cleaned = []
    for item in data:
        cleaned_item = {k: v for k, v in item.items() if not k.startswith("_")}
        cleaned.append(cleaned_item)
    return cleaned


async def clear_all_football_tables(client: httpx.AsyncClient):
    """Clear all football tables in correct order (respecting foreign keys)."""
    # Must delete in reverse dependency order
    tables_to_clear = [
        "football_transfers",
        "football_player_weights",
        "football_player_diets", 
        "football_player_match_stats",
        "football_match_events",
        "football_match_lineups",
        "football_matches",
        "football_jersey_assignments",
        "football_player_contracts",
        "football_players",
        "football_seasons",
        "football_teams",
    ]
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    
    print("  Clearing existing data...")
    for table in tables_to_clear:
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        try:
            resp = await client.delete(url, headers=headers, params={"id": "not.is.null"})
            if resp.status_code in (200, 204):
                pass  # Silent success
            elif resp.status_code == 404:
                pass  # Table doesn't exist yet
            else:
                print(f"    Warning: {table} clear returned {resp.status_code}")
        except Exception as e:
            print(f"    Warning: {table} clear failed: {e}")


async def insert_data(client: httpx.AsyncClient, table: str, data: list[dict], skip_clear: bool = True) -> bool:
    if not data:
        print(f"  ⚠ {table}: no data to insert")
        return True
    
    # Remove internal fields
    data = clean_for_insert(data)
    
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    try:
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
    
    print("Generating football data...")
    teams = generate_teams()
    seasons = generate_seasons()
    players = generate_players(25)  # 25 players per team = 500 total
    contracts = generate_contracts(players, teams)
    jerseys = generate_jersey_assignments(players, contracts, teams)
    matches = generate_matches(teams, seasons)
    lineups, events = generate_lineups_and_events(matches, players, jerseys, teams)
    stats = generate_player_stats(lineups, matches)
    weights = generate_weights(players, 90)
    diets = generate_diets(players, 60)
    transfers = generate_transfers(players, contracts, teams)
    
    print(f"\n  ⚽ {len(teams)} teams")
    print(f"  📅 {len(seasons)} seasons")
    print(f"  👤 {len(players)} players")
    print(f"  📝 {len(contracts)} contracts")
    print(f"  👕 {len(jerseys)} jersey assignments")
    print(f"  🏟️ {len(matches)} matches")
    print(f"  📋 {len(lineups)} lineups")
    print(f"  ⚡ {len(events)} events (goals, cards)")
    print(f"  📊 {len(stats)} player stats")
    print(f"  ⚖️ {len(weights)} weight records")
    print(f"  🍽️ {len(diets)} diet records")
    print(f"  💰 {len(transfers)} transfers")
    
    # Profile statistics
    print("\n📊 Career Profile Distribution:")
    profile_counts = {}
    for p in players:
        prof = p.get("_career_profile", "STABLE")
        profile_counts[prof] = profile_counts.get(prof, 0) + 1
    for prof, count in sorted(profile_counts.items()):
        print(f"  {prof}: {count} players ({100*count/len(players):.1f}%)")
    
    # Jersey changes statistics
    print("\n👕 Jersey Changes per Player:")
    player_jersey_counts = {}
    for j in jerseys:
        pid = j["player_id"]
        player_jersey_counts[pid] = player_jersey_counts.get(pid, 0) + 1
    
    change_distribution = {}
    for pid, count in player_jersey_counts.items():
        change_distribution[count] = change_distribution.get(count, 0) + 1
    for changes, num_players in sorted(change_distribution.items()):
        print(f"  {changes} jersey(s): {num_players} players")
    
    # Contract changes (transfers)
    print(f"\n💰 Players with transfers: {len(set(t['player_id'] for t in transfers))} ({100*len(set(t['player_id'] for t in transfers))/len(players):.1f}%)")
    
    # Validate data consistency
    print("\n🔍 Validating data consistency...")
    issues = validate_data_consistency(jerseys, events, matches, lineups)
    if issues:
        print(f"  ⚠️ Found {len(issues)} issues:")
        for issue in issues[:20]:  # Show first 20
            print(f"    - {issue}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")
        print("\n  ❌ Please fix issues before inserting!")
        return
    else:
        print("  ✅ All data is consistent!")
    
    print("\nInserting into Supabase (football_* tables only)...")
    async with httpx.AsyncClient(timeout=120) as client:
        # First clear all existing data in correct order
        await clear_all_football_tables(client)
        
        # Then insert in dependency order
        await insert_data(client, "football_teams", teams)
        await insert_data(client, "football_seasons", seasons)
        await insert_data(client, "football_players", players)
        await insert_data(client, "football_player_contracts", contracts)
        await insert_data(client, "football_jersey_assignments", jerseys)
        await insert_data(client, "football_matches", matches)
        await insert_data(client, "football_match_lineups", lineups)
        await insert_data(client, "football_match_events", events)
        await insert_data(client, "football_player_match_stats", stats)
        await insert_data(client, "football_player_weights", weights)
        await insert_data(client, "football_player_diets", diets)
        await insert_data(client, "football_transfers", transfers)
    
    print("\n✅ Football database populated!")
    print("\n📌 KEY INSIGHT: Match events store jersey_number + team_id")
    print("   To find who scored a goal, you must:")
    print("   1. Get match date from football_matches")
    print("   2. Look up jersey_number in football_jersey_assignments for that date")
    print("   3. Get player name from football_players")


def main():
    parser = argparse.ArgumentParser(description="Setup football database")
    parser.add_argument("--populate", action="store_true", help="Populate tables with data")
    args = parser.parse_args()
    
    if args.populate:
        asyncio.run(populate())
    else:
        print("Football Database Setup")
        print("\n1. Run SQL from scripts/football_schema.sql in Supabase SQL Editor")
        print("2. Then run: python scripts/setup_football_db.py --populate")


if __name__ == "__main__":
    main()

