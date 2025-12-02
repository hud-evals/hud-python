-- ============================================================
-- FOOTBALL DATABASE SCHEMA
-- Creates NEW tables with football_ prefix
-- Does NOT touch any existing tables!
-- ============================================================

-- FOOTBALL TEAMS
CREATE TABLE IF NOT EXISTS football_teams (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    stadium TEXT NOT NULL,
    stadium_capacity INTEGER,
    founded_year INTEGER,
    primary_color TEXT,
    secondary_color TEXT
);
ALTER TABLE football_teams DISABLE ROW LEVEL SECURITY;

-- FOOTBALL SEASONS
CREATE TABLE IF NOT EXISTS football_seasons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    is_current BOOLEAN DEFAULT FALSE
);
ALTER TABLE football_seasons DISABLE ROW LEVEL SECURITY;

-- FOOTBALL PLAYERS (base info only - no current team, that's in contracts)
CREATE TABLE IF NOT EXISTS football_players (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    nationality TEXT NOT NULL,
    birth_date DATE NOT NULL,
    height_cm INTEGER,
    preferred_foot TEXT CHECK (preferred_foot IN ('left', 'right', 'both')),
    primary_position TEXT NOT NULL,
    secondary_position TEXT
);
ALTER TABLE football_players DISABLE ROW LEVEL SECURITY;

-- FOOTBALL PLAYER CONTRACTS (tracks which team a player is on)
CREATE TABLE IF NOT EXISTS football_player_contracts (
    id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES football_players(id),
    team_id TEXT NOT NULL REFERENCES football_teams(id),
    start_date DATE NOT NULL,
    end_date DATE,  -- NULL means current contract
    salary_weekly INTEGER,
    contract_type TEXT CHECK (contract_type IN ('permanent', 'loan', 'youth'))
);
ALTER TABLE football_player_contracts DISABLE ROW LEVEL SECURITY;

-- FOOTBALL JERSEY ASSIGNMENTS (KEY TABLE - tracks jersey numbers over time)
-- A player can change jersey numbers, and different players can have same number at different times
CREATE TABLE IF NOT EXISTS football_jersey_assignments (
    id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES football_players(id),
    team_id TEXT NOT NULL REFERENCES football_teams(id),
    jersey_number INTEGER NOT NULL CHECK (jersey_number >= 1 AND jersey_number <= 99),
    start_date DATE NOT NULL,
    end_date DATE,  -- NULL means current assignment
    UNIQUE(team_id, jersey_number, start_date)  -- No two players can have same number on same team on same date
);
ALTER TABLE football_jersey_assignments DISABLE ROW LEVEL SECURITY;

-- FOOTBALL MATCHES
CREATE TABLE IF NOT EXISTS football_matches (
    id TEXT PRIMARY KEY,
    season_id TEXT NOT NULL REFERENCES football_seasons(id),
    home_team_id TEXT NOT NULL REFERENCES football_teams(id),
    away_team_id TEXT NOT NULL REFERENCES football_teams(id),
    match_date DATE NOT NULL,
    kickoff_time TIME NOT NULL,
    venue TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    attendance INTEGER,
    status TEXT CHECK (status IN ('scheduled', 'in_progress', 'completed', 'postponed', 'cancelled'))
);
ALTER TABLE football_matches DISABLE ROW LEVEL SECURITY;

-- FOOTBALL MATCH LINEUPS (who played - links player to match)
CREATE TABLE IF NOT EXISTS football_match_lineups (
    id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES football_matches(id),
    team_id TEXT NOT NULL REFERENCES football_teams(id),
    player_id TEXT NOT NULL REFERENCES football_players(id),
    jersey_number INTEGER NOT NULL,  -- The number they wore in THIS match
    position_played TEXT NOT NULL,
    is_starter BOOLEAN DEFAULT TRUE,
    subbed_in_minute INTEGER,  -- NULL if started
    subbed_out_minute INTEGER,  -- NULL if played full match or never entered
    rating DECIMAL(3,1)  -- Match rating 1.0-10.0
);
ALTER TABLE football_match_lineups DISABLE ROW LEVEL SECURITY;

-- FOOTBALL MATCH EVENTS (goals, cards, etc. - STORED BY JERSEY NUMBER!)
-- This is the key table: events reference jersey_number + team, NOT player directly
CREATE TABLE IF NOT EXISTS football_match_events (
    id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES football_matches(id),
    team_id TEXT NOT NULL REFERENCES football_teams(id),
    jersey_number INTEGER NOT NULL,  -- WHO did it (by jersey number, need to look up player)
    event_type TEXT NOT NULL CHECK (event_type IN ('goal', 'own_goal', 'penalty_scored', 'penalty_missed', 'yellow_card', 'red_card', 'second_yellow', 'assist', 'substitution_in', 'substitution_out')),
    minute INTEGER NOT NULL CHECK (minute >= 0 AND minute <= 120),
    added_time INTEGER DEFAULT 0,  -- Injury time minutes
    assist_jersey_number INTEGER,  -- For goals, who assisted (also by jersey number)
    description TEXT
);
ALTER TABLE football_match_events DISABLE ROW LEVEL SECURITY;

-- FOOTBALL PLAYER MATCH STATS (detailed per-match statistics)
CREATE TABLE IF NOT EXISTS football_player_match_stats (
    id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES football_matches(id),
    player_id TEXT NOT NULL REFERENCES football_players(id),
    minutes_played INTEGER NOT NULL DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    passes_attempted INTEGER DEFAULT 0,
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    fouls_committed INTEGER DEFAULT 0,
    fouls_suffered INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,  -- For goalkeepers
    distance_covered_km DECIMAL(4,2)
);
ALTER TABLE football_player_match_stats DISABLE ROW LEVEL SECURITY;

-- FOOTBALL PLAYER DAILY WEIGHTS
CREATE TABLE IF NOT EXISTS football_player_weights (
    id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES football_players(id),
    recorded_date DATE NOT NULL,
    weight_kg DECIMAL(5,2) NOT NULL,
    body_fat_percent DECIMAL(4,1),
    notes TEXT
);
ALTER TABLE football_player_weights DISABLE ROW LEVEL SECURITY;

-- FOOTBALL PLAYER DAILY DIETS
CREATE TABLE IF NOT EXISTS football_player_diets (
    id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES football_players(id),
    diet_date DATE NOT NULL,
    total_calories INTEGER NOT NULL,
    protein_g INTEGER,
    carbs_g INTEGER,
    fat_g INTEGER,
    water_liters DECIMAL(3,1),
    meal_plan TEXT  -- e.g., 'match_day', 'recovery', 'training', 'rest'
);
ALTER TABLE football_player_diets DISABLE ROW LEVEL SECURITY;

-- FOOTBALL TRANSFERS (historical record of player movements)
CREATE TABLE IF NOT EXISTS football_transfers (
    id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL REFERENCES football_players(id),
    from_team_id TEXT REFERENCES football_teams(id),  -- NULL if first professional contract
    to_team_id TEXT NOT NULL REFERENCES football_teams(id),
    transfer_date DATE NOT NULL,
    transfer_fee INTEGER,  -- In thousands (e.g., 50000 = 50 million)
    transfer_type TEXT CHECK (transfer_type IN ('permanent', 'loan', 'free', 'youth_promotion', 'loan_return'))
);
ALTER TABLE football_transfers DISABLE ROW LEVEL SECURITY;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_fb_contracts_player ON football_player_contracts(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_contracts_team ON football_player_contracts(team_id);
CREATE INDEX IF NOT EXISTS idx_fb_contracts_dates ON football_player_contracts(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_fb_jersey_player ON football_jersey_assignments(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_jersey_team ON football_jersey_assignments(team_id);
CREATE INDEX IF NOT EXISTS idx_fb_jersey_dates ON football_jersey_assignments(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_fb_jersey_number ON football_jersey_assignments(team_id, jersey_number);
CREATE INDEX IF NOT EXISTS idx_fb_matches_season ON football_matches(season_id);
CREATE INDEX IF NOT EXISTS idx_fb_matches_date ON football_matches(match_date);
CREATE INDEX IF NOT EXISTS idx_fb_matches_teams ON football_matches(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_fb_lineups_match ON football_match_lineups(match_id);
CREATE INDEX IF NOT EXISTS idx_fb_lineups_player ON football_match_lineups(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_events_match ON football_match_events(match_id);
CREATE INDEX IF NOT EXISTS idx_fb_events_type ON football_match_events(event_type);
CREATE INDEX IF NOT EXISTS idx_fb_stats_match ON football_player_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_fb_stats_player ON football_player_match_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_weights_player ON football_player_weights(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_weights_date ON football_player_weights(recorded_date);
CREATE INDEX IF NOT EXISTS idx_fb_diets_player ON football_player_diets(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_diets_date ON football_player_diets(diet_date);
CREATE INDEX IF NOT EXISTS idx_fb_transfers_player ON football_transfers(player_id);
CREATE INDEX IF NOT EXISTS idx_fb_transfers_date ON football_transfers(transfer_date);

