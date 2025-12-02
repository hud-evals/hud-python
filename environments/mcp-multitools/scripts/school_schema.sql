-- School Database Schema
-- Design principle: Maximum normalization with junction tables everywhere
-- Simple questions require 5+ joins to answer

-- ============================================
-- CORE ENTITY TABLES (no foreign keys here!)
-- ============================================

CREATE TABLE school_cities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    population INTEGER,
    country TEXT,
    timezone TEXT
);
ALTER TABLE school_cities DISABLE ROW LEVEL SECURITY;

CREATE TABLE school_schools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    square_meters INTEGER,
    floors INTEGER,
    rating DECIMAL(3,2),  -- 1.00 to 5.00
    founded_year INTEGER
);
ALTER TABLE school_schools DISABLE ROW LEVEL SECURITY;

CREATE TABLE school_clubs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    club_type TEXT,  -- 'sports', 'academic', 'arts', 'social'
    max_members INTEGER,
    meeting_day TEXT,  -- 'Monday', 'Tuesday', etc.
    parent_club_id TEXT REFERENCES school_clubs(id)  -- Self-reference for sub-clubs
);
ALTER TABLE school_clubs DISABLE ROW LEVEL SECURITY;

CREATE TABLE school_students (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    birth_date DATE,
    gender TEXT,
    gpa DECIMAL(3,2)
);
ALTER TABLE school_students DISABLE ROW LEVEL SECURITY;

CREATE TABLE school_teachers (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    subject TEXT,
    hire_date DATE,
    salary INTEGER
);
ALTER TABLE school_teachers DISABLE ROW LEVEL SECURITY;

CREATE TABLE school_events (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    event_type TEXT,  -- 'competition', 'performance', 'meeting', 'fundraiser'
    budget DECIMAL(10,2),
    is_recurring BOOLEAN DEFAULT FALSE
);
ALTER TABLE school_events DISABLE ROW LEVEL SECURITY;

CREATE TABLE school_rooms (
    id TEXT PRIMARY KEY,
    room_number TEXT NOT NULL,
    capacity INTEGER,
    room_type TEXT,  -- 'classroom', 'gym', 'lab', 'auditorium', 'office'
    has_projector BOOLEAN DEFAULT FALSE
);
ALTER TABLE school_rooms DISABLE ROW LEVEL SECURITY;

-- ============================================
-- JUNCTION/BRIDGE TABLES (all relationships here!)
-- ============================================

-- Which school is in which city (with temporal validity)
CREATE TABLE school_city_school_assignments (
    id TEXT PRIMARY KEY,
    city_id TEXT NOT NULL REFERENCES school_cities(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    established_date DATE NOT NULL,
    closed_date DATE,  -- NULL means still active
    is_primary_location BOOLEAN DEFAULT TRUE
);
ALTER TABLE school_city_school_assignments DISABLE ROW LEVEL SECURITY;

-- Which club belongs to which school (with temporal validity)
CREATE TABLE school_club_school_assignments (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL REFERENCES school_clubs(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    start_date DATE NOT NULL,
    end_date DATE,  -- NULL means still active
    assigned_room_id TEXT REFERENCES school_rooms(id),
    budget_allocation DECIMAL(10,2)
);
ALTER TABLE school_club_school_assignments DISABLE ROW LEVEL SECURITY;

-- Which room belongs to which school
CREATE TABLE school_room_school_assignments (
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL REFERENCES school_rooms(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    floor_number INTEGER,
    wing TEXT  -- 'North', 'South', 'East', 'West', 'Main'
);
ALTER TABLE school_room_school_assignments DISABLE ROW LEVEL SECURITY;

-- Which student is enrolled in which school (with dates)
CREATE TABLE school_student_enrollments (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES school_students(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    enrollment_date DATE NOT NULL,
    graduation_date DATE,  -- NULL means still enrolled
    grade_level INTEGER,  -- 1-12
    is_transfer BOOLEAN DEFAULT FALSE
);
ALTER TABLE school_student_enrollments DISABLE ROW LEVEL SECURITY;

-- Which student is member of which club (with role and dates)
CREATE TABLE school_club_memberships (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES school_students(id),
    club_id TEXT NOT NULL REFERENCES school_clubs(id),
    role TEXT,  -- 'member', 'president', 'vice_president', 'secretary', 'treasurer'
    join_date DATE NOT NULL,
    leave_date DATE,  -- NULL means still active
    is_active BOOLEAN DEFAULT TRUE
);
ALTER TABLE school_club_memberships DISABLE ROW LEVEL SECURITY;

-- Which teacher works at which school
CREATE TABLE school_teacher_school_assignments (
    id TEXT PRIMARY KEY,
    teacher_id TEXT NOT NULL REFERENCES school_teachers(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    start_date DATE NOT NULL,
    end_date DATE,
    department TEXT,
    is_department_head BOOLEAN DEFAULT FALSE
);
ALTER TABLE school_teacher_school_assignments DISABLE ROW LEVEL SECURITY;

-- Which teacher advises which club
CREATE TABLE school_club_advisors (
    id TEXT PRIMARY KEY,
    teacher_id TEXT NOT NULL REFERENCES school_teachers(id),
    club_id TEXT NOT NULL REFERENCES school_clubs(id),
    start_date DATE NOT NULL,
    end_date DATE,
    is_primary_advisor BOOLEAN DEFAULT TRUE
);
ALTER TABLE school_club_advisors DISABLE ROW LEVEL SECURITY;

-- Which event happens at which school (in which room)
CREATE TABLE school_event_locations (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES school_events(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    room_id TEXT REFERENCES school_rooms(id),
    event_date DATE NOT NULL,
    start_time TIME,
    end_time TIME
);
ALTER TABLE school_event_locations DISABLE ROW LEVEL SECURITY;

-- Which student participates in which event
CREATE TABLE school_event_participants (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES school_events(id),
    student_id TEXT NOT NULL REFERENCES school_students(id),
    role TEXT,  -- 'participant', 'organizer', 'volunteer', 'performer', 'competitor'
    registration_date DATE,
    attended BOOLEAN
);
ALTER TABLE school_event_participants DISABLE ROW LEVEL SECURITY;

-- Which club organizes which event
CREATE TABLE school_event_organizers (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES school_events(id),
    club_id TEXT NOT NULL REFERENCES school_clubs(id),
    contribution_percent DECIMAL(5,2)  -- How much of the event this club organized
);
ALTER TABLE school_event_organizers DISABLE ROW LEVEL SECURITY;

-- ============================================
-- ATTRIBUTE/METRIC TABLES (for even more complexity)
-- ============================================

-- Student test scores (separate from student table)
CREATE TABLE school_student_scores (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES school_students(id),
    subject TEXT NOT NULL,
    score DECIMAL(5,2),
    max_score DECIMAL(5,2),
    test_date DATE NOT NULL,
    semester TEXT  -- 'Fall 2024', 'Spring 2024'
);
ALTER TABLE school_student_scores DISABLE ROW LEVEL SECURITY;

-- Student attendance (separate tracking)
CREATE TABLE school_student_attendance (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES school_students(id),
    school_id TEXT NOT NULL REFERENCES school_schools(id),
    attendance_date DATE NOT NULL,
    status TEXT,  -- 'present', 'absent', 'late', 'excused'
    notes TEXT
);
ALTER TABLE school_student_attendance DISABLE ROW LEVEL SECURITY;

-- Club meeting records
CREATE TABLE school_club_meetings (
    id TEXT PRIMARY KEY,
    club_id TEXT NOT NULL REFERENCES school_clubs(id),
    meeting_date DATE NOT NULL,
    room_id TEXT REFERENCES school_rooms(id),
    duration_minutes INTEGER,
    attendee_count INTEGER,
    notes TEXT
);
ALTER TABLE school_club_meetings DISABLE ROW LEVEL SECURITY;

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX idx_school_city_assignments_city ON school_city_school_assignments(city_id);
CREATE INDEX idx_school_city_assignments_school ON school_city_school_assignments(school_id);
CREATE INDEX idx_school_club_assignments_club ON school_club_school_assignments(club_id);
CREATE INDEX idx_school_club_assignments_school ON school_club_school_assignments(school_id);
CREATE INDEX idx_school_room_assignments_room ON school_room_school_assignments(room_id);
CREATE INDEX idx_school_room_assignments_school ON school_room_school_assignments(school_id);
CREATE INDEX idx_school_enrollments_student ON school_student_enrollments(student_id);
CREATE INDEX idx_school_enrollments_school ON school_student_enrollments(school_id);
CREATE INDEX idx_school_memberships_student ON school_club_memberships(student_id);
CREATE INDEX idx_school_memberships_club ON school_club_memberships(club_id);
CREATE INDEX idx_school_teacher_assignments_teacher ON school_teacher_school_assignments(teacher_id);
CREATE INDEX idx_school_teacher_assignments_school ON school_teacher_school_assignments(school_id);
CREATE INDEX idx_school_advisors_teacher ON school_club_advisors(teacher_id);
CREATE INDEX idx_school_advisors_club ON school_club_advisors(club_id);
CREATE INDEX idx_school_event_locations_event ON school_event_locations(event_id);
CREATE INDEX idx_school_event_locations_school ON school_event_locations(school_id);
CREATE INDEX idx_school_event_participants_event ON school_event_participants(event_id);
CREATE INDEX idx_school_event_participants_student ON school_event_participants(student_id);
CREATE INDEX idx_school_scores_student ON school_student_scores(student_id);
CREATE INDEX idx_school_attendance_student ON school_student_attendance(student_id);
CREATE INDEX idx_school_meetings_club ON school_club_meetings(club_id);

