#!/usr/bin/env python3
"""Calculate answers for hard school tasks by querying Supabase."""

import os
from dotenv import load_dotenv
from supabase import create_client
from collections import defaultdict, Counter

# Load environment variables
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_API_KEY")

print(f"URL: {SUPABASE_URL}")
print(f"Key exists: {bool(SUPABASE_KEY)}")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_teacher_name(t):
    return f"{t.get('first_name', '')} {t.get('last_name', '')}".strip()


def get_student_name(s):
    return f"{s.get('first_name', '')} {s.get('last_name', '')}".strip()


def explore_tables():
    """List all school_ tables and their columns."""
    print("\n" + "=" * 60)
    print("EXPLORING SCHOOL TABLES")
    print("=" * 60)

    known_tables = [
        "school_cities",
        "school_city_school_assignments",
        "school_schools",
        "school_students",
        "school_student_enrollments",
        "school_teachers",
        "school_teacher_school_assignments",
        "school_clubs",
        "school_club_school_assignments",
        "school_club_memberships",
        "school_club_advisors",
        "school_events",
        "school_event_participants",
        "school_student_attendance",
        "school_student_scores",
    ]

    for table in known_tables:
        try:
            result = supabase.table(table).select("*").limit(1).execute()
            if result.data:
                print(f"\n{table}:")
                print(f"  Columns: {list(result.data[0].keys())}")
            else:
                print(f"\n{table}: (empty)")
        except Exception as e:
            print(f"\n{table}: ERROR - {e}")


def task1_teachers_multi_school_high_salary():
    """
    Find teachers who teach at multiple schools and have salary > $75,000.
    Count how many such teachers exist.
    """
    print("\n" + "=" * 60)
    print("TASK 1: Teachers at 2+ schools with salary > $75k")
    print("=" * 60)

    assignments = supabase.table("school_teacher_school_assignments").select("*").execute()
    teachers = supabase.table("school_teachers").select("*").execute()

    teacher_salaries = {t["id"]: t["salary"] for t in teachers.data}
    teacher_names = {t["id"]: get_teacher_name(t) for t in teachers.data}

    # Count schools per teacher
    teacher_schools = defaultdict(set)
    for a in assignments.data:
        teacher_schools[a["teacher_id"]].add(a["school_id"])

    # Find teachers at multiple schools with high salary
    multi_school_high_salary = []
    for tid, schools in teacher_schools.items():
        if len(schools) >= 2:
            salary = teacher_salaries.get(tid, 0)
            if salary > 75000:
                multi_school_high_salary.append((teacher_names.get(tid, "Unknown"), len(schools), salary))

    print(f"Teachers at 2+ schools with salary > $75k: {len(multi_school_high_salary)}")
    for name, num_schools, salary in sorted(multi_school_high_salary, key=lambda x: -x[2])[:10]:
        print(f"  {name}: {num_schools} schools, ${salary:,.0f}")

    return len(multi_school_high_salary)


def task2_students_multiple_clubs_same_type():
    """
    Find students who are active members of 2+ clubs of the SAME type.
    """
    print("\n" + "=" * 60)
    print("TASK 2: Students in 2+ clubs of same type")
    print("=" * 60)

    memberships = supabase.table("school_club_memberships").select("*").execute()
    clubs = supabase.table("school_clubs").select("*").execute()
    students = supabase.table("school_students").select("*").execute()

    club_types = {c["id"]: c["club_type"] for c in clubs.data}
    student_names = {s["id"]: get_student_name(s) for s in students.data}

    # Group memberships by student and club type
    student_clubs_by_type = defaultdict(lambda: defaultdict(list))

    for m in memberships.data:
        if m.get("is_active"):  # Using is_active instead of status
            sid = m["student_id"]
            cid = m["club_id"]
            ctype = club_types.get(cid, "unknown")
            student_clubs_by_type[sid][ctype].append(cid)

    # Find students with 2+ clubs of same type
    result = []
    for sid, types in student_clubs_by_type.items():
        for ctype, club_list in types.items():
            if len(club_list) >= 2:
                result.append((student_names.get(sid, "Unknown"), ctype, len(club_list)))

    print(f"Students in 2+ clubs of same type: {len(result)}")
    for name, ctype, count in result[:10]:
        print(f"  {name}: {count} {ctype} clubs")

    return len(result)


def task3_schools_above_city_avg_salary():
    """
    Find schools where average teacher salary is above their city's average.
    """
    print("\n" + "=" * 60)
    print("TASK 3: Schools with avg salary > city avg")
    print("=" * 60)

    teachers = supabase.table("school_teachers").select("*").execute()
    teacher_assignments = supabase.table("school_teacher_school_assignments").select("*").execute()
    schools = supabase.table("school_schools").select("*").execute()
    city_assignments = supabase.table("school_city_school_assignments").select("*").execute()
    cities = supabase.table("school_cities").select("*").execute()

    teacher_salaries = {t["id"]: t["salary"] for t in teachers.data}
    school_names = {s["id"]: s["name"] for s in schools.data}
    city_names = {c["id"]: c["name"] for c in cities.data}

    # School -> City mapping
    school_to_city = {a["school_id"]: a["city_id"] for a in city_assignments.data}

    # Calculate avg salary per school
    school_salaries = defaultdict(list)
    for a in teacher_assignments.data:
        salary = teacher_salaries.get(a["teacher_id"], 0)
        if salary > 0:
            school_salaries[a["school_id"]].append(salary)

    school_avg = {sid: sum(sals) / len(sals) for sid, sals in school_salaries.items() if sals}

    # Calculate avg salary per city
    city_salaries = defaultdict(list)
    for sid, avg in school_avg.items():
        cid = school_to_city.get(sid)
        if cid:
            city_salaries[cid].append(avg)

    city_avg = {cid: sum(avgs) / len(avgs) for cid, avgs in city_salaries.items() if avgs}

    # Find schools above city average
    above_avg = []
    for sid, s_avg in school_avg.items():
        cid = school_to_city.get(sid)
        c_avg = city_avg.get(cid, 0)
        if s_avg > c_avg and c_avg > 0:
            above_avg.append((school_names.get(sid, "Unknown"), s_avg, city_names.get(cid, "?"), c_avg))

    print(f"Schools above city average: {len(above_avg)}")
    for name, s_avg, city, c_avg in sorted(above_avg, key=lambda x: x[1] - x[3], reverse=True)[:10]:
        print(f"  {name} ({city}): ${s_avg:,.0f} vs city ${c_avg:,.0f}")

    return len(above_avg)


def task4_high_absence_students():
    """
    Find students with 5+ absences. Count them.
    """
    print("\n" + "=" * 60)
    print("TASK 4: Students with 5+ absences")
    print("=" * 60)

    attendance = supabase.table("school_student_attendance").select("*").execute()
    students = supabase.table("school_students").select("*").execute()

    student_names = {s["id"]: get_student_name(s) for s in students.data}

    absence_count = Counter()
    for a in attendance.data:
        if a.get("status") == "absent":
            absence_count[a["student_id"]] += 1

    high_absence = [(sid, count) for sid, count in absence_count.items() if count >= 5]
    print(f"Students with 5+ absences: {len(high_absence)}")

    for sid, count in sorted(high_absence, key=lambda x: -x[1])[:10]:
        print(f"  {student_names.get(sid, 'Unknown')}: {count} absences")

    return len(high_absence)


def task5_clubs_no_active_members():
    """
    Find clubs with zero active members.
    """
    print("\n" + "=" * 60)
    print("TASK 5: Clubs with no active members")
    print("=" * 60)

    clubs = supabase.table("school_clubs").select("*").execute()
    memberships = supabase.table("school_club_memberships").select("*").execute()

    all_club_ids = {c["id"] for c in clubs.data}
    clubs_with_active = {m["club_id"] for m in memberships.data if m.get("is_active")}

    clubs_no_members = all_club_ids - clubs_with_active
    club_names = {c["id"]: c["name"] for c in clubs.data}

    print(f"Clubs with no active members: {len(clubs_no_members)}")
    for cid in list(clubs_no_members)[:10]:
        print(f"  {club_names.get(cid, 'Unknown')}")

    return len(clubs_no_members)


def task6_city_highest_avg_test_score():
    """
    Find the city with highest average test score.
    """
    print("\n" + "=" * 60)
    print("TASK 6: City with highest average test score")
    print("=" * 60)

    cities = supabase.table("school_cities").select("*").execute()
    city_assignments = supabase.table("school_city_school_assignments").select("*").execute()

    # Check if we have enrollments table
    try:
        enrollments = supabase.table("school_student_enrollments").select("*").execute()
        student_to_school = {e["student_id"]: e["school_id"] for e in enrollments.data}
    except:
        students = supabase.table("school_students").select("*").execute()
        student_to_school = {s["id"]: s.get("school_id") for s in students.data}

    test_scores = supabase.table("school_student_scores").select("*").execute()

    city_names = {c["id"]: c["name"] for c in cities.data}
    school_to_city = {a["school_id"]: a["city_id"] for a in city_assignments.data}

    # Aggregate scores by city
    city_scores = defaultdict(list)
    for ts in test_scores.data:
        sid = ts["student_id"]
        school_id = student_to_school.get(sid)
        city_id = school_to_city.get(school_id)
        if city_id and ts.get("score") is not None:
            city_scores[city_id].append(ts["score"])

    city_avgs = {cid: sum(scores) / len(scores) for cid, scores in city_scores.items() if scores}

    print("City averages:")
    for cid, avg in sorted(city_avgs.items(), key=lambda x: -x[1]):
        print(f"  {city_names.get(cid, 'Unknown')}: {avg:.2f}")

    if city_avgs:
        best_city_id = max(city_avgs, key=city_avgs.get)
        best_avg = city_avgs[best_city_id]
        print(f"\nHighest: {city_names.get(best_city_id)} with avg {best_avg:.2f}")
        return city_names.get(best_city_id), round(best_avg, 2)
    return None, None


def task7_students_shared_clubs():
    """
    Find pairs of students who share at least 2 clubs.
    """
    print("\n" + "=" * 60)
    print("TASK 7: Student pairs sharing 2+ clubs")
    print("=" * 60)

    memberships = supabase.table("school_club_memberships").select("*").execute()
    students = supabase.table("school_students").select("*").execute()

    student_names = {s["id"]: get_student_name(s) for s in students.data}

    # Build student -> clubs mapping (active only)
    student_clubs = defaultdict(set)
    for m in memberships.data:
        if m.get("is_active"):
            student_clubs[m["student_id"]].add(m["club_id"])

    # Find pairs with 2+ shared clubs
    student_ids = list(student_clubs.keys())
    pairs = []

    for i in range(len(student_ids)):
        for j in range(i + 1, len(student_ids)):
            s1, s2 = student_ids[i], student_ids[j]
            shared = student_clubs[s1] & student_clubs[s2]
            if len(shared) >= 2:
                pairs.append((s1, s2, len(shared)))

    print(f"Student pairs sharing 2+ clubs: {len(pairs)}")
    for s1, s2, count in pairs[:10]:
        print(f"  {student_names.get(s1, '?')} & {student_names.get(s2, '?')}: {count} clubs")

    return len(pairs)


def task8_teacher_highest_avg_student_score():
    """
    Find the teacher whose students have the highest average test score.
    """
    print("\n" + "=" * 60)
    print("TASK 8: Teacher with highest avg student score")
    print("=" * 60)

    teachers = supabase.table("school_teachers").select("*").execute()
    teacher_assignments = supabase.table("school_teacher_school_assignments").select("*").execute()

    try:
        enrollments = supabase.table("school_student_enrollments").select("*").execute()
        student_to_school = {e["student_id"]: e["school_id"] for e in enrollments.data}
    except:
        students = supabase.table("school_students").select("*").execute()
        student_to_school = {s["id"]: s.get("school_id") for s in students.data}

    test_scores = supabase.table("school_student_scores").select("*").execute()

    teacher_names = {t["id"]: get_teacher_name(t) for t in teachers.data}

    # Teacher -> Schools
    teacher_schools = defaultdict(set)
    for a in teacher_assignments.data:
        teacher_schools[a["teacher_id"]].add(a["school_id"])

    # Student scores
    student_avg_scores = defaultdict(list)
    for ts in test_scores.data:
        if ts.get("score") is not None:
            student_avg_scores[ts["student_id"]].append(ts["score"])

    student_avg = {sid: sum(s) / len(s) for sid, s in student_avg_scores.items() if s}

    # Aggregate by teacher
    teacher_student_scores = defaultdict(list)
    for sid, avg in student_avg.items():
        school_id = student_to_school.get(sid)
        for tid, schools in teacher_schools.items():
            if school_id in schools:
                teacher_student_scores[tid].append(avg)

    teacher_avgs = {tid: sum(s) / len(s) for tid, s in teacher_student_scores.items() if s}

    print("Top 10 teachers by student avg score:")
    for tid, avg in sorted(teacher_avgs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {teacher_names.get(tid, 'Unknown')}: {avg:.2f}")

    if teacher_avgs:
        best_tid = max(teacher_avgs, key=teacher_avgs.get)
        return teacher_names.get(best_tid), round(teacher_avgs[best_tid], 2)
    return None, None


def task9_gpa_club_correlation():
    """
    Compare average GPA of students in academic clubs vs sports clubs.
    """
    print("\n" + "=" * 60)
    print("TASK 9: Avg GPA - academic vs sports club members")
    print("=" * 60)

    students = supabase.table("school_students").select("*").execute()
    clubs = supabase.table("school_clubs").select("*").execute()
    memberships = supabase.table("school_club_memberships").select("*").execute()

    student_gpas = {s["id"]: s.get("gpa", 0) for s in students.data}
    club_types = {c["id"]: c.get("club_type", "") for c in clubs.data}

    academic_students = set()
    sports_students = set()

    for m in memberships.data:
        if not m.get("is_active"):
            continue
        ctype = club_types.get(m["club_id"], "")
        sid = m["student_id"]
        if ctype == "academic":
            academic_students.add(sid)
        elif ctype == "sports":
            sports_students.add(sid)

    academic_gpas = [student_gpas[sid] for sid in academic_students if sid in student_gpas and student_gpas[sid]]
    sports_gpas = [student_gpas[sid] for sid in sports_students if sid in student_gpas and student_gpas[sid]]

    avg_academic = sum(academic_gpas) / len(academic_gpas) if academic_gpas else 0
    avg_sports = sum(sports_gpas) / len(sports_gpas) if sports_gpas else 0

    print(f"Academic club members avg GPA: {avg_academic:.2f} ({len(academic_gpas)} unique students)")
    print(f"Sports club members avg GPA: {avg_sports:.2f} ({len(sports_gpas)} unique students)")

    return round(avg_academic, 2), round(avg_sports, 2)


def task10_event_participation_by_gpa():
    """
    Find average GPA of students who participated in events vs those who didn't.
    """
    print("\n" + "=" * 60)
    print("TASK 10: Avg GPA - event participants vs non-participants")
    print("=" * 60)

    students = supabase.table("school_students").select("*").execute()
    participants = supabase.table("school_event_participants").select("*").execute()

    student_gpas = {s["id"]: s.get("gpa", 0) for s in students.data}
    all_students = set(student_gpas.keys())
    participating_students = {p["student_id"] for p in participants.data}
    non_participating = all_students - participating_students

    part_gpas = [student_gpas[sid] for sid in participating_students if sid in student_gpas and student_gpas[sid]]
    non_part_gpas = [student_gpas[sid] for sid in non_participating if sid in student_gpas and student_gpas[sid]]

    avg_part = sum(part_gpas) / len(part_gpas) if part_gpas else 0
    avg_non = sum(non_part_gpas) / len(non_part_gpas) if non_part_gpas else 0

    print(f"Event participants avg GPA: {avg_part:.2f} ({len(part_gpas)} students)")
    print(f"Non-participants avg GPA: {avg_non:.2f} ({len(non_part_gpas)} students)")

    return round(avg_part, 2), round(avg_non, 2)


def task11_total_salary_teachers_with_advisees():
    """
    Sum of salaries for teachers who are club advisors.
    """
    print("\n" + "=" * 60)
    print("TASK 11: Total salary of teachers who are club advisors")
    print("=" * 60)

    teachers = supabase.table("school_teachers").select("*").execute()
    advisors = supabase.table("school_club_advisors").select("*").execute()

    teacher_salaries = {t["id"]: t["salary"] for t in teachers.data}
    advisor_teachers = {a["teacher_id"] for a in advisors.data}

    total = sum(teacher_salaries.get(tid, 0) for tid in advisor_teachers)
    print(f"Total salary of advisor teachers: ${total:,.0f}")
    print(f"Number of advisor teachers: {len(advisor_teachers)}")

    return total


def task12_students_in_all_club_types():
    """
    Find students who are active in at least one club of EACH type.
    """
    print("\n" + "=" * 60)
    print("TASK 12: Students in all club types")
    print("=" * 60)

    students = supabase.table("school_students").select("*").execute()
    clubs = supabase.table("school_clubs").select("*").execute()
    memberships = supabase.table("school_club_memberships").select("*").execute()

    student_names = {s["id"]: get_student_name(s) for s in students.data}
    club_types = {c["id"]: c.get("club_type", "") for c in clubs.data}
    all_types = set(club_types.values())

    print(f"All club types: {all_types}")

    # Track which types each student has
    student_types = defaultdict(set)
    for m in memberships.data:
        if m.get("is_active"):
            ctype = club_types.get(m["club_id"], "")
            if ctype:
                student_types[m["student_id"]].add(ctype)

    # Find students with all types
    all_type_students = [sid for sid, types in student_types.items() if types == all_types]

    print(f"Students in ALL club types: {len(all_type_students)}")
    for sid in all_type_students[:10]:
        print(f"  {student_names.get(sid, 'Unknown')}")

    return len(all_type_students)


if __name__ == "__main__":
    print("Calculating answers for hard school tasks...")

    explore_tables()

    results = {}

    results["task1_teachers_multi_school"] = task1_teachers_multi_school_high_salary()
    results["task2_students_multi_club_type"] = task2_students_multiple_clubs_same_type()
    results["task3_schools_above_city_avg"] = task3_schools_above_city_avg_salary()
    results["task4_high_absence"] = task4_high_absence_students()
    results["task5_clubs_no_members"] = task5_clubs_no_active_members()
    results["task6_city_best_scores"] = task6_city_highest_avg_test_score()
    results["task7_shared_clubs"] = task7_students_shared_clubs()
    results["task8_best_teacher"] = task8_teacher_highest_avg_student_score()
    results["task9_gpa_academic_vs_sports"] = task9_gpa_club_correlation()
    results["task10_gpa_event_participants"] = task10_event_participation_by_gpa()
    results["task11_advisor_salaries"] = task11_total_salary_teachers_with_advisees()
    results["task12_all_club_types"] = task12_students_in_all_club_types()

    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    for task, result in results.items():
        print(f"{task}: {result}")
