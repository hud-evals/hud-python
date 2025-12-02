#!/usr/bin/env python3
"""
Verify plant task answers by calculating them from actual data.
"""

import json
import re
from pathlib import Path

BASE = Path("/home/rs/projects/hud-python/environments/mcp-multitools")

# Load ground truth data
with open(BASE / "scripts/plant_data.json") as f:
    DATA = json.load(f)

PLANTS = {p["id"]: p for p in DATA["plants"]}
SPECIES = DATA["species"]
CARETAKERS = {c["id"]: c["name"] for c in DATA["caretakers"]}

# Parse growth logs
GROWTH_LOGS = {}  # {plant_id: {week: {height, width, leaves, health}}}
for log in DATA["growth_logs"]:
    pid = log["plant_id"]
    week = log["week_number"]
    if pid not in GROWTH_LOGS:
        GROWTH_LOGS[pid] = {}
    GROWTH_LOGS[pid][week] = {
        "height": log["height_cm"],
        "width": log["width_cm"],
        "leaves": log["leaf_count"],
        "health": log["health_score"]
    }

# Parse watering logs
WATERING_LOGS = DATA["watering_logs"]


def task_1_w1_five_monstera():
    """W1 formula for all Monstera at T=24°C"""
    # Find all Monstera
    monstera_plants = [p for p in DATA["plants"] if p["species"] == "Monstera Deliciosa"]
    
    # W1 formula: Water_ml = 250 × (1 + (T - 22) × 0.03)
    T = 24
    base_water = 250
    water_per_plant = base_water * (1 + (T - 22) * 0.03)
    
    total = water_per_plant * len(monstera_plants)
    
    print(f"Task 1: W1 for {len(monstera_plants)} Monstera at T={T}°C")
    print(f"  Formula: 250 × (1 + ({T}-22) × 0.03) = {water_per_plant:.1f} ml each")
    print(f"  Total: {len(monstera_plants)} × {water_per_plant:.1f} = {total:.0f}")
    return str(int(round(total)))


def task_2_w3_indirect():
    """W3 for Pothos in Zone-C assigned to Elena Kowalski"""
    # Find the plant
    plant = None
    for p in DATA["plants"]:
        if p["species"] == "Pothos Golden" and p["zone"] == "Zone-C" and p["caretaker_name"] == "Elena Kowalski":
            plant = p
            break
    
    if not plant:
        print("Task 2: ERROR - Plant not found!")
        return "ERROR"
    
    print(f"Task 2: Found {plant['id']} - {plant['species']}")
    print(f"  Zone: {plant['zone']}, Caretaker: {plant['caretaker_name']}")
    print(f"  Pot size: {plant['pot_size_cm']} cm")
    
    # W3 formula: Water_ml = 150 × (1 + (T-22) × 0.03) × (1 + (40-M)/100) × (P/20)
    T = 23
    M = 35
    P = plant["pot_size_cm"]
    base_water = 150  # Pothos base
    
    water = base_water * (1 + (T - 22) * 0.03) * (1 + (40 - M) / 100) * (P / 20)
    
    print(f"  W3: 150 × (1 + ({T}-22)×0.03) × (1 + (40-{M})/100) × ({P}/20)")
    print(f"     = 150 × {1 + (T-22)*0.03:.3f} × {1 + (40-M)/100:.3f} × {P/20:.3f}")
    print(f"     = {water:.1f} ml")
    return str(int(round(water)))


def task_3_modified_coefficient():
    """W1 with modified coefficient (0.05 instead of 0.03) for PLT-FIC-002"""
    plant = PLANTS["PLT-FIC-002"]
    
    # Modified W1: Water_ml = 300 × (1 + (T-22) × 0.05)
    T = 26
    base_water = 300  # Ficus base
    coefficient = 0.05  # Modified from 0.03
    
    water = base_water * (1 + (T - 22) * coefficient)
    
    print(f"Task 3: Modified W1 for {plant['id']} (Ficus)")
    print(f"  Modified formula: 300 × (1 + ({T}-22) × {coefficient})")
    print(f"     = 300 × {1 + (T-22)*coefficient:.2f} = {water:.0f} ml")
    return str(int(round(water)))


def task_4_n1_nutrition():
    """N1 formula for PLT-MON-013 using Week 12 height"""
    plant = PLANTS["PLT-MON-013"]
    week12 = GROWTH_LOGS["PLT-MON-013"][12]
    
    Ht = week12["height"]
    P = plant["pot_size_cm"]
    
    # N1: Fertilizer_ml = (Ht / 10) × (P / 15) × 2
    fertilizer = (Ht / 10) * (P / 15) * 2
    
    print(f"Task 4: N1 for {plant['id']}")
    print(f"  Week 12 height: {Ht} cm, Pot size: {P} cm")
    print(f"  N1: ({Ht}/10) × ({P}/15) × 2 = {fertilizer:.2f} ml")
    return f"{fertilizer:.1f}"


def task_5_growth_comparison():
    """Compare actual vs expected growth for PLT-FIC-008"""
    plant_id = "PLT-FIC-008"
    
    week1 = GROWTH_LOGS[plant_id][1]["height"]
    week12 = GROWTH_LOGS[plant_id][12]["height"]
    
    actual_weekly = (week12 - week1) / 11
    expected_weekly = SPECIES["Ficus Lyrata"]["growth_rate_cm_week"]
    
    percentage = (actual_weekly / expected_weekly) * 100
    
    print(f"Task 5: Growth comparison for {plant_id}")
    print(f"  Week 1: {week1} cm, Week 12: {week12} cm")
    print(f"  Actual growth: ({week12} - {week1}) / 11 = {actual_weekly:.3f} cm/week")
    print(f"  Expected: {expected_weekly} cm/week")
    print(f"  Percentage: ({actual_weekly:.3f} / {expected_weekly}) × 100 = {percentage:.1f}%")
    return f"{int(round(percentage))}%"


def task_6_zone_a_heights():
    """Get Week 12 heights for all Zone-A plants"""
    zone_a_plants = [p for p in DATA["plants"] if p["zone"] == "Zone-A"]
    zone_a_plants.sort(key=lambda x: x["id"])
    
    results = []
    print("Task 6: Zone-A Week 12 heights")
    for plant in zone_a_plants:
        height = GROWTH_LOGS[plant["id"]][12]["height"]
        results.append(f"{plant['id']}: {height} cm")
        print(f"  {plant['id']}: {height} cm")
    
    return "; ".join(results)


def task_7_maria_watering_count():
    """Count Maria Santos watering events"""
    count = sum(1 for log in WATERING_LOGS if log["watered_by_name"] == "Maria Santos")
    
    print(f"Task 7: Maria Santos watering count")
    print(f"  Total entries: {count}")
    return str(count)


def task_8_complex_filter():
    """(Zone-A OR Zone-B) AND pot>=25 AND NOT 'Needs Attention'"""
    matching = []
    
    print("Task 8: Complex filter")
    for plant in DATA["plants"]:
        zone_ok = plant["zone"] in ["Zone-A", "Zone-B"]
        pot_ok = plant["pot_size_cm"] >= 25
        status_ok = plant["status"] != "Needs Attention"
        
        if zone_ok and pot_ok and status_ok:
            matching.append(plant["id"])
            print(f"  ✓ {plant['id']}: {plant['zone']}, pot={plant['pot_size_cm']}, status={plant['status']}")
        else:
            reason = []
            if not zone_ok: reason.append("wrong zone")
            if not pot_ok: reason.append(f"pot={plant['pot_size_cm']}")
            if not status_ok: reason.append(f"status={plant['status']}")
            # print(f"  ✗ {plant['id']}: {', '.join(reason)}")
    
    matching.sort()
    return "; ".join(matching)


def task_9_low_health_week8():
    """Find plants with health <= 7 in Week 8"""
    low_health = []
    
    print("Task 9: Week 8 health <= 7")
    for plant_id, weeks in GROWTH_LOGS.items():
        if 8 in weeks:
            health = weeks[8]["health"]
            if health <= 7:
                low_health.append((plant_id, health))
                print(f"  {plant_id}: {health}")
    
    low_health.sort(key=lambda x: x[0])
    return "; ".join(f"{pid}: {h}" for pid, h in low_health)


def task_10_caretaker_performance():
    """James Chen's average growth performance"""
    # Find James Chen's plants
    james_plants = [p for p in DATA["plants"] if p["caretaker_name"] == "James Chen"]
    
    print("Task 10: James Chen's plants performance")
    ratios = []
    
    for plant in james_plants:
        pid = plant["id"]
        species = plant["species"]
        expected = SPECIES[species]["growth_rate_cm_week"]
        
        week1 = GROWTH_LOGS[pid][1]["height"]
        week12 = GROWTH_LOGS[pid][12]["height"]
        actual = (week12 - week1) / 11
        
        ratio = actual / expected
        ratios.append(ratio)
        
        print(f"  {pid} ({species[:10]}...): actual={actual:.2f}, expected={expected}, ratio={ratio:.2%}")
    
    avg_ratio = sum(ratios) / len(ratios)
    percentage = avg_ratio * 100
    
    print(f"  Average ratio: {avg_ratio:.3f} = {percentage:.1f}%")
    return f"{int(round(percentage))}%"


def main():
    print("=" * 60)
    print("PLANT TASK VERIFICATION")
    print("=" * 60)
    print()
    
    answers = {}
    
    print("-" * 40)
    answers["task-plant-w1-five-monstera"] = task_1_w1_five_monstera()
    print()
    
    print("-" * 40)
    answers["task-plant-w3-indirect-reference"] = task_2_w3_indirect()
    print()
    
    print("-" * 40)
    answers["task-plant-modified-coefficient"] = task_3_modified_coefficient()
    print()
    
    print("-" * 40)
    answers["task-plant-n1-nutrition"] = task_4_n1_nutrition()
    print()
    
    print("-" * 40)
    answers["task-plant-growth-comparison"] = task_5_growth_comparison()
    print()
    
    print("-" * 40)
    answers["task-plant-zone-a-heights"] = task_6_zone_a_heights()
    print()
    
    print("-" * 40)
    answers["task-plant-caretaker-watering-count"] = task_7_maria_watering_count()
    print()
    
    print("-" * 40)
    answers["task-plant-complex-filter"] = task_8_complex_filter()
    print()
    
    print("-" * 40)
    answers["task-plant-low-health-week8"] = task_9_low_health_week8()
    print()
    
    print("-" * 40)
    answers["task-plant-caretaker-performance"] = task_10_caretaker_performance()
    print()
    
    print("=" * 60)
    print("SUMMARY OF VERIFIED ANSWERS")
    print("=" * 60)
    for task_id, answer in answers.items():
        print(f"{task_id}:")
        print(f"  {answer}")
    
    return answers


if __name__ == "__main__":
    main()

