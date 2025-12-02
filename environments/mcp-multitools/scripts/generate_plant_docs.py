#!/usr/bin/env python3
"""
Generate interconnected plant growth management documents.
20 plants, 6 species, multiple data points, formulas, and logs.
All data flows from single source of truth - no contradictions.
"""

import random
import json
from datetime import datetime, timedelta

random.seed(123)  # Reproducible

# =============================================================================
# SINGLE SOURCE OF TRUTH
# =============================================================================

SPECIES = {
    "Monstera Deliciosa": {
        "code": "MON",
        "optimal_temp_min": 18,
        "optimal_temp_max": 27,
        "water_base_ml": 250,
        "light_hours_min": 6,
        "light_hours_max": 10,
        "growth_rate_cm_week": 2.5,
        "nutrition_npk": "10-10-10",
        "soil_ph_min": 5.5,
        "soil_ph_max": 7.0,
    },
    "Ficus Lyrata": {
        "code": "FIC",
        "optimal_temp_min": 16,
        "optimal_temp_max": 24,
        "water_base_ml": 300,
        "light_hours_min": 8,
        "light_hours_max": 12,
        "growth_rate_cm_week": 1.8,
        "nutrition_npk": "12-8-10",
        "soil_ph_min": 6.0,
        "soil_ph_max": 7.0,
    },
    "Pothos Golden": {
        "code": "POT",
        "optimal_temp_min": 15,
        "optimal_temp_max": 30,
        "water_base_ml": 150,
        "light_hours_min": 4,
        "light_hours_max": 8,
        "growth_rate_cm_week": 3.0,
        "nutrition_npk": "8-8-8",
        "soil_ph_min": 6.0,
        "soil_ph_max": 6.5,
    },
    "Snake Plant": {
        "code": "SNK",
        "optimal_temp_min": 13,
        "optimal_temp_max": 29,
        "water_base_ml": 100,
        "light_hours_min": 4,
        "light_hours_max": 10,
        "growth_rate_cm_week": 0.8,
        "nutrition_npk": "5-5-5",
        "soil_ph_min": 5.5,
        "soil_ph_max": 7.5,
    },
    "Peace Lily": {
        "code": "PLY",
        "optimal_temp_min": 18,
        "optimal_temp_max": 26,
        "water_base_ml": 200,
        "light_hours_min": 6,
        "light_hours_max": 8,
        "growth_rate_cm_week": 1.5,
        "nutrition_npk": "10-15-10",
        "soil_ph_min": 5.8,
        "soil_ph_max": 6.5,
    },
    "Rubber Plant": {
        "code": "RUB",
        "optimal_temp_min": 16,
        "optimal_temp_max": 27,
        "water_base_ml": 280,
        "light_hours_min": 6,
        "light_hours_max": 10,
        "growth_rate_cm_week": 2.0,
        "nutrition_npk": "10-10-10",
        "soil_ph_min": 5.5,
        "soil_ph_max": 7.0,
    },
}

ZONES = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]

CARETAKERS = [
    ("CT-01", "Maria Santos"),
    ("CT-02", "James Chen"),
    ("CT-03", "Elena Kowalski"),
    ("CT-04", "David Okonkwo"),
    ("CT-05", "Sarah Mitchell"),
]

NUTRIENT_TYPES = ["Liquid Fertilizer", "Slow Release Granules", "Foliar Spray", "Compost Tea"]

# =============================================================================
# GENERATE PLANTS
# =============================================================================

def generate_plants(count=20):
    plants = []
    species_list = list(SPECIES.keys())
    
    for i in range(count):
        species = species_list[i % len(species_list)]
        spec_data = SPECIES[species]
        plant_id = f"PLT-{spec_data['code']}-{i+1:03d}"
        
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60))
        zone = ZONES[i % len(ZONES)]
        caretaker = CARETAKERS[i % len(CARETAKERS)]
        
        initial_height = random.uniform(10, 30)
        
        plants.append({
            "id": plant_id,
            "species": species,
            "species_code": spec_data["code"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "zone": zone,
            "caretaker_id": caretaker[0],
            "caretaker_name": caretaker[1],
            "initial_height_cm": round(initial_height, 1),
            "pot_size_cm": random.choice([15, 20, 25, 30]),
            "status": random.choice(["Healthy", "Healthy", "Healthy", "Needs Attention", "Thriving"]),
        })
    
    return plants

# =============================================================================
# GENERATE GROWTH LOGS
# =============================================================================

def generate_growth_logs(plants, weeks=12):
    logs = []
    
    for plant in plants:
        spec_data = SPECIES[plant["species"]]
        current_height = plant["initial_height_cm"]
        current_width = current_height * random.uniform(0.3, 0.5)
        leaves = random.randint(4, 10)
        
        start = datetime.strptime(plant["start_date"], "%Y-%m-%d")
        
        for week in range(weeks):
            log_date = start + timedelta(weeks=week)
            
            # Growth with some randomness
            growth_rate = spec_data["growth_rate_cm_week"]
            height_gain = growth_rate * random.uniform(0.7, 1.3)
            current_height += height_gain
            current_width += height_gain * 0.3
            
            if random.random() > 0.7:
                leaves += random.randint(0, 2)
            
            logs.append({
                "plant_id": plant["id"],
                "date": log_date.strftime("%Y-%m-%d"),
                "week_number": week + 1,
                "height_cm": round(current_height, 1),
                "width_cm": round(current_width, 1),
                "leaf_count": leaves,
                "health_score": random.randint(7, 10),
                "notes": random.choice([
                    "Normal growth",
                    "Strong new growth",
                    "Slight yellowing on lower leaves",
                    "New leaf unfurling",
                    "Healthy development",
                    "",
                ]),
            })
    
    return logs

# =============================================================================
# GENERATE WATERING LOGS
# =============================================================================

def generate_watering_logs(plants, weeks=12):
    logs = []
    
    for plant in plants:
        spec_data = SPECIES[plant["species"]]
        base_water = spec_data["water_base_ml"]
        
        start = datetime.strptime(plant["start_date"], "%Y-%m-%d")
        
        # Water every 3-5 days depending on species
        water_interval = 4 if base_water > 200 else 5
        
        current_date = start
        end_date = start + timedelta(weeks=weeks)
        
        while current_date < end_date:
            soil_moisture_before = random.randint(20, 45)
            
            # Adjust water based on moisture
            moisture_factor = 1.0 + (40 - soil_moisture_before) / 100
            water_amount = int(base_water * moisture_factor * random.uniform(0.9, 1.1))
            
            soil_moisture_after = min(80, soil_moisture_before + random.randint(25, 40))
            
            caretaker = random.choice(CARETAKERS)
            
            logs.append({
                "plant_id": plant["id"],
                "date": current_date.strftime("%Y-%m-%d"),
                "water_ml": water_amount,
                "soil_moisture_before": soil_moisture_before,
                "soil_moisture_after": soil_moisture_after,
                "watered_by": caretaker[0],
                "watered_by_name": caretaker[1],
                "method": random.choice(["Top water", "Bottom water", "Top water"]),
            })
            
            current_date += timedelta(days=water_interval + random.randint(-1, 1))
    
    return logs

# =============================================================================
# GENERATE NUTRITION LOGS
# =============================================================================

def generate_nutrition_logs(plants, weeks=12):
    logs = []
    
    for plant in plants:
        spec_data = SPECIES[plant["species"]]
        start = datetime.strptime(plant["start_date"], "%Y-%m-%d")
        
        # Fertilize every 2-3 weeks
        current_date = start + timedelta(days=14)
        end_date = start + timedelta(weeks=weeks)
        
        while current_date < end_date:
            nutrient_type = random.choice(NUTRIENT_TYPES)
            
            # Amount based on pot size and type
            if "Liquid" in nutrient_type:
                amount = f"{random.randint(5, 15)} ml"
            elif "Granules" in nutrient_type:
                amount = f"{random.randint(10, 30)} g"
            elif "Foliar" in nutrient_type:
                amount = f"{random.randint(50, 100)} ml spray"
            else:
                amount = f"{random.randint(100, 200)} ml"
            
            logs.append({
                "plant_id": plant["id"],
                "date": current_date.strftime("%Y-%m-%d"),
                "nutrient_type": nutrient_type,
                "npk_ratio": spec_data["nutrition_npk"],
                "amount": amount,
                "applied_by": random.choice(CARETAKERS)[1],
            })
            
            current_date += timedelta(days=random.randint(14, 21))
    
    return logs

# =============================================================================
# GENERATE ENVIRONMENTAL DATA
# =============================================================================

def generate_environmental_data(weeks=12):
    logs = []
    start = datetime(2024, 1, 1)
    
    for zone in ZONES:
        base_temp = 20 + ZONES.index(zone) * 2  # Different zones have different temps
        base_humidity = 55 + ZONES.index(zone) * 5
        
        for day in range(weeks * 7):
            log_date = start + timedelta(days=day)
            
            # Seasonal variation
            seasonal_adj = 3 * (1 - abs(day - 45) / 45)  # Peak at day 45
            
            temp_high = base_temp + seasonal_adj + random.uniform(-2, 2)
            temp_low = temp_high - random.uniform(3, 6)
            humidity = base_humidity + random.uniform(-10, 10)
            light_hours = 8 + seasonal_adj + random.uniform(-1, 1)
            soil_ph = 6.2 + random.uniform(-0.3, 0.3)
            
            logs.append({
                "zone": zone,
                "date": log_date.strftime("%Y-%m-%d"),
                "temp_high_c": round(temp_high, 1),
                "temp_low_c": round(temp_low, 1),
                "avg_temp_c": round((temp_high + temp_low) / 2, 1),
                "humidity_percent": round(humidity, 1),
                "light_hours": round(light_hours, 1),
                "soil_ph_avg": round(soil_ph, 2),
            })
    
    return logs

# =============================================================================
# WRITE DOCUMENTS
# =============================================================================

def write_plant_registry(plants, filepath):
    lines = []
    lines.append("=" * 70)
    lines.append("GREENHOUSE PLANT REGISTRY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE & ORDERING")
    lines.append("-" * 40)
    lines.append("This registry contains all 20 active plants in the greenhouse.")
    lines.append("")
    lines.append("ORGANIZATION:")
    lines.append("  - Plants are grouped by ZONE (Zone-A, Zone-B, Zone-C, Zone-D)")
    lines.append("  - Within each zone, plants are listed by Plant ID ascending")
    lines.append("  - 5 plants per zone")
    lines.append("")
    lines.append("PLANT ID FORMAT: PLT-XXX-NNN")
    lines.append("  - PLT = Plant prefix")
    lines.append("  - XXX = Species code (MON, FIC, POT, SNK, PLY, RUB)")
    lines.append("  - NNN = Sequential number (001-020)")
    lines.append("")
    lines.append("SPECIES CODES:")
    lines.append("  MON = Monstera Deliciosa")
    lines.append("  FIC = Ficus Lyrata")
    lines.append("  POT = Pothos Golden")
    lines.append("  SNK = Snake Plant")
    lines.append("  PLY = Peace Lily")
    lines.append("  RUB = Rubber Plant")
    lines.append("")
    lines.append("RELATED DOCUMENTS:")
    lines.append("  - 'plant_growth_logs' - height/growth by plant ID, weekly")
    lines.append("  - 'plant_watering_logs' - watering records by plant ID")
    lines.append("  - 'plant_nutrition_logs' - fertilizer by plant ID")
    lines.append("  - 'environmental_data' - zone temperature/humidity by week")
    lines.append("  - 'care_formulas' - calculation formulas by species")
    lines.append("  - 'species_reference' - species care requirements")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    for zone in ZONES:
        zone_plants = [p for p in plants if p["zone"] == zone]
        lines.append(f"ZONE: {zone}")
        lines.append("-" * 50)
        lines.append(f"Plants in zone: {len(zone_plants)}")
        lines.append("")
        
        for p in zone_plants:
            lines.append(f"  Plant ID: {p['id']}")
            lines.append(f"    Species: {p['species']}")
            lines.append(f"    Started: {p['start_date']}")
            lines.append(f"    Initial Height: {p['initial_height_cm']} cm")
            lines.append(f"    Pot Size: {p['pot_size_cm']} cm")
            lines.append(f"    Caretaker: {p['caretaker_name']} ({p['caretaker_id']})")
            lines.append(f"    Status: {p['status']}")
            lines.append("")
        
        lines.append("")
    
    # Summary table
    lines.append("=" * 70)
    lines.append("CARETAKER ASSIGNMENTS")
    lines.append("-" * 50)
    for ct_id, ct_name in CARETAKERS:
        assigned = [p for p in plants if p["caretaker_id"] == ct_id]
        lines.append(f"  {ct_id} ({ct_name}): {len(assigned)} plants")
        for p in assigned:
            lines.append(f"    - {p['id']}: {p['species']}")
    lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def write_growth_logs(logs, filepath):
    lines = []
    lines.append("=" * 70)
    lines.append("PLANT GROWTH MEASUREMENT LOGS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE & ORDERING")
    lines.append("-" * 40)
    lines.append("Weekly growth measurements for all 20 plants over 12 weeks.")
    lines.append("")
    lines.append("ORGANIZATION:")
    lines.append("  - Plants listed alphabetically by Plant ID")
    lines.append("  - Each plant has 12 weekly entries (Week 1-12)")
    lines.append("  - Weeks ordered chronologically within each plant")
    lines.append("")
    lines.append("MEASUREMENT FIELDS:")
    lines.append("  - Height (cm): Current plant height")
    lines.append("  - Width (cm): Current plant spread")
    lines.append("  - Leaves: Number of leaves")
    lines.append("  - Health: Score 1-10 (10 = excellent)")
    lines.append("  - Notes: Optional observations")
    lines.append("")
    lines.append("SCHEDULE: Measurements taken every Monday")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    # Group by plant
    plant_ids = sorted(set(log["plant_id"] for log in logs))
    
    for plant_id in plant_ids:
        plant_logs = [l for l in logs if l["plant_id"] == plant_id]
        
        lines.append(f"PLANT: {plant_id}")
        lines.append("-" * 50)
        
        for log in sorted(plant_logs, key=lambda x: x["date"]):
            lines.append(f"  Week {log['week_number']:2d} ({log['date']})")
            lines.append(f"    Height: {log['height_cm']:6.1f} cm | Width: {log['width_cm']:5.1f} cm")
            lines.append(f"    Leaves: {log['leaf_count']:3d} | Health: {log['health_score']}/10")
            if log["notes"]:
                lines.append(f"    Notes: {log['notes']}")
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def write_watering_logs(logs, filepath):
    lines = []
    lines.append("=" * 70)
    lines.append("PLANT WATERING LOGS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("Records of all watering activities.")
    lines.append("Includes water amount, soil moisture readings, and method.")
    lines.append("")
    lines.append("SOIL MOISTURE SCALE: 0-100% (optimal range: 40-60%)")
    lines.append("WATERING METHODS: Top water, Bottom water")
    lines.append("")
    lines.append("For watering calculations, see 'care_formulas' document.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    # Group by plant
    plant_ids = sorted(set(log["plant_id"] for log in logs))
    
    for plant_id in plant_ids:
        plant_logs = [l for l in logs if l["plant_id"] == plant_id]
        
        lines.append(f"PLANT: {plant_id}")
        lines.append("-" * 50)
        
        for log in sorted(plant_logs, key=lambda x: x["date"]):
            lines.append(f"  {log['date']}")
            lines.append(f"    Water: {log['water_ml']:4d} ml | Method: {log['method']}")
            lines.append(f"    Moisture: {log['soil_moisture_before']}% -> {log['soil_moisture_after']}%")
            lines.append(f"    By: {log['watered_by_name']} ({log['watered_by']})")
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def write_nutrition_logs(logs, filepath):
    lines = []
    lines.append("=" * 70)
    lines.append("PLANT NUTRITION APPLICATION LOGS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("Records of fertilizer and nutrient applications.")
    lines.append("NPK ratios are species-specific (see 'species_reference').")
    lines.append("")
    lines.append("APPLICATION FREQUENCY: Every 2-3 weeks during growing season")
    lines.append("")
    lines.append("For nutrition calculations, see 'care_formulas' document.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    # Group by plant
    plant_ids = sorted(set(log["plant_id"] for log in logs))
    
    for plant_id in plant_ids:
        plant_logs = [l for l in logs if l["plant_id"] == plant_id]
        
        lines.append(f"PLANT: {plant_id}")
        lines.append("-" * 50)
        
        for log in sorted(plant_logs, key=lambda x: x["date"]):
            lines.append(f"  {log['date']}")
            lines.append(f"    Type: {log['nutrient_type']}")
            lines.append(f"    NPK: {log['npk_ratio']} | Amount: {log['amount']}")
            lines.append(f"    Applied by: {log['applied_by']}")
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def write_environmental_data(logs, filepath):
    lines = []
    lines.append("=" * 70)
    lines.append("GREENHOUSE ENVIRONMENTAL DATA")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("Daily environmental readings by zone.")
    lines.append("Includes temperature, humidity, light hours, and soil pH.")
    lines.append("")
    lines.append("ZONES: Zone-A, Zone-B, Zone-C, Zone-D")
    lines.append("  Zone-A: Cooler, lower humidity (tropical understory)")
    lines.append("  Zone-B: Moderate temperature and humidity")
    lines.append("  Zone-C: Warmer, higher humidity (tropical)")
    lines.append("  Zone-D: Warmest, highest humidity")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    for zone in ZONES:
        zone_logs = [l for l in logs if l["zone"] == zone]
        
        lines.append(f"ZONE: {zone}")
        lines.append("-" * 50)
        
        # Show weekly summaries instead of daily to save space
        weeks = {}
        for log in zone_logs:
            week_start = datetime.strptime(log["date"], "%Y-%m-%d")
            week_key = week_start.strftime("%Y-W%W")
            if week_key not in weeks:
                weeks[week_key] = []
            weeks[week_key].append(log)
        
        for week_key in sorted(weeks.keys()):
            week_logs = weeks[week_key]
            avg_temp = sum(l["avg_temp_c"] for l in week_logs) / len(week_logs)
            avg_humidity = sum(l["humidity_percent"] for l in week_logs) / len(week_logs)
            avg_light = sum(l["light_hours"] for l in week_logs) / len(week_logs)
            avg_ph = sum(l["soil_ph_avg"] for l in week_logs) / len(week_logs)
            
            lines.append(f"  Week {week_key}")
            lines.append(f"    Avg Temp: {avg_temp:.1f}°C | Humidity: {avg_humidity:.1f}%")
            lines.append(f"    Light: {avg_light:.1f} hrs/day | Soil pH: {avg_ph:.2f}")
        
        lines.append("")
        
        # Also show some daily details for the last week
        lines.append(f"  DAILY DETAILS (Last 7 days):")
        last_week = sorted(zone_logs, key=lambda x: x["date"])[-7:]
        for log in last_week:
            lines.append(f"    {log['date']}: {log['temp_low_c']:.1f}-{log['temp_high_c']:.1f}°C, "
                        f"{log['humidity_percent']:.0f}% humidity, {log['light_hours']:.1f}h light")
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def write_care_formulas(filepath):
    """Write formulas and equations for plant care calculations."""
    lines = []
    lines.append("=" * 70)
    lines.append("PLANT CARE FORMULAS AND CALCULATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("This document contains all formulas for calculating:")
    lines.append("  1. Watering amounts based on conditions")
    lines.append("  2. Nutrition amounts based on growth")
    lines.append("  3. Expected growth rates")
    lines.append("  4. Health score interpretations")
    lines.append("")
    lines.append("Variables used in formulas:")
    lines.append("  T = Average temperature (°C)")
    lines.append("  H = Humidity (%)")
    lines.append("  M = Soil moisture before watering (%)")
    lines.append("  L = Light hours per day")
    lines.append("  W = Current plant width (cm)")
    lines.append("  Ht = Current plant height (cm)")
    lines.append("  A = Days since last application")
    lines.append("  P = Pot size (cm)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    # Watering formulas
    lines.append("SECTION 1: WATERING FORMULAS")
    lines.append("=" * 50)
    lines.append("")
    
    for species, data in SPECIES.items():
        lines.append(f"Species: {species} (Code: {data['code']})")
        lines.append("-" * 40)
        base = data["water_base_ml"]
        lines.append(f"  Base Water Amount: {base} ml")
        lines.append("")
        lines.append("  FORMULA W1 - Temperature Adjustment:")
        lines.append(f"    Water_ml = {base} × (1 + (T - 22) × 0.03)")
        lines.append(f"    Example: At 26°C: {base} × (1 + (26-22) × 0.03) = {int(base * 1.12)} ml")
        lines.append("")
        lines.append("  FORMULA W2 - Moisture Adjustment:")
        lines.append(f"    Water_ml = {base} × (1 + (40 - M) / 100)")
        lines.append(f"    Example: At 30% moisture: {base} × (1 + (40-30)/100) = {int(base * 1.1)} ml")
        lines.append("")
        lines.append("  FORMULA W3 - Combined (use this):")
        lines.append(f"    Water_ml = {base} × (1 + (T - 22) × 0.03) × (1 + (40 - M) / 100) × (P / 20)")
        lines.append("")
    
    # Nutrition formulas
    lines.append("")
    lines.append("SECTION 2: NUTRITION FORMULAS")
    lines.append("=" * 50)
    lines.append("")
    
    for species, data in SPECIES.items():
        lines.append(f"Species: {species} (Code: {data['code']})")
        lines.append("-" * 40)
        lines.append(f"  Recommended NPK: {data['nutrition_npk']}")
        lines.append("")
        lines.append("  FORMULA N1 - Liquid Fertilizer Amount:")
        lines.append(f"    Fertilizer_ml = (Ht / 10) × (P / 15) × 2")
        lines.append("    Apply every 14-21 days during growing season")
        lines.append("")
        lines.append("  FORMULA N2 - Growth-Based Adjustment:")
        lines.append("    If weekly growth < expected: increase by 20%")
        lines.append(f"    Expected weekly growth: {data['growth_rate_cm_week']} cm")
        lines.append("")
        lines.append("  FORMULA N3 - Dilution Ratio:")
        lines.append("    Standard dilution: 1:100 (fertilizer:water)")
        lines.append("    Young plants (<30cm): 1:150 (weaker)")
        lines.append("    Mature plants (>60cm): 1:80 (stronger)")
        lines.append("")
    
    # Growth estimation
    lines.append("")
    lines.append("SECTION 3: GROWTH ESTIMATION")
    lines.append("=" * 50)
    lines.append("")
    
    for species, data in SPECIES.items():
        lines.append(f"Species: {species}")
        lines.append("-" * 40)
        rate = data["growth_rate_cm_week"]
        lines.append(f"  Base Growth Rate: {rate} cm/week")
        lines.append("")
        lines.append("  FORMULA G1 - Temperature Effect:")
        t_min, t_max = data["optimal_temp_min"], data["optimal_temp_max"]
        lines.append(f"    Optimal range: {t_min}-{t_max}°C")
        lines.append(f"    Growth_factor = 1 - |T - {(t_min+t_max)/2:.0f}| × 0.05")
        lines.append("")
        lines.append("  FORMULA G2 - Light Effect:")
        l_min, l_max = data["light_hours_min"], data["light_hours_max"]
        lines.append(f"    Optimal light: {l_min}-{l_max} hours/day")
        lines.append(f"    If L < {l_min}: Growth reduced by 30%")
        lines.append(f"    If L > {l_max}: Growth reduced by 10%")
        lines.append("")
        lines.append("  FORMULA G3 - Weekly Height Estimate:")
        lines.append(f"    New_height = Current_height + ({rate} × Growth_factor × Light_factor)")
        lines.append("")
    
    # Health score
    lines.append("")
    lines.append("SECTION 4: HEALTH SCORE INTERPRETATION")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Score | Status         | Action Required")
    lines.append("-" * 50)
    lines.append("9-10  | Thriving       | Maintain current care")
    lines.append("7-8   | Healthy        | No changes needed")
    lines.append("5-6   | Needs Attention| Check water/light/nutrients")
    lines.append("3-4   | Stressed       | Immediate intervention")
    lines.append("1-2   | Critical       | Emergency care protocol")
    lines.append("")
    lines.append("FORMULA H1 - Health Score Calculation:")
    lines.append("  Base Score = 10")
    lines.append("  Deductions:")
    lines.append("    - Temperature outside optimal: -1 per 3°C deviation")
    lines.append("    - Soil moisture <30% or >70%: -1")
    lines.append("    - Growth rate <50% expected: -2")
    lines.append("    - No fertilizer in 30+ days: -1")
    lines.append("    - Pest/disease visible: -3")
    lines.append("")
    
    # Quick reference coefficients
    lines.append("")
    lines.append("SECTION 5: QUICK REFERENCE COEFFICIENTS")
    lines.append("=" * 50)
    lines.append("")
    lines.append("TEMPERATURE COEFFICIENTS (multiply water amount)")
    lines.append("-" * 40)
    lines.append("  < 15°C:  0.70")
    lines.append("  15-18°C: 0.85")
    lines.append("  18-22°C: 1.00")
    lines.append("  22-26°C: 1.15")
    lines.append("  26-30°C: 1.30")
    lines.append("  > 30°C:  1.50")
    lines.append("")
    lines.append("HUMIDITY COEFFICIENTS (multiply water amount)")
    lines.append("-" * 40)
    lines.append("  < 40%:   1.20")
    lines.append("  40-50%:  1.10")
    lines.append("  50-60%:  1.00")
    lines.append("  60-70%:  0.90")
    lines.append("  > 70%:   0.80")
    lines.append("")
    lines.append("POT SIZE COEFFICIENTS (multiply all amounts)")
    lines.append("-" * 40)
    lines.append("  15 cm:   0.75")
    lines.append("  20 cm:   1.00")
    lines.append("  25 cm:   1.25")
    lines.append("  30 cm:   1.50")
    lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def write_species_reference(filepath):
    """Write species-specific reference information."""
    lines = []
    lines.append("=" * 70)
    lines.append("PLANT SPECIES REFERENCE GUIDE")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DOCUMENT STRUCTURE")
    lines.append("-" * 40)
    lines.append("Comprehensive care requirements for each species.")
    lines.append("Use this document to determine optimal conditions.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    
    for species, data in SPECIES.items():
        lines.append(f"SPECIES: {species}")
        lines.append(f"Code: {data['code']}")
        lines.append("=" * 50)
        lines.append("")
        lines.append("ENVIRONMENTAL REQUIREMENTS")
        lines.append("-" * 40)
        lines.append(f"  Temperature Range: {data['optimal_temp_min']}-{data['optimal_temp_max']}°C")
        lines.append(f"  Optimal Temperature: {(data['optimal_temp_min']+data['optimal_temp_max'])/2:.0f}°C")
        lines.append(f"  Light Requirements: {data['light_hours_min']}-{data['light_hours_max']} hours/day")
        lines.append(f"  Soil pH Range: {data['soil_ph_min']}-{data['soil_ph_max']}")
        lines.append("")
        lines.append("WATER REQUIREMENTS")
        lines.append("-" * 40)
        lines.append(f"  Base Water Amount: {data['water_base_ml']} ml per watering")
        lines.append(f"  Watering Frequency: Every {3 if data['water_base_ml'] > 200 else 5} days (adjust based on conditions)")
        lines.append(f"  Preferred Soil Moisture: 40-60%")
        lines.append("")
        lines.append("NUTRITION REQUIREMENTS")
        lines.append("-" * 40)
        lines.append(f"  NPK Ratio: {data['nutrition_npk']}")
        lines.append(f"  Fertilizing Frequency: Every 2-3 weeks")
        lines.append("")
        lines.append("GROWTH CHARACTERISTICS")
        lines.append("-" * 40)
        lines.append(f"  Expected Growth Rate: {data['growth_rate_cm_week']} cm/week")
        lines.append(f"  Monthly Growth Estimate: {data['growth_rate_cm_week'] * 4:.1f} cm/month")
        lines.append("")
        lines.append("CARE NOTES")
        lines.append("-" * 40)
        if "Monstera" in species:
            lines.append("  - Likes to climb; provide moss pole for best growth")
            lines.append("  - Fenestrations appear on mature leaves")
            lines.append("  - Sensitive to cold drafts")
        elif "Ficus" in species:
            lines.append("  - Drops leaves when stressed or moved")
            lines.append("  - Prefers consistent conditions")
            lines.append("  - Wipe leaves monthly to remove dust")
        elif "Pothos" in species:
            lines.append("  - Very tolerant of low light")
            lines.append("  - Can be grown in water")
            lines.append("  - Trail or climb with support")
        elif "Snake" in species:
            lines.append("  - Extremely drought tolerant")
            lines.append("  - Sensitive to overwatering (root rot)")
            lines.append("  - Tolerates low light well")
        elif "Peace" in species:
            lines.append("  - Droops dramatically when thirsty")
            lines.append("  - Flowers with adequate light")
            lines.append("  - Sensitive to chlorine in tap water")
        elif "Rubber" in species:
            lines.append("  - Wipe leaves to maintain shine")
            lines.append("  - Prune to encourage bushiness")
            lines.append("  - Sap can irritate skin")
        lines.append("")
        lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {filepath} ({len(lines)} lines)")


def save_ground_truth(plants, growth_logs, watering_logs, nutrition_logs, env_data, filepath):
    """Save all data for task generation."""
    data = {
        "plants": plants,
        "species": SPECIES,
        "caretakers": [{"id": c[0], "name": c[1]} for c in CARETAKERS],
        "zones": ZONES,
        "growth_logs": growth_logs,
        "watering_logs": watering_logs,
        "nutrition_logs": nutrition_logs,
        "environmental_data": env_data,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Written: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Generating Plant Growth Documents...")
    print("=" * 50)
    
    # Generate all data
    plants = generate_plants(20)
    growth_logs = generate_growth_logs(plants, weeks=12)
    watering_logs = generate_watering_logs(plants, weeks=12)
    nutrition_logs = generate_nutrition_logs(plants, weeks=12)
    env_data = generate_environmental_data(weeks=12)
    
    # Write documents
    write_plant_registry(plants, "documents/plant_registry.txt")
    write_growth_logs(growth_logs, "documents/plant_growth_logs.txt")
    write_watering_logs(watering_logs, "documents/plant_watering_logs.txt")
    write_nutrition_logs(nutrition_logs, "documents/plant_nutrition_logs.txt")
    write_environmental_data(env_data, "documents/environmental_data.txt")
    write_care_formulas("documents/care_formulas.txt")
    write_species_reference("documents/species_reference.txt")
    
    # Save ground truth
    save_ground_truth(plants, growth_logs, watering_logs, nutrition_logs, env_data,
                      "scripts/plant_data.json")
    
    print("=" * 50)
    print("Done! Generated 7 interconnected documents:")
    print("  1. plant_registry.txt - Master plant list")
    print("  2. plant_growth_logs.txt - Height/growth measurements")
    print("  3. plant_watering_logs.txt - Watering records")
    print("  4. plant_nutrition_logs.txt - Fertilizer applications")
    print("  5. environmental_data.txt - Zone conditions")
    print("  6. care_formulas.txt - Calculation formulas")
    print("  7. species_reference.txt - Species care guide")


if __name__ == "__main__":
    main()

