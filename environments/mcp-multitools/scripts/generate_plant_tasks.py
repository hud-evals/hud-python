#!/usr/bin/env python3
"""
Generate 10 plant growth management tasks.
5 formula-based tasks, 5 data retrieval tasks.
"""

import json

MCP_CONFIG = {
    "local": {
        "command": "docker",
        "args": [
            "run", "--rm", "-i",
            "--env-file", "/home/rs/projects/hud-python/.env",
            "mcp-multitools:latest"
        ]
    }
}

SETUP_TOOL = {"name": "setup", "arguments": {}}
AGENT_CONFIG = {
    "allowed_tools": ["scratchpad_write", "scratchpad_read", "read_document", "search_document"]
}

DOCS_NOTE = """DOCUMENTS: 'plant_registry', 'plant_growth_logs', 'plant_watering_logs', 'plant_nutrition_logs', 'environmental_data', 'care_formulas', 'species_reference'. Each document explains its structure at the beginning."""


# =============================================================================
# TASK 1: Simple W1 formula for 5 Monstera plants
# Find all Monstera, use W1 with T=24°C, return sum
# =============================================================================
TASK_1 = {
    "id": "task-plant-w1-five-monstera",
    "prompt": f"""Find all Monstera Deliciosa plants in the greenhouse.

Using formula W1 from care_formulas (Temperature Adjustment), calculate the water amount for each Monstera at T = 24°C.

Return the SUM of water amounts for all Monstera plants.

FORMAT: Just the number in ml, rounded to nearest integer (e.g., "1120")

{DOCS_NOTE}

Store the sum as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Monstera W1: Water_ml = 250 × (1 + (T - 22) × 0.03)
                # At T=24: 250 × (1 + 2 × 0.03) = 250 × 1.06 = 265 ml per plant
                # 4 Monstera plants (MON-001, MON-007, MON-013, MON-019): 4 × 265 = 1060
                "answer": "1060"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 2: W3 formula for 1 plant referenced indirectly
# Find Elena Kowalski's Pothos in Zone-C, use W3
# =============================================================================
TASK_2 = {
    "id": "task-plant-w3-indirect-reference",
    "prompt": f"""Find the Pothos Golden plant that is:
- In Zone-C
- Assigned to Elena Kowalski

Using formula W3 from care_formulas, calculate the recommended water amount.
Use these values: T = 23°C, M = 35% (soil moisture before watering).
Get the pot size (P) from the plant registry.

Return the calculated water amount.

FORMAT: Number in ml, rounded to nearest integer (e.g., "187")

{DOCS_NOTE}

Store the water amount as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # PLT-POT-003: Zone-C, Elena Kowalski, pot_size=25cm
                # Pothos W3: Water_ml = 150 × (1 + (T-22) × 0.03) × (1 + (40-M)/100) × (P/20)
                # = 150 × (1 + 1×0.03) × (1 + 5/100) × (25/20)
                # = 150 × 1.03 × 1.05 × 1.25
                # = 150 × 1.353... = 202.9 ≈ 203
                "answer": "203"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 3: Coefficient override - temperature coefficient increased by 0.02
# =============================================================================
TASK_3 = {
    "id": "task-plant-modified-coefficient",
    "prompt": f"""IMPORTANT: Due to recent calibration, the temperature coefficient in formula W1 should be increased by 0.02 (making it 0.05 instead of the documented 0.03).

Find plant PLT-FIC-002 (Ficus Lyrata).

Using the MODIFIED formula W1 (with coefficient 0.05), calculate the water amount at T = 26°C.

Show your calculation and return the final water amount.

FORMAT: Just the number in ml, rounded to nearest integer (e.g., "360")

{DOCS_NOTE}

Store the water amount as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Ficus W1 normal: Water_ml = 300 × (1 + (T-22) × 0.03)
                # Modified: Water_ml = 300 × (1 + (T-22) × 0.05)
                # At T=26: 300 × (1 + 4 × 0.05) = 300 × 1.20 = 360
                "answer": "360"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 4: N1 nutrition formula for a specific plant
# =============================================================================
TASK_4 = {
    "id": "task-plant-n1-nutrition",
    "prompt": f"""Find the current height (Week 12) of plant PLT-MON-013 from plant_growth_logs.

Using formula N1 from care_formulas, calculate the recommended liquid fertilizer amount.
Use the pot size from plant_registry and the Week 12 height as Ht.

Return the calculated fertilizer amount.

FORMAT: Number in ml, rounded to 1 decimal place (e.g., "8.5")

{DOCS_NOTE}

Store the fertilizer amount as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # PLT-MON-013: pot_size=30cm, Week 12 height=58.2cm
                # N1: Fertilizer_ml = (Ht / 10) × (P / 15) × 2
                # = (58.2/10) × (30/15) × 2 = 5.82 × 2 × 2 = 23.28
                "answer": "23.3"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 5: Growth comparison - actual vs expected
# =============================================================================
TASK_5 = {
    "id": "task-plant-growth-comparison",
    "prompt": f"""For plant PLT-FIC-008 (Ficus Lyrata):

1. Find its Week 1 and Week 12 heights from plant_growth_logs
2. Calculate the ACTUAL average weekly growth: (Week12 - Week1) / 11
3. Find the EXPECTED weekly growth rate from species_reference
4. Calculate: (Actual / Expected) × 100 to get percentage of expected growth

Return the percentage.

FORMAT: Number rounded to nearest integer with % sign (e.g., "95%")

{DOCS_NOTE}

Store the percentage as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # PLT-FIC-008: Ficus Lyrata
                # Week 1: 25.2 cm, Week 12: 44.0 cm
                # Actual: (44.0 - 25.2) / 11 = 1.709 cm/week
                # Expected: 1.8 cm/week
                # Percentage: (1.709 / 1.8) × 100 = 95%
                "answer": "95%"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 6: Retrieve Zone-A plant heights at Week 12
# =============================================================================
TASK_6 = {
    "id": "task-plant-zone-a-heights",
    "prompt": f"""Find all plants in Zone-A from plant_registry.

For each Zone-A plant, find its Week 12 height from plant_growth_logs.

Return the plant IDs and their Week 12 heights.

FORMAT: "PLT-XXX-NNN: HH.H cm; PLT-XXX-NNN: HH.H cm; ..."
- Entries separated by "; " (semicolon space)
- Sorted by Plant ID ascending
- Heights to 1 decimal place

{DOCS_NOTE}

Store the list as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Zone-A plants sorted by ID: MON-001, MON-013, PLY-005, PLY-017, POT-009
                # Week 12 heights: 46.1, 58.2, 28.2, 37.0, 62.7
                "answer": "PLT-MON-001: 46.1 cm; PLT-MON-013: 58.2 cm; PLT-PLY-005: 28.2 cm; PLT-PLY-017: 37.0 cm; PLT-POT-009: 62.7 cm"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 7: Count total watering events for a caretaker
# =============================================================================
TASK_7 = {
    "id": "task-plant-caretaker-watering-count",
    "prompt": f"""Count how many times Maria Santos (CT-01) performed watering across ALL plants.

Look through plant_watering_logs and count every watering entry where Maria Santos is listed as the person who watered.

Return the total count.

FORMAT: Just the number (e.g., "42")

{DOCS_NOTE}

Store the count as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Grep count: 100 entries for Maria Santos
                "answer": "100"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 8: AND/OR/NOT combined query
# =============================================================================
TASK_8 = {
    "id": "task-plant-complex-filter",
    "prompt": f"""Find plants that match ALL of these criteria:
1. (Zone-A OR Zone-B)
2. AND pot size >= 25 cm
3. AND NOT status "Needs Attention"

Return the plant IDs that match.

FORMAT: "PLT-XXX-NNN; PLT-XXX-NNN; ..."
- Separated by "; " (semicolon space)
- Sorted alphabetically by Plant ID

{DOCS_NOTE}

Store the list as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Zone-A: MON-001(30), PLY-005(30), POT-009(30), MON-013(30), PLY-017(30)
                # Zone-B: FIC-002(15,Needs Attention), RUB-006(15), SNK-010(25), FIC-014(30), RUB-018(25)
                # Filter: pot>=25, not "Needs Attention"
                # Zone-A all pass (pot=30, all healthy/thriving)
                # Zone-B: SNK-010(25,Healthy), FIC-014(30,Thriving), RUB-018(25,Healthy)
                "answer": "PLT-FIC-014; PLT-MON-001; PLT-MON-013; PLT-PLY-005; PLT-PLY-017; PLT-POT-009; PLT-RUB-018; PLT-SNK-010"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 9: Find plants with low health scores
# =============================================================================
TASK_9 = {
    "id": "task-plant-low-health-week8",
    "prompt": f"""Find all plants that had a health score of 7 or below during Week 8.

Return the plant IDs and their Week 8 health scores.

FORMAT: "PLT-XXX-NNN: N; PLT-XXX-NNN: N; ..."
- Separated by "; " (semicolon space)
- Sorted by Plant ID ascending
- N is the health score (integer)

{DOCS_NOTE}

Store the list as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Verified: sorted by plant ID
                "answer": "PLT-FIC-008: 7; PLT-MON-001: 7; PLT-MON-019: 7; PLT-POT-009: 7; PLT-RUB-006: 7; PLT-RUB-012: 7; PLT-SNK-004: 7"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 10: Multi-step - caretaker's plants average growth vs species expected
# =============================================================================
TASK_10 = {
    "id": "task-plant-caretaker-performance",
    "prompt": f"""Evaluate James Chen's (CT-02) plant care performance:

1. Find all plants assigned to James Chen from plant_registry
2. For each plant, calculate actual weekly growth: (Week12 height - Week1 height) / 11
3. Get the expected growth rate for each plant's species from species_reference
4. Calculate: average of (actual / expected) across all his plants
5. Return the average as a percentage

FORMAT: Number rounded to nearest integer with % sign (e.g., "105%")

{DOCS_NOTE}

Store the percentage as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Verified: FIC-002(95%), MON-007(108%), RUB-012(105%), PLY-017(112%)
                # Average: 104.9% -> 105%
                "answer": "105%"
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


def main():
    tasks = [TASK_1, TASK_2, TASK_3, TASK_4, TASK_5, 
             TASK_6, TASK_7, TASK_8, TASK_9, TASK_10]
    
    output_path = "/home/rs/projects/hud-python/environments/mcp-multitools/task_jsons/plant_tasks.json"
    
    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Generated {len(tasks)} plant tasks")
    print(f"Saved to: {output_path}")
    
    print("\nTask summary:")
    for t in tasks:
        print(f"  - {t['id']}")


if __name__ == "__main__":
    main()

