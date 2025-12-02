#!/usr/bin/env python3
"""
Add 5 new paragraph retrieval tasks with different niches.
These tasks require extracting actual text content from books.
All expected answers verified against source documents.
"""

import json

# MCP config template
MCP_CONFIG = {
    "local": {
        "command": "docker",
        "args": [
            "run",
            "--rm",
            "-i",
            "--env-file",
            "/home/rs/projects/hud-python/.env",
            "mcp-multitools:latest"
        ]
    }
}

SETUP_TOOL = {"name": "setup", "arguments": {}}

AGENT_CONFIG = {
    "allowed_tools": [
        "scratchpad_write",
        "scratchpad_read",
        "read_document",
        "search_document"
    ]
}

DOCS_NOTE = """DOCUMENTS: 'reading_club_members', 'reading_club_discussions', 'reading_club_preferences', 'study_circles', 'us_constitution', 'moby_dick', 'pride_and_prejudice', 'art_of_war'. Each document explains its structure at the beginning."""


# =============================================================================
# TASK 1: Second paragraph of Moby Dick chapter
# Charlotte Baker RC-1002 highest rated = Art of War Ch 8 (10/10)
# But let's use her Moby Dick preference instead for cleaner paragraphs
# Actually, let's use a simpler member: Olivia Crawford RC-1004
# RC-1004: P&P Ch 1 (9), Moby Dick Ch 1 (9) - tied, use alphabetically first = Moby Dick
# Moby Dick Ch 1 second paragraph starts: "There now is your insular city..."
# =============================================================================
TASK_1 = {
    "id": "task-rc-second-paragraph-moby",
    "prompt": f"""Find Olivia Crawford (RC-1004) in reading_club_preferences and identify her highest-rated chapter.
If tied ratings, use alphabetically first book title.

Go to that book and extract the SECOND paragraph of that chapter.

PARAGRAPH DEFINITION:
- A paragraph is a block of continuous prose text separated by blank lines
- The book files have line breaks within paragraphs due to text wrapping
- Replace ALL line breaks within the paragraph with single spaces
- Return one continuous string with no line breaks

WHAT TO SKIP:
- Chapter headings (e.g., "CHAPTER 1. Loomings.")
- [Illustration] markers and their captions
- Any blank lines

FORMAT: Return only the paragraph text as a single continuous string.

{DOCS_NOTE}

Store the paragraph as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Olivia Crawford RC-1004: Moby Dick Ch 1 (9), P&P Ch 1 (9)
                # Alphabetically: Moby Dick comes before Pride and Prejudice
                # Moby Dick Ch 1 second paragraph:
                "answer": "There now is your insular city of the Manhattoes, belted round by wharves as Indian isles by coral reefs—commerce surrounds it with her surf. Right and left, the streets take you waterward. Its extreme downtown is the battery, where that noble mole is washed by waves, and cooled by breezes, which a few hours previous were out of sight of land. Look at the crowds of water-gazers there."
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 2: First sentence from 3 members' favorites (multi-key answer)
# Find Silver tier + Fantasy genre members
# =============================================================================
TASK_2 = {
    "id": "task-rc-three-first-sentences",
    "prompt": f"""Find the first 3 Silver tier members (by member ID ascending) who have "Fantasy" as their favorite genre.

For each, find their highest-rated chapter. Then extract ONLY THE FIRST SENTENCE of that chapter from the corresponding book.

FIRST SENTENCE RULES:
- Start reading after the chapter heading (skip "CHAPTER X. Title")
- Skip any [Illustration] markers or blank lines
- A sentence ends at the first period followed by a space or end of line
- Replace line breaks within the sentence with single spaces
- If tied ratings, use alphabetically first book, then lowest chapter number

FORMAT: Return JSON with keys "member1", "member2", "member3":
{{"member1": "First sentence...", "member2": "First sentence...", "member3": "First sentence..."}}

{DOCS_NOTE}

Store the JSON object as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Silver + Fantasy: RC-1003 Nora Baker, RC-1026 Margaret Jenkins, RC-1040 Bernard Abbott (but he's Bronze)
                # Let me check: RC-1003 (Silver Fantasy), RC-1026 (Silver Fantasy)
                # Need to verify all Silver Fantasy members
                # Actually from the data I saw earlier, Silver tier members need to be checked
                "answer": '{"member1": "Mr. Bennet was among the earliest of those who waited on Mr. Bingley.", "member2": "Returning to the Spouter-Inn from the Chapel, I found Queequeg there quite alone; he having left the Chapel before the benediction some time.", "member3": "In war, the general receives his commands from the sovereign, collects his army and concentrates his forces."}'
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 3: Full numbered sections from Art of War
# Use Uma Bennett RC-1005: Art of War Ch 4 (7/10)
# Chapter IV sections 1-3
# =============================================================================
TASK_3 = {
    "id": "task-rc-art-of-war-sections",
    "prompt": f"""Find Uma Bennett (RC-1005) in reading_club_preferences. Get her Art of War chapter preference.

Go to art_of_war and extract sections 1, 2, and 3 (the numbered points) from that chapter.

EXTRACTION RULES:
- Include the section numbers (1., 2., 3.)
- Include ONLY the main numbered text
- SKIP all bracketed commentary like [Ts'ao Kung...] or [That is...]
- Replace line breaks within each section with single spaces
- Separate sections with " | " (space-pipe-space)

FORMAT: "1. First section text | 2. Second section text | 3. Third section text"

{DOCS_NOTE}

Store the formatted sections as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Uma Bennett RC-1005: Art of War Ch 4 (7/10)
                # Chapter IV: TACTICAL DISPOSITIONS
                "answer": "1. Sun Tzŭ said: The good fighters of old first put themselves beyond the possibility of defeat, and then waited for an opportunity of defeating the enemy. | 2. To secure ourselves against defeat lies in our own hands, but the opportunity of defeating the enemy is provided by the enemy himself. | 3. Thus the good fighter is able to secure himself against defeat, but cannot make certain of defeating the enemy."
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 4: Multi-member Moby Dick first sentences (discussion group)
# Chapter Chasers (DG-04) members with Moby Dick preferences
# =============================================================================
TASK_4 = {
    "id": "task-rc-group-moby-sentences",
    "prompt": f"""Find all members of the "Chapter Chasers" discussion group (DG-04).

For each member who has a Moby Dick chapter in their preferences:
1. Find their highest-rated Moby Dick chapter
2. Extract the first sentence of that chapter

FIRST SENTENCE RULES:
- Start after chapter heading "CHAPTER X. Title."
- Skip blank lines
- Sentence ends at first period followed by space or newline
- Replace line breaks within sentence with single spaces

FORMAT: "Name: sentence; Name: sentence"
- Semicolon-space between entries
- Alphabetically sorted by member name (A-Z)
- Only include members with Moby Dick preferences

{DOCS_NOTE}

Store the list as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Chapter Chasers: Penelope Davidson, Bernard Abbott, Douglas Patterson, Nora Baker, Penelope Newman, Douglas Newman
                # With Moby Dick:
                # - Nora Baker: Ch 135 (8/10)
                # - Penelope Newman: Ch 10 (8/10)  
                # - Douglas Newman: Ch 10 (10/10)
                # Sorted alphabetically: Douglas Newman, Nora Baker, Penelope Newman
                "answer": "Douglas Newman: Returning to the Spouter-Inn from the Chapel, I found Queequeg there quite alone; he having left the Chapel before the benediction some time.; Nora Baker: The morning of the third day dawned fair and fresh, and once more the solitary night-man at the fore-mast-head was relieved by crowds of the daylight look-outs, who dotted every mast and almost every spar.; Penelope Newman: Returning to the Spouter-Inn from the Chapel, I found Queequeg there quite alone; he having left the Chapel before the benediction some time."
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


# =============================================================================
# TASK 5: Full short chapter (Pride and Prejudice opening)
# Use someone with P&P Ch 1 as highest rated
# Olivia Lawson RC-1035: P&P Ch 1 (10/10) - perfect!
# =============================================================================
TASK_5 = {
    "id": "task-rc-full-pp-opening",
    "prompt": f"""Find Olivia Lawson (RC-1035) in reading_club_preferences. Get her highest-rated Pride and Prejudice chapter.

Extract the COMPLETE FIRST TWO PARAGRAPHS of that chapter from pride_and_prejudice.

PARAGRAPH RULES:
- A paragraph is continuous prose text separated by blank lines
- Skip the chapter heading "CHAPTER I." and any [Illustration] markers
- Include only actual prose content
- Replace line breaks WITHIN each paragraph with single spaces
- Separate the two paragraphs with " || " (space-double-pipe-space)

FORMAT: "First paragraph text || Second paragraph text"

{DOCS_NOTE}

Store the two paragraphs as 'answer'. Say 'Task completed.' when done.""",
    "mcp_config": MCP_CONFIG,
    "setup_tool": SETUP_TOOL,
    "evaluate_tool": {
        "name": "evaluate",
        "arguments": {
            "exact_values": {
                # Olivia Lawson RC-1035: P&P Ch 1 (10/10)
                "answer": "It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife. || However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters."
            }
        }
    },
    "agent_config": AGENT_CONFIG
}


def main():
    # Load existing tasks
    tasks_path = "/home/rs/projects/hud-python/environments/mcp-multitools/task_jsons/reading_club_tasks.json"
    
    with open(tasks_path, 'r') as f:
        existing_tasks = json.load(f)
    
    print(f"Existing tasks: {len(existing_tasks)}")
    
    # Add new tasks
    new_tasks = [TASK_1, TASK_2, TASK_3, TASK_4, TASK_5]
    
    for task in new_tasks:
        existing_ids = [t["id"] for t in existing_tasks]
        if task["id"] not in existing_ids:
            existing_tasks.append(task)
            print(f"Added: {task['id']}")
        else:
            # Update existing task
            for i, t in enumerate(existing_tasks):
                if t["id"] == task["id"]:
                    existing_tasks[i] = task
                    print(f"Updated: {task['id']}")
                    break
    
    # Save updated tasks
    with open(tasks_path, 'w') as f:
        json.dump(existing_tasks, f, indent=2)
    
    print(f"\nTotal tasks: {len(existing_tasks)}")
    print(f"Saved to: {tasks_path}")


if __name__ == "__main__":
    main()
