"""Task definitions for the CUA environment.

`hud eval tasks.py` and `hud sync tasks` collect the public `tasks` list. Add a task by
calling `cua_task(...)`, setting a `.slug`, and adding it to the list. Verify a CUA env with a
real `hud eval tasks.py claude --runtime hud` rollout - the rfb desktop cannot run on macOS.
The multi-step research task wants more steps:

    hud eval tasks.py claude --task-ids shannon-multistep-research -y --max-steps 100
"""

from env import cua_task
from env import env as env

# Navigate to Wikipedia and read the tagline - LLM-judge grading
_open_website = cua_task(
    prompt=(
        "A Chromium browser is open on the desktop. "
        "Navigate to https://www.wikipedia.org and wait for the page to fully load.\n\n"
        "Once the page is loaded, find the tagline shown below the Wikipedia logo. "
        "Reply with your answer as plain text."
    ),
    grading_criteria=[
        "The agent's answer mentions 'free encyclopedia' in any form - this is part of Wikipedia's tagline",
    ],
)
_open_website.slug = "open-website-example"


# Write a text file to the Desktop - deterministic bash grading only
_create_document = cua_task(
    prompt=(
        "A Chromium browser and an XFCE desktop are available.\n\n"
        "Open a terminal (right-click the desktop and select 'Open Terminal Here', "
        "or find it in Applications > System) and create a file at "
        "/home/ubuntu/Desktop/hello.txt with exactly the content:\n"
        "Hello from HUD!\n\n"
        "You can use any method (echo, nano, cat, etc.)."
    ),
    bash_checks=[
        {"name": "file_exists", "command": "test -f /home/ubuntu/Desktop/hello.txt", "weight": 0.4},
        {
            "name": "content_correct",
            "command": "cmp -s /home/ubuntu/Desktop/hello.txt <(printf 'Hello from HUD!\\n')",
            "weight": 0.6,
        },
    ],
)
_create_document.slug = "create-document-example"


# Long multi-step task: research across two Wikipedia pages, then a cross-app terminal write.
# Exercises address-bar typing, Enter, link clicks, scrolling, multi-hop navigation, and the
# terminal. The authored bash weights sum to 1.0; env.py adds a 1.0 slot for the judge, so the
# final grade is a clean 0.5 bash / 0.5 judge that totals exactly 1.0:
#   file_exists 0.2 | file_has_birth_year 0.15 | file_has_mit 0.15 | llm_judge 0.5
_shannon_research = cua_task(
    prompt=(
        "A Chromium browser and an XFCE desktop are available. Complete this "
        "multi-step research task, using the browser and the desktop:\n\n"
        "1. In the browser, go to https://en.wikipedia.org/wiki/Claude_Shannon "
        "and let the page load.\n"
        "2. Find the YEAR Claude Shannon was born.\n"
        "3. Find the university where he earned his PhD.\n"
        "4. Navigate to that university's own Wikipedia article.\n"
        "5. Find the CITY and state where that university is located.\n"
        "6. Open a terminal (right-click the desktop and choose 'Open Terminal Here', "
        "or Applications > System) and save your findings to "
        "/home/ubuntu/Desktop/shannon.txt, one fact per line, exactly:\n"
        "   born: <year>\n"
        "   phd: <university>\n"
        "   city: <city, state>\n"
        "7. Reply with all three facts as plain text."
    ),
    bash_checks=[
        {"name": "file_exists", "command": "test -f /home/ubuntu/Desktop/shannon.txt", "weight": 0.4},
        {
            "name": "file_has_birth_year",
            "command": "grep -q '1916' /home/ubuntu/Desktop/shannon.txt",
            "weight": 0.3,
        },
        {
            "name": "file_has_mit",
            "command": "grep -qiE 'MIT|Massachusetts Institute' /home/ubuntu/Desktop/shannon.txt",
            "weight": 0.3,
        },
    ],
    grading_criteria=[
        "The agent states that Claude Shannon was born in 1916",
        "The agent states that Shannon earned his PhD at MIT (the Massachusetts Institute of Technology)",
        "The agent states that MIT is located in Cambridge, Massachusetts",
    ],
)
_shannon_research.slug = "shannon-multistep-research"


tasks = [_open_website, _create_document, _shannon_research]
