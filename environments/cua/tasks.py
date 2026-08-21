"""Example computer-use tasks."""

from env import cua_task
from env import env as env

_open_website = cua_task(
    prompt=(
        "A Chromium browser is open on the desktop. "
        "Navigate to https://www.wikipedia.org and wait for the page to fully load.\n\n"
        "Once the page is loaded, find the tagline shown below the Wikipedia logo. "
        "Reply with your answer as plain text."
    ),
    bash_checks=[
        {
            "name": "visited_wikipedia",
            "command": (
                "curl -fsS http://127.0.0.1:9222/json/list "
                "| jq -e 'any(.[]; .url == \"https://www.wikipedia.org/\")' >/dev/null"
            ),
            "weight": 1.0,
        }
    ],
    grading_criteria=[
        "The agent's answer mentions 'free encyclopedia' in any form - this is part of Wikipedia's tagline",
    ],
)
_open_website.slug = "open-website-example"


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


_shannon_research = cua_task(
    prompt=(
        "A Chromium browser and an XFCE desktop are available. Complete this "
        "multi-step research task, using the browser and the desktop:\n\n"
        "1. In the browser, go to https://en.wikipedia.org/wiki/Claude_Shannon "
        "and let the page load.\n"
        "2. Find the YEAR Claude Shannon was born.\n"
        "3. Find the university where he earned his PhD.\n"
        "4. Open that university's own Wikipedia article in a new tab and leave "
        "both articles open.\n"
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
        {
            "name": "file_contents",
            "command": (
                "cmp -s /home/ubuntu/Desktop/shannon.txt "
                "<(printf 'born: 1916\\nphd: Massachusetts Institute of Technology\\n"
                "city: Cambridge, Massachusetts\\n')"
            ),
            "weight": 1.0,
        },
        {
            "name": "research_pages_open",
            "command": (
                "curl -fsS http://127.0.0.1:9222/json/list | jq -e "
                "--arg shannon https://en.wikipedia.org/wiki/Claude_Shannon "
                "--arg mit https://en.wikipedia.org/wiki/Massachusetts_Institute_of_Technology "
                "'map(.url) | contains([$shannon, $mit])' >/dev/null"
            ),
            "weight": 1.0,
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
