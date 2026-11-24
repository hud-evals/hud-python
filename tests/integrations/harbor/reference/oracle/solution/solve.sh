#!/bin/sh
set -e
# Reaches the internet through the agent's egress, as a solution that installs tools does.
python -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=30)"
printf '%s\n' "$WORD" > result.txt
id -un
