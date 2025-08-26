# HUD Deep Research MCP Server

Local Playwright-based environment tailored for reading and analyzing Wikipedia pages.

## Build

```bash
docker build -t hud-deep-research:dev .
```

## Run (production)

```bash
docker run --rm -i \
  -e INITIAL_URL="https://en.wikipedia.org/wiki/Main_Page" \
  hud-deep-research:dev
```

## Develop (hot-reload)

```bash
# From repo root
python -m hud.cli dev environments/deep_research --build
```

## Debug (stdio inspector)

```bash
# Basic MCP initialize + list tools
python -m hud.cli debug environments/deep_research --max-phase 2 --build
```

## Tools

- setup.navigate(url)
- setup.open_wikipedia_page(title, lang="en")
- evaluate.url_match(expected_substring)
- evaluate.page_contains(search_terms, partial_rewarding=True)
- playwright (navigate, screenshot, click, type, wait_for_element, get_page_info)

## Environment variables

- INITIAL_URL: Default page to open. Defaults to Wikipedia Main Page.
- BROWSER_URL: Alternative variable name for initial URL.

