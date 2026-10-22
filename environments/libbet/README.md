# Libbet

A visual Game Boy environment built from the freely licensed homebrew puzzle
game *Libbet and the Magic Floor*. Agents see the current screen and play with
MCP tools for the directional pad, buttons, waiting, and screenshots.

The emulator advances only when a tool is called, so the game stays paused
while the agent reasons. The featured task gives the agent 40 action steps to
explore the game.

Scoring rewards visually distinct progress discovered during play rather than
requiring a particular level completion. Screenshots and scoring details are
recorded in the trace.

## Run the example task

```bash
uv sync
uv run hud eval tasks.py claude --runtime local -y
```

`tasks.json` defines the bundled 40-step task; `tasks.py` binds it to the local environment. To add another game, create a
`games/<name>/config.yaml` with its ROM, prompt, action timing, and scoring rules,
then add a `play-game` task row in `tasks.json`. Supply only ROMs you are allowed to
distribute. The included Libbet ROM and attribution are in `roms/`.

## Deploy

Deploy from this directory with:

```bash
hud deploy
```
