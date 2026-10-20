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

Deploy from this directory with:

```bash
hud deploy
```
