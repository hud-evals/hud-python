# Flask Coding

A coding environment that gives the agent an isolated Flask repository and
grades its changes with hidden tests. Each task starts from a clean git
baseline, exposes only the working tree to the agent, and applies the hidden
test patch after the agent finishes.

The featured task asks the agent to add binary-file loading support to
`flask.Config.from_file` while preserving the existing text and error-handling
behavior. The score comes from targeted fail-to-pass and regression tests.

The workspace blocks network access and runs the agent as an unprivileged user.
Existing runs remain reproducible because each task pins its repository
baseline and test contract.

Deploy from this directory with:

```bash
hud deploy
```
