# Rocket animation

A browser simulation coding task and a starting point for a new HUD environment. The agent implements
`sceneAt(t)` in JavaScript, previews selected frames, and submits its workspace.
The provided canvas viewer supplies playback and scrubbing. It draws the exact
validated state returned by the submitted module, which runs in a DOM-free worker.

Grading samples the animation in both directions and checks ignition, liftoff,
visible smoke emission and fading, localized angular max-Q shake, coherent breakup,
inherited velocity, gravity, and fragment rotation. It grades the artifact rather
than the final written answer. This is a constrained physics coding demo, not an
open-ended visual design benchmark.

The client brief describes the animation and viewer contract without disclosing
scoring weights or sampling rules. Keep grading aligned with those outcomes:
accept different trajectories, smoke styles, and shake patterns that satisfy the
brief. Before changing the grader, check an independently written valid scene
and variants with incorrect gravity or lost breakup momentum in the real viewer.

The preview tool exports `rocket-preview.html` (standalone interactive replay)
and `rocket-sequence.png` (nine rendered frames) into the tracked workspace before
the agent finishes, so HUD preserves them as trace downloads. Final grading
rechecks the submitted code and records whether the last preview matches it.
Preview tool calls also put screenshots directly in the trace.
The final `save_replay` tool also exports an animated WebP for motion playback
inside HUD's existing Workspace image preview. Interactive HTML must be downloaded:
the platform's HTML iframe intentionally blocks scripts.
The starter remains unsolved; reference implementations exist only in tests, which
are excluded from the deployment image.

```sh
hud init my-animation --template rocket-animation
cd my-animation
docker build --platform linux/amd64 -f Dockerfile.hud -t rocket-animation .
hud deploy . --project <project-id> --no-env --runtime hud
hud sync tasks <taskset-name> tasks.py --yes
```

`tasks.json` uses the stable environment template ID
`rocket-launch-and-failure`. The platform task's human-readable name is
**Rocket launch and failure**.

`tasks.json` is the shared task definition. `tasks.py` binds it to the current
environment name, including the name chosen by `hud init`. Customize the brief
and starter files for another animation task, and keep its grader aligned.

The full client brief is stored in `args.prompt`, so it appears in the task's
Parameters panel. The environment uses that value for both the agent's opening
message and `/workspace/README.md`. Older tasks with empty args still use the
canonical brief from `starter/README.md`. Keep `tasks.json`'s prompt aligned with
that file; do not duplicate the brief in the agent's system prompt.

Dependencies target Linux/Python 3.12; regenerate the lock with:

```sh
uv pip compile requirements.in --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements.txt
```
