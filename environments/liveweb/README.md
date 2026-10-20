# LiveWeb research

Research arbitrary live websites in a Chromium browser and Linux desktop. Agents
can use their browser tools over CDP, computer tools over RFB, and an isolated
shell workspace. Task arguments choose the starting URL, question and grading
rubric. This differs from `browser-use`, whose bundled task requires verified
screenshot evidence from one rental website.

```sh
hud init my-research --template liveweb
cd my-research
hud deploy . --project <project-id> --no-env
hud sync tasks my-research-tasks tasks.py --yes
```

The image starts Xvfb, XFCE, Chromium, x11vnc and the HUD server with Supervisor.
The environment checks both the CDP endpoint and the real RFB handshake before
publishing either capability. Browser/desktop processes run as an unprivileged
user with a clean environment. The grader and its credentials remain outside the
agent's isolated workspace.

`tasks.json` contains the three existing LiveWeb examples and is the canonical
task definition. `tasks.py` binds those tasks to the name selected by `hud init`.
Edit a task's `prompt`, `url`, and weighted `rubric_items` for another research
question. The `web-research` template ID and the existing task slugs are stable.

Grading preserves the existing LiveWeb behavior: `patterns` uses case-insensitive
regex matching; otherwise each rubric criterion receives a MET/UNMET verdict,
with the existing rounded weight normalization, penalty scaling and score clamp.
Rubric grading needs `HUD_API_KEY` through the HUD gateway, or
`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` for Anthropic directly. Set
`GRADER_MODEL` to override the existing `claude-opus-4-5` default. Hosted tasks
declare `hud_api_key` so the platform can inject protected grader credentials.
Do not put credentials in task arguments or this source folder.

All three bundled tasks use text rubric grading. Optional `use_visual_grading`
also supplies recent browser screenshots. The shell workspace is tracked for
trace artifacts. Live websites can change, so inspect a fresh run before using
old reference answers in a demo.

This template intentionally uses the live web. Nonempty `archives` is rejected;
WARC replay, custom site substitutions and the old orchestration endpoints
belong to the older runtime and are not included here.

Dependencies target Linux/Python 3.12. To refresh the public dependency lock:

```sh
uv pip compile requirements.in --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements.txt
```
