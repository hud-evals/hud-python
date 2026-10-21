# Browser Use

One Chromium desktop serves visual browser tasks and LiveWeb research. Agents can
use CDP, the RFB desktop, an isolated shell workspace, and MCP screenshot/navigation
tools against the same browser context.

`tasks.json` is the canonical task source for evaluation and public taskset sync;
`tasks.py` binds all four tasks to the local environment. The existing
`rental-kitchen-chairs`, `diamond-rose-seats`, `epa-air-quality`, and
`ladybird-spot-num` slugs and template IDs are preserved.

## Run the examples

```sh
hud init my-research --template browser-use
cd my-research
hud deploy . --project <project-id> --no-env
hud sync tasks my-research-tasks tasks.py --yes
```

The image starts Xvfb, XFCE, Chromium, x11vnc and HUD with Supervisor. Browser and
desktop processes run as `pwuser` with a clean environment. The environment checks
both CDP and the RFB handshake before advertising capabilities. Grader credentials
stay outside the agent's isolated workspace.

Builds install an exact SDK wheel when supplied as `.hud-sdk/hud-*.whl`; otherwise
they install the SDK from GitHub `main`. To use a local SDK build, run from the SDK
repository root:

```sh
uv build --wheel --out-dir environments/browser-use/.hud-sdk
```

For local Python development, install the dependency lock and then either the
supplied wheel or current SDK source:

```sh
uv venv --python 3.12
uv pip install --require-hashes -r requirements.txt
uv pip install .hud-sdk/hud-*.whl
# Without a local wheel:
uv pip install "hud[agents] @ git+https://github.com/hud-evals/hud-python.git@main"
```

A local environment process connects to an existing desktop at CDP port 9222 and
RFB port 5900; the container starts those services for you.

The browser/grader dependency lock excludes HUD so it cannot select an older SDK
release. Refresh it from this directory with:

```sh
uv pip compile requirements.in --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements.txt
```

## Adapt the tasks

The `rental-kitchen-chairs` task grades the precise chair count and recorded
kitchen-photo evidence. Its MCP navigation tool stays on the official rental
website. `web-research` accepts arbitrary HTTP(S) URLs, questions, and weighted
`rubric_items`; `patterns` instead enables case-insensitive regex grading. Nonempty
`archives` is rejected because this runtime uses live websites.

Rubric grading preserves the existing rounded weight normalization, penalty
scaling and score clamp. It uses `HUD_API_KEY` via the HUD gateway, or
`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` directly. `GRADER_MODEL` overrides the
existing `claude-opus-4-5` default. Hosted research tasks opt into protected grader
credential injection through `hud_api_key`; never store credentials in task data.
Optional `use_visual_grading` supplies recent browser screenshots to the grader.

The bundled websites can change. Inspect a fresh run before presenting a result
or reusing historical reference answers.
