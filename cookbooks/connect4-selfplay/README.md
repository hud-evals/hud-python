# Connect Four self-play

Symmetric self-play RL on a 6×7 Connect Four board. Draws are rare (you need a
full 42-cell board with no four-in-a-row), so the win/loss reward signal
persists as the policy improves and the GRPO advantage stays non-zero.

## How it works

- One agent ("outer") plays a full game against an inner model on the **same
  slug** — true self-play. `seed % 2` decides who drops first, for symmetric
  first-move coverage.
- Each game trains **both sides at once**: the outer agent's `Run` (reward from
  its perspective) plus a hand-built `TrajectoryPayload` for the inner model
  with the flipped reward (`1 - outer_reward`).
- `group_size=2` pairs each game's two trajectories so the GRPO advantage is
  `reward - 0.5` per game.
- `loss_fn="ppo"` clips the importance-sampling ratio, so a single lucky game
  can't blow up the update.

The training loop uses the public API directly — `forward_backward` accepts
`Run` and `TrajectoryPayload` mixed, so no private helpers are needed.

## Setup

```bash
hud models fork Qwen/Qwen3.5-4B --name c4-selfplay   # prints a slug like c4-selfplay-<id>
```

Put your `HUD_API_KEY` in a `.env` here (or the environment).

## Run

Local sanity check (one game, cheap external model as the outer agent):

```bash
hud eval env.py claude --model claude-haiku-4-5
```

Train:

```bash
python train.py --model c4-selfplay-<id> --steps 20 --group 4 --lr 1e-5
```

## Tuning notes

- **Memory scales with `tasks × group`.** Each task×rollout is a fresh `env.py`
  subprocess. With 8 tasks and `--group 4` that's 32 concurrent games. Connect
  Four games can run up to 42 plies, so they cost more tokens and time per game —
  start at `--group 4` and raise only if you have RAM headroom.
- **Watch the server-side metrics.** The loop prints local win/draw/loss counts
  each step and the last few checkpoints' `mean_reward` / `reward_std` via
  `trainer.checkpoints()` at the end. A healthy run keeps non-trivial
  `reward_std` (within-group spread); if it collapses, the policy has saturated.
- **Reset on changes.** If you edit the reward or the board, roll the head back
  to a clean checkpoint (`hud models head <slug> --set <id>`) or fork fresh —
  don't keep training a policy shaped by the old objective.
