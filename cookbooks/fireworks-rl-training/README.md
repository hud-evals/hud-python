# Fireworks Serverless RL with HUD

On-policy reinforcement learning where HUD owns the tasks and Fireworks owns
the weights: roll out a HUD taskset against the current LoRA adapter, train on
the graded trajectories through the Fireworks Serverless Training API, and
sample the next step from the updated weights.

| File | What it does |
|------|--------------|
| `env.py` | A tiny verifiable env: ask for `a × b`, reward 1.0 if the final integer is right |
| `train.py` | The loop: rollouts → group-relative advantages → importance-sampling update |
| `test_train.py` | Unit tests for the advantage and batch logic (no API key needed) |

## Run

Needs `FIREWORKS_API_KEY` with Serverless Training enabled, from your
environment or a `.env` file here:

```bash
uv sync
uv run train.py
```

The defaults are the full recipe: 30 optimizer steps, 8 task groups per step,
8 rollouts per group, 1,024 generated tokens per rollout. With the default 9B
model expect a couple of hours and a few dollars, scaling with the model,
token budgets, and shared-pool load.

To sanity check the setup first, run one small step. It exercises auth,
sampling, HUD grading, one paid update, checkpointing, and eval:

```bash
uv run train.py --steps 1 --tasks-per-step 2 --group-size 4 --max-tokens 256 --eval-tasks 4
```

Metrics append to `runs/fireworks-serverless/metrics.jsonl` as JSON lines.
`uv run pytest` runs the unit tests offline.

**Models.** The default is `accounts/fireworks/models/qwen3p5-9b`. To train
another serverless-enabled model, pass `--base-model`, `--tokenizer-model`,
and `--renderer` together. Prompts are rendered and tokenized client-side
and must match the model's chat template. The default renderer disables the
model's thinking mode: with it on, rollouts spend the whole token budget
reasoning and never emit the graded final line.

**Concurrency.** Capacity is a shared pool. The SDK retries transient
throttling, but a large burst still makes steps slower, so keep
`--max-concurrent` conservative.

## Calibrate task difficulty first

What GRPO trains on is within-group reward spread: advantages are computed
within each group, so a group whose rollouts all score the same (all 0 or
all 1) produces zero gradient, even when the overall mean looks healthy.
Check the spread on the untrained adapter before paying for a full run:

```bash
uv run train.py --calibrate --tasks-per-step 6 --group-size 6 --debug-samples 4
```

This rolls out one batch, reports `reward_mean` and
`within_group_reward_std`, and exits without training. `--debug-samples N`
prints the first N rollouts (reward, output tokens, text) so you can see why
a group scored the way it did. During training the same number is logged as
`reward_std_within_group` in each step's metrics.

Tune the multiplication ranges until the spread is clearly above zero:

- Groups all correct: make the task harder
  (`--min-a/--max-a/--min-b/--max-b`).
- Groups all wrong: make it easier, or raise `--max-tokens` so the answer
  still fits after the model's working.

The default 3-digit by 3-digit range sits mid-difficulty for the 9B model:
right often but not always, which is the regime RL needs.

## The loop

Each step samples from a snapshot of the current adapter, lets HUD grade the
rollouts, and sends one update:

```python
snapshot = training_client.save_weights_for_sampler(f"policy-{step}").result()
sampler = service.create_sampling_client(model_path=snapshot.path, tokenizer=tokenizer)

job = await taskset.run(agent, runtime=LocalRuntime("env.py"), group=8)

datums, kept_groups = make_training_batch(job.runs)
training_client.forward_backward(datums, "importance_sampling").result()
training_client.optim_step(adam).result()
```

HUD runs and grades each rollout. The agent samples from the in-session
Fireworks snapshot and records token ids and sampling logprobs on the run, so
the update trains exactly the tokens the policy produced. Rewards are
normalized within each group of rollouts of the same task (group-relative
advantages), and a group where every rollout earns the same reward carries no
signal, so it is skipped. If `kept_groups` sits at zero, make the task harder
or the group larger before paying for a long run.

## Loss functions

`forward_backward` takes the loss as a string. The serverless trainer
supports `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, and `dro`.
This loop exposes the three ratio-based RL losses, which consume the same
datums (target tokens, sampling logprobs, per-token advantages):

```bash
uv run train.py --loss-fn ppo   # or cispo; importance_sampling is the default
```

The same client also covers:

- Supervised and preference training: `cross_entropy` over labeled datums
  for SFT, and DPO over chosen/rejected pairs, run on the same serverless
  session (see the [Fireworks docs](https://docs.fireworks.ai/fine-tuning/training-api/serverless#what-you-can-run)).
- Gradient accumulation: call `forward_backward` several times before one
  `optim_step`.
- Custom losses: `forward_backward_custom` splits the step so you compute
  the loss client-side from per-token logprobs.

## Checkpoints and resume

Two kinds of checkpoints:

- `policy-*` and `final` are sampler snapshots, the exact adapter weights
  each step sampled from.
- `state-*` and `final-state` carry adapter plus optimizer state, for
  resuming (`--checkpoint-every` controls the interval).

```bash
uv run train.py --resume-from "<account>/<run-id>/state-0005"
```

A resumed run continues in a fresh Fireworks run: metrics append to the same
`metrics.jsonl` and step numbering restarts at 1. Sampler snapshots are
session-scoped. To keep the final adapter deployable after the session is
gone, [promote the checkpoint to a model](https://docs.fireworks.ai/fine-tuning/training-api/serverless#saving-and-loading-checkpoints)
first.

## Dedicated training

Everything above runs on the serverless shared pool: LoRA only, per-token
billing, nothing to provision. Fireworks also has a dedicated path where you
provision a trainer (and an inference deployment for rollouts) on a chosen
training shape. Use it for full-parameter training, models or methods not in
the serverless catalog (ORPO, distillation), or sustained load where
reserved capacity beats per-token pricing.

The training client API is the same, so this loop ports over: the trainer
comes from a provisioned training shape instead of
`create_lora_training_client`, and rollouts sample from a dedicated
deployment that hot-loads each snapshot instead of an in-session sampler.
Start from the `rl_loop.py` and `async_rl_loop.py` recipes in the
[Fireworks cookbook](https://github.com/fw-ai/cookbook/tree/main/training/recipes),
and see [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)
for shapes, lifecycle, and teardown.

## BFCL env training example

`env.py` is an ordinary HUD environment, and any `@env.template()` that
yields a prompt and grades the answer slots straight in. For a real example,
we ran the same recipe on the
[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) multi-turn suite
packaged as a HUD environment.

How that env works:

- Each task is one BFCL entry. A fresh env process starts per rollout, builds
  the entry's stateful backends (file system, trading bot, Twitter, ...) from
  its initial config, and registers the entry's involved functions as native
  MCP tools with their canonical descriptions.
- BFCL scripts several user turns per entry. The agent works through a turn's
  tool calls against the live backends, then calls a `next_turn` tool to pull
  the next scripted user message, until the entry is done.
- Grading runs BFCL's own state and response checkers over the rollout's
  per-turn call strings, the same way the official harness scores it.
- The reward is the fraction of leading turns that pass, which gives GRPO a
  dense signal. The strict benchmark binary (all turns pass) is reported
  separately in the grade info, so eval numbers stay faithful to the
  official metric.

Trained with the same group-relative recipe as this cookbook (groups of 8,
4 tasks per step, learning rate 1e-5, importance-sampling loss), a
Qwen3.5-4B policy went from 0.13 to ~0.52 mean reward on the 200-entry
`multi_turn_base` taskset in under 20 optimizer steps.

Pointing the loop at a deployed taskset like that one is a two-line change:

```python
taskset = Taskset.from_api("bfcl-multi-turn-base")  # instead of make_taskset()
job = await taskset.run(agent, runtime=HUDRuntime(), group=8)  # env runs on HUD boxes
```

One requirement: this cookbook's agent is one-turn text. A tool-calling env
like BFCL needs an agent that renders tool schemas, executes calls, and feeds
results back between sampling requests, while still recording token ids and
logprobs each turn. Everything from `make_training_batch` on is unchanged.

## References

- [Fireworks Serverless Training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- [Fireworks cookbook](https://github.com/fw-ai/cookbook): runnable recipes for
  serverless RL, dedicated RL, SFT, DPO, and distillation
- [HUD tasks and tasksets](https://docs.hud.ai/v6/reference/tasks)
- [HUD task design for training](https://docs.hud.ai/v6/reference/advice)
