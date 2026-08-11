# Fireworks Serverless RL with HUD

This cookbook trains a LoRA adapter on Fireworks using rewards from a HUD
environment. The division of labor is simple: HUD owns the tasks (defining
them, running rollouts, grading answers) and Fireworks owns the weights
(sampling, gradients, optimizer state, checkpoints, all on their serverless
shared pool). `train.py` is the loop that connects the two.

| File | What it does |
|------|--------------|
| `env.py` | A small verifiable environment: ask for `a × b`, reward 1.0 if the final integer is right |
| `train.py` | The training loop: rollouts, group-relative advantages, importance-sampling updates |
| `test_train.py` | Unit tests for the advantage and batch logic, no API key needed |

## Running it

You need a `FIREWORKS_API_KEY` with Serverless Training enabled, either in
your environment or in a `.env` file in this directory. Then:

```bash
uv sync
uv run train.py
```

The defaults run the full recipe: 30 optimizer steps, 8 task groups per
step, 8 rollouts per group, 1,024 generated tokens per rollout. On the
default 9B model this takes a couple of hours and a few dollars, and both
grow with model size, token budgets, and how busy the shared pool is.

Don't start there though. Run one small step first. It exercises the whole
path (auth, sampling, HUD grading, one paid update, checkpointing, eval) and
costs a few cents:

```bash
uv run train.py --steps 1 --tasks-per-step 2 --group-size 4 --max-tokens 256 --eval-tasks 4
```

Metrics are appended to `runs/fireworks-serverless/metrics.jsonl` as JSON
lines. `uv run pytest` runs the unit tests without touching the API.

The default model is `accounts/fireworks/models/qwen3p5-9b`. If you want a
different serverless-enabled model, pass `--base-model`, `--tokenizer-model`,
and `--renderer` together, since prompts are rendered and tokenized
client-side and have to match the model's chat template. One gotcha we hit
while building this: the default renderer disables the model's thinking mode
on purpose. With thinking on, the model spends its entire token budget
reasoning, never reaches the final answer line, and every reward comes back
zero.

On concurrency: the pool is shared. The SDK retries transient throttling,
but a big burst of requests still makes your steps slower, so keep
`--max-concurrent` modest.

## Calibrate before you train

GRPO does not train on absolute rewards. It trains on the reward spread
*within* each group of rollouts: advantages are normalized per group, so a
group where every rollout scores the same (all 0 or all 1) contributes
exactly zero gradient, even if the overall mean looks healthy. A task that
is too easy or too hard for your model will happily burn money while
teaching it nothing.

So before a full run, check the spread on the untrained adapter:

```bash
uv run train.py --calibrate --tasks-per-step 6 --group-size 6 --debug-samples 4
```

This rolls out one batch, reports `reward_mean` and
`within_group_reward_std`, and exits without training. `--debug-samples N`
prints the first N rollouts (reward, output tokens, text) so you can see
*why* a group scored the way it did. During training the same number shows
up as `reward_std_within_group` in every step's metrics.

If every group comes back all-correct, make the task harder with
`--min-a/--max-a/--min-b/--max-b`. If everything is wrong, make it easier,
or raise `--max-tokens`; very often the model is running out of budget
mid-working rather than failing the math. The default 3-digit by 3-digit
range sits in the middle for the 9B model: right often but not always,
which is exactly the regime RL needs.

## How the loop works

Each step samples from a snapshot of the current adapter, lets HUD grade the
rollouts, and pushes one update:

```python
snapshot = training_client.save_weights_for_sampler(f"policy-{step}").result()
sampler = service.create_sampling_client(model_path=snapshot.path, tokenizer=tokenizer)

job = await taskset.run(agent, runtime=LocalRuntime("env.py"), group=8)

datums, kept_groups = make_training_batch(job.runs)
training_client.forward_backward(datums, "importance_sampling").result()
training_client.optim_step(adam).result()
```

HUD runs and grades each rollout. The agent samples from the in-session
Fireworks snapshot and records token ids and sampling logprobs on the run,
so the update trains exactly the tokens the policy produced. Rewards are
normalized within each task's group of rollouts, and flat groups are
skipped. If `kept_groups` sits at zero, go back to calibration before paying
for a longer run.

## Loss functions

`forward_backward` takes the loss as a string. The serverless trainer
supports `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, and `dro`.
This loop exposes the three ratio-based RL losses, which all consume the
same datums (target tokens, sampling logprobs, per-token advantages):

```bash
uv run train.py --loss-fn ppo   # or cispo; importance_sampling is the default
```

The same client covers a few things this cookbook doesn't use:
`cross_entropy` over labeled datums for SFT, DPO over chosen/rejected pairs
(see the [Fireworks docs](https://docs.fireworks.ai/fine-tuning/training-api/serverless#what-you-can-run)),
gradient accumulation by calling `forward_backward` several times before one
`optim_step`, and fully custom losses through `forward_backward_custom`,
which hands you per-token logprobs and lets you compute the loss
client-side.

## Checkpoints and resume

The script saves two kinds of checkpoints, and they are not interchangeable.
`policy-*` and `final` are sampler snapshots, the exact adapter weights each
step sampled from. `state-*` and `final-state` carry the adapter plus the
optimizer state, and those are what you resume from (`--checkpoint-every`
controls the cadence):

```bash
uv run train.py --resume-from "<account>/<run-id>/state-0005"
```

A resumed run continues in a fresh Fireworks run: metrics append to the same
`metrics.jsonl` and step numbering restarts at 1. One thing to be careful
about: sampler snapshots are session-scoped, so if you want the final
adapter to stay deployable,
[promote the checkpoint to a model](https://docs.fireworks.ai/fine-tuning/training-api/serverless#saving-and-loading-checkpoints)
before the session goes away.

## Dedicated training

Everything above runs on the serverless shared pool: LoRA only, per-token
billing, nothing to provision. Fireworks also has a dedicated path where you
provision a trainer (and an inference deployment for rollouts) on a chosen
training shape. That is the right tool for full-parameter training, methods
outside the serverless catalog (ORPO, distillation), or sustained load where
reserved capacity beats per-token pricing.

The training client API is the same, so this loop ports over: the trainer
comes from a provisioned shape instead of `create_lora_training_client`, and
rollouts sample from a dedicated deployment that hot-loads each snapshot
instead of an in-session sampler. Start from the `rl_loop.py` and
`async_rl_loop.py` recipes in the
[Fireworks cookbook](https://github.com/fw-ai/cookbook/tree/main/training/recipes),
and see [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)
for shapes, lifecycle, and teardown.

## BFCL env training example

The arithmetic task keeps this cookbook self-contained, but nothing in the
loop depends on it. `env.py` is an ordinary HUD environment, and any
`@env.template()` that yields a prompt and grades the answer works the same
way. For a real example, we ran this same recipe on the
[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) multi-turn suite
packaged as a HUD environment.

The packaging is worth describing because it shows what a HUD environment
can carry. Each task is one BFCL entry. A fresh environment process starts
per rollout, builds the entry's stateful backends (file system, trading bot,
Twitter, and so on) from its initial config, and registers the entry's
functions as native MCP tools with their canonical descriptions. BFCL
scripts several user turns per entry, so the agent works through a turn's
tool calls against the live backends, then calls a `next_turn` tool to pull
the next scripted user message until the entry is done. Grading runs BFCL's
own state and response checkers over the per-turn call strings, the same way
the official harness scores it. The reward is the fraction of leading turns
that pass, which gives GRPO a dense signal; the strict all-turns-pass binary
is reported separately in the grade info so eval numbers stay faithful to
the official metric.

Trained with the same group-relative recipe as this cookbook (groups of 8,
4 tasks per step, learning rate 1e-5, importance-sampling loss), a
Qwen3.5-4B policy went from 0.13 to roughly 0.52 mean reward on the
200-entry `multi_turn_base` taskset in under 20 optimizer steps.

Pointing the loop at a deployed taskset like that one is a two-line change:

```python
taskset = Taskset.from_api("bfcl-multi-turn-base")  # instead of make_taskset()
job = await taskset.run(agent, runtime=HUDRuntime(), group=8)  # env runs on HUD boxes
```

One requirement to be aware of: the agent in this cookbook is one-turn text.
A tool-calling environment like BFCL needs an agent that renders tool
schemas, executes calls, and feeds results back between sampling requests,
while still recording token ids and logprobs each turn. Everything from
`make_training_batch` onward stays the same.

## References

- [Fireworks Serverless Training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- [Fireworks cookbook](https://github.com/fw-ai/cookbook): runnable recipes for
  serverless RL, dedicated RL, SFT, DPO, and distillation
- [HUD tasks and tasksets](https://docs.hud.ai/v6/reference/tasks)
- [HUD task design for training](https://docs.hud.ai/v6/reference/advice)
