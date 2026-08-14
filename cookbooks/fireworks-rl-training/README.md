# Fireworks Serverless RL with HUD

This cookbook trains a Fireworks LoRA adapter on trajectories graded by a
HUD environment. HUD defines the task, executes each rollout, and returns
the reward. Fireworks samples from the current adapter and performs the
forward, backward, optimizer, and checkpoint operations on its serverless
training service.

| File | Purpose |
|------|---------|
| `env.py` | Local multiplication environment and grader |
| `train.py` | Rollout, advantage, training, evaluation, and checkpoint loop |
| `test_train.py` | Unit tests for URL handling, reward normalization, batch construction, and reward spread |

## Setup

Install Git, Python 3.11 or 3.12, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). Run all
commands in this README from `cookbooks/fireworks-rl-training`.

Set `FIREWORKS_API_KEY` in the environment or in a local `.env` file. The
key must have access to Fireworks Serverless Training.

```bash
uv sync
```

The default model is `accounts/fireworks/models/qwen3p5-9b`. A different
model requires a matching `--base-model`, `--tokenizer-model`, and
`--renderer`, since tokenization and prompt rendering happen on the client.

The default renderer disables thinking mode. With thinking enabled and a
small generation budget, the model may spend all available tokens on
reasoning without producing the final answer expected by the grader.

## Run

Calibrate the default task before taking an optimizer step:

```bash
uv run train.py \
  --calibrate \
  --tasks-per-step 6 \
  --group-size 6 \
  --max-tokens 512 \
  --debug-samples 4
```

If the output shows useful reward spread, run one bounded training step:

```bash
uv run train.py \
  --steps 1 \
  --tasks-per-step 2 \
  --group-size 4 \
  --max-tokens 512 \
  --eval-tasks 4 \
  --require-update
```

`--require-update` fails if every group has identical rewards and the script
cannot apply an optimizer update. A successful command verifies
authentication, sampling, HUD grading, one update, checkpoint creation, and
evaluation.

The default command requests 30 training steps with 8 task groups per step,
8 rollouts per group, and up to 1,024 generated tokens per rollout. A step
with no reward variation skips its optimizer update. The run still collects
1,920 paid training rollouts, followed by 16 evaluation rollouts. Fireworks
meters prompt prefill, sampled output, and training tokens separately.

```bash
uv run train.py
```

See [Serverless Training pricing](https://docs.fireworks.ai/fine-tuning/training-api/serverless#pricing).
Metrics are written to `runs/fireworks-serverless/metrics.jsonl`. The unit
tests do not require an API key:

```bash
uv run pytest
```

Serverless capacity is shared. `--max-concurrent` controls the number of
simultaneous rollout requests; lower values reduce burst pressure when the
pool is busy.

## Task calibration

Group-relative training requires reward variation within each group. If all
rollouts for a task receive the same reward, their normalized advantages
are zero and the group contributes no gradient.

Use calibration mode to sample from the untrained adapter without applying
an optimizer update:

```bash
uv run train.py \
  --calibrate \
  --tasks-per-step 6 \
  --group-size 6 \
  --debug-samples 4
```

The command reports:

| Metric | Meaning |
|--------|---------|
| `reward_mean` | Mean reward across completed rollouts |
| `within_group_reward_std` | Mean reward standard deviation within rollout groups |

`--debug-samples N` prints the reward, output-token count, and text for the
first N rollouts. During training, the within-group statistic is stored as
`reward_std_within_group`. A positive value confirms reward variation in at
least one group. Inspect the sample text to verify that the rewards track
answer quality.

If groups are uniformly correct, increase the task difficulty with
`--min-a`, `--max-a`, `--min-b`, and `--max-b`. If groups are uniformly
incorrect, reduce the operand range or increase `--max-tokens`. The default
three-digit multiplication range is intended to produce both correct and
incorrect samples with the 9B model.

## Training flow

Each training step uses a sampler snapshot of the current adapter:

_Conceptual excerpt from `train.py`; this is not standalone code._

```python
snapshot = training_client.save_weights_for_sampler(f"policy-{step}").result()
sampler = service.create_sampling_client(
    model_path=snapshot.path,
    tokenizer=tokenizer,
)

job = await taskset.run(
    agent,
    runtime=runtime,
    group=8,
)

datums, kept_groups = make_training_batch(job.runs)
if datums:
    training_client.forward_backward(datums, "importance_sampling").result()
    training_client.optim_step(adam).result()
```

`FireworksAgent` records prompt tokens, output tokens, and sampling
logprobs on each HUD run. `make_training_batch` groups runs by task,
normalizes rewards within each group, and converts the trajectories into
training datums. Groups with no reward variation are omitted.

After `optim_step`, the next sampler snapshot contains the updated adapter
weights. This preserves the on-policy sequence of rollout, grade, update,
and resample.

## Loss functions

The `--loss-fn` option selects one of the ratio-based server-side objectives
used by this script:

| Value | Description |
|-------|-------------|
| `importance_sampling` | Importance-sampling policy-gradient loss; default |
| `ppo` | PPO objective |
| `cispo` | CISPO objective |

All three use the same target tokens, rollout logprobs, and per-token
advantages produced by `make_training_batch`.

Fireworks Serverless Training also supports SFT and DPO with their
respective datum formats. `forward_backward_custom` is available for
client-defined losses, and repeated `forward_backward` calls before
`optim_step` provide gradient accumulation. These paths are outside the
scope of this script; see the
[Fireworks Serverless Training documentation](https://docs.fireworks.ai/fine-tuning/training-api/serverless#what-you-can-run).

## Checkpoints and resume

The script writes separate checkpoints for sampling and training:

| Checkpoint | Contents | Use |
|------------|----------|-----|
| `policy-*`, `final` | Adapter weights | Create an in-session sampler or promote the adapter |
| `state-*`, `final-state` | Adapter weights and optimizer state | Resume training |

`--checkpoint-every N` controls the training-state checkpoint interval.
Resume a previous run with a fully qualified training checkpoint:

```bash
uv run train.py --resume-from "<account>/<run-id>/state-0005"
```

Resume creates a new Fireworks run, appends metrics to the existing
`metrics.jsonl`, and restarts local step numbering at 1. Sampler checkpoints
are session-scoped. The script prints the final `Sampler checkpoint` path.
Use it to identify and
[promote the checkpoint](https://docs.fireworks.ai/fine-tuning/training-api/serverless#promote-a-sampler-checkpoint-to-a-model)
before the session is removed.

## Other HUD tasks

For a local one-turn environment, pass both its task source and environment
source:

```bash
uv run train.py \
  --tasks-file "../my-environment/tasks.py" \
  --env-path "../my-environment/env.py" \
  --calibrate
```

For an environment and taskset already deployed to HUD, set `HUD_API_KEY`
and pass its name or id:

```bash
uv run train.py --taskset "my-taskset" --calibrate
```

Calibration needs at least `--tasks-per-step` tasks. A training run needs
`--tasks-per-step + --eval-tasks`; the script creates deterministic,
disjoint training and evaluation subsets. The included `FireworksAgent` and
batch builder support one generated assistant response per HUD Run.
Multi-turn and tool-using environments need a different agent adapter and
batch construction.

## Serverless and dedicated training

This cookbook uses the serverless path. Fireworks also provides dedicated
training for workloads that require full-parameter updates, methods outside
the serverless catalog, reserved capacity, or explicit control over trainer
and inference resources.

| | Serverless | Dedicated |
|---|---|---|
| Training mode | LoRA on a shared pool | LoRA or full-parameter on provisioned resources |
| Rollout serving | In-session sampler snapshot | Inference deployment with snapshot hot-loading |
| Resource lifecycle | Managed by Fireworks | Trainer and deployment are provisioned and released explicitly |
| Typical use | Short or variable workloads billed per token | Sustained training or broader method and model support |

The client-side rollout and datum construction in this cookbook can be
reused with dedicated training, but resource creation and sampler refresh
follow the dedicated lifecycle. Start from `rl_loop.py` or
`async_rl_loop.py` in the
[Fireworks cookbook](https://github.com/fw-ai/cookbook/tree/main/training/recipes)
and refer to the
[Dedicated Training documentation](https://docs.fireworks.ai/fine-tuning/training-api/dedicated).

## BFCL environment example

The local arithmetic environment demonstrates the training interface with a
single-turn text task. The same HUD task model can represent multi-turn,
tool-using workloads. The
[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
multi-turn suite has been packaged as a HUD environment with the following
structure:

| Component | Implementation |
|-----------|----------------|
| Task | One BFCL entry |
| State | Fresh stateful backends for each rollout |
| Tools | Entry-specific backend functions exposed as MCP tools |
| Turns | Scripted user messages retrieved through a `next_turn` tool |
| Grading | BFCL state and response checkers |
| Training reward | Fraction of leading turns that pass |
| Evaluation metric | Strict all-turns-pass result stored separately in grade info |

In a separate run using HUD's managed training service, a Qwen3.5-4B policy
trained on the 200-entry `multi_turn_base` taskset with groups of 8, 4 tasks
per step, a learning rate of `1e-5`, and an importance-sampling loss. Mean
training reward increased from 0.13 to approximately 0.52 in fewer than
20 optimizer steps. These results validate the environment and reward
design; they are not a Fireworks serverless benchmark.

BFCL is not a drop-in input to this cookbook. `--taskset` can select the
hosted taskset and runtime, but the included adapter cannot execute the tool
loop and the batch builder keeps only one assistant turn. See the
[HUD RL training cookbook](https://github.com/hud-evals/hud-python/tree/main/cookbooks/rl-training)
for a runnable multi-turn implementation.

## References

- [Fireworks Serverless Training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- [Fireworks cookbook](https://github.com/fw-ai/cookbook): runnable recipes for
  serverless RL, dedicated RL, SFT, DPO, and distillation
- [HUD tasks and tasksets](https://docs.hud.ai/v6/reference/tasks)
- [HUD task design for training](https://docs.hud.ai/v6/reference/advice)
