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
| `test_train.py` | Unit tests for batching and the HUD rollout lifecycle; tokenizer compatibility check |

## Setup

Install Git, Python 3.11 or 3.12, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). Run all
commands in this README from `cookbooks/fireworks-rl-training`.

Set `FIREWORKS_API_KEY` in the environment or in a local `.env` file. The
key must have access to Fireworks Serverless Training.

```bash
uv sync
```

The default model is Qwen 3.8 27B (`accounts/fireworks/models/qwen3p8-27b`),
with tokenizer `Qwen/Qwen3.8-27B`. Fireworks deprecated the Qwen 3.5 9B and
Qwen 3.6 27B Serverless Training pools; see the
[Fireworks changelog](https://docs.fireworks.ai/updates/changelog).
A different model requires a matching `--base-model`, `--tokenizer-model`, and
`--renderer`, since tokenization and prompt rendering happen on the client.

The default renderer is `qwen3_8_disable_thinking`, provided by
`tinker-cookbook>=0.5.7`. With thinking enabled and a small generation
budget, the model may spend all available tokens on
reasoning without producing the final answer expected by the grader.

## Run

Calibrate the default task before taking an optimizer step:

```bash
uv run train.py \
  --calibrate \
  --tasks-per-step 6 \
  --group-size 4 \
  --max-tokens 2048 \
  --debug-samples 4
```

If the output shows useful reward spread, run one bounded training step:

```bash
uv run train.py \
  --steps 1 \
  --tasks-per-step 2 \
  --group-size 4 \
  --max-tokens 2048 \
  --eval-tasks 4 \
  --require-update
```

`--require-update` fails if every group has identical rewards and the script
cannot apply an optimizer update. A successful command verifies
authentication, sampling, HUD grading, one update, checkpoint creation, and
evaluation. Calibration, training, and evaluation fail if any rollout has a
sampling or grading error; failed attempts are not treated as zero-reward
training examples.

The default command requests 30 training steps with 8 task groups per step,
8 rollouts per group, and up to 2,048 generated tokens per rollout. A step
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

To check the default renderer against Qwen's published chat template:

```bash
uv run pytest -m integration test_train.py -k default_renderer
```

This downloads the tokenizer from Hugging Face and requires network access,
but does not call Fireworks or train a model.

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
  --group-size 4 \
  --debug-samples 4
```

The command reports:

| Metric | Meaning |
|--------|---------|
| `reward_mean` | Mean reward across completed rollouts |
| `within_group_reward_std` | Mean reward standard deviation within rollout groups |

`--debug-samples N` prints the reward, output-token count, and full response for the
first N rollouts. During training, the within-group statistic is stored as
`reward_std_within_group`. A positive value confirms reward variation in at
least one group. Inspect the sample text to verify that the rewards track
answer quality.

If groups are uniformly correct, increase the task difficulty with
`--min-a`, `--max-a`, `--min-b`, and `--max-b`. If groups are uniformly
incorrect, reduce the operand range or increase `--max-tokens`. The default
four-digit multiplication range uses a 2,048-token budget. Three-digit tasks
were nearly saturated on Qwen 3.8 27B, while smaller generation budgets cut
off many answers. Inspect both correctness and output-token counts when
calibrating; a clipped response does not establish arithmetic difficulty.

## Compare before and after training

Use `--eval-before` to evaluate the initial and final adapters on the same
held-out taskset. Both evaluations use temperature 0 and the same token budget:

```bash
uv run train.py \
  --eval-before \
  --steps 8 \
  --tasks-per-step 16 \
  --group-size 4 \
  --eval-tasks 128 \
  --seed 42 \
  --max-tokens 2048 \
  --max-concurrent 8 \
  --output-dir runs/before-after
```

This requests 512 training rollouts and 256 evaluation rollouts. The 16
training tasks repeat across steps; all 128 evaluation tasks are held out.
`--seed` controls task generation and splitting, not model sampling randomness.

The output directory contains `config.json`, `eval-before.json`, and
`eval-after.json`. Each evaluation records the checkpoint, prompts, full
responses, rewards, grader info, output-token counts, and whether the response
reached the token limit. Compare answers by prompt, including tasks that changed
from correct to incorrect. Report token-limit and format failures alongside
accuracy.

Without `--eval-before`, the script performs only the final evaluation. Use a
separate output directory for each experiment to preserve its artifacts.

One run with the configuration above applied all eight updates:

| Held-out metric | Before | After |
| --- | ---: | ---: |
| Correct answers | 100/128 (78.1%) | 126/128 (98.4%) |
| Invalid final line | 19 | 0 |
| Incorrect valid integer | 9 | 2 |
| Responses at the token limit | 20 | 0 |
| Mean output tokens | 1,085 | 704 |

Accuracy increased by 20.3 percentage points: 27 tasks improved and one
regressed. Twenty improvements came from responses previously at the token
limit. This measures exact-answer success within the fixed budget, with shorter
completed responses contributing to the gain. It is one run on this arithmetic
distribution; repeat across training runs to assess reproducibility.

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

Resume restores the checkpoint's original base model; it does not migrate
an older Qwen adapter to Qwen 3.8. Start a new run when changing base models.
For a supported non-default checkpoint, pass its matching tokenizer and renderer.

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
