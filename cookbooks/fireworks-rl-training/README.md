# Fireworks RL Training

Direct Fireworks Training API loop over the same arithmetic preview task used by
`cookbooks/rl-training`.

This does **not** use Fireworks native datasets or RFT jobs. It follows the
Training API service path from the Fireworks docs:

1. `FiretitanServiceClient.from_firetitan_config(...)`
2. `create_deployment_sampler(...)` for high-parallel rollouts
3. local grading of HUD-style multiplication tasks
4. `forward_backward_custom(...)` + `optim_step(...)`
5. `save_weights_for_sampler(...)` + sampler refresh

References:

- Fireworks Training API introduction: https://docs.fireworks.ai/fine-tuning/training-api/introduction
- Training and sampling lifecycle: https://docs.fireworks.ai/fine-tuning/training-api/training-and-sampling
- Loss functions / GRPO reference: https://docs.fireworks.ai/fine-tuning/training-api/loss-functions

## Setup

The repo-level `.env` is loaded automatically. It must contain:

```bash
FIREWORKS_API_KEY=...
FIREWORKS_ACCOUNT_ID=...
```

Install the isolated cookbook environment:

```bash
uv sync --pre
```

## Calibrate task difficulty first

Calibration defaults to Fireworks' OpenAI-compatible inference API, so it does
**not** create a trainer, provision a Training API deployment, or call
`optim_step`. This is the cheap way to tune task difficulty before paying for a
Training API run.

The calibration model is separate from the training base model because the
`lorenss` key currently exposes only a small serverless inference catalog (no
Qwen3 8B deployment). Override it with `--inference-model` if you have a closer
deployed model.

```bash
uv run train.py --calibrate-only --groups-per-step 8 --rollouts-per-prompt 8 --parallelism 32
```

The goal is a reward distribution with variance. If reward is all zero, make the
task easier:

```bash
uv run train.py --calibrate-only --min-a 10 --max-a 99 --min-b 2 --max-b 9
```

If reward is all one, make the task harder:

```bash
uv run train.py --calibrate-only --min-a 1000 --max-a 9999 --min-b 11 --max-b 99
```

The current defaults are calibrated for the visible `gpt-oss-120b` inference
model on the `lorenss` key: 2-digit by 1-digit multiplication with a direct
"reply only with the integer" prompt. A 32-rollout calibration gave a non-trivial
baseline (`reward_mean ~= 0.22`, `reward_std ~= 0.42`), while the original
3-digit by 2-digit range was all-zero.

## Train

Once calibration has non-trivial rewards:

```bash
uv run train.py --steps 5 --groups-per-step 8 --rollouts-per-prompt 8 --parallelism 32
```

This uses the direct Training API managed service path. If you want calibration
to go through the managed deployment sampler too, pass
`--calibration-backend managed`; this provisions the same resources as training.

### Current Fireworks preview account blocker

On the `lorenss` preview account, trainer creation currently fails before the
first train step with:

```text
failed to ensure FIREWORKS_API_KEY secret: unkey inference api id is not configured
```

This happens even with `create_deployment=False`, so it is an account/control
plane provisioning issue rather than a problem in the rollout or loss code. Once
Fireworks enables the missing Unkey inference API config for the account, the
same `uv run train.py ...` command should proceed to trainer startup and the
first `forward_backward_custom(...)` call.

Metrics are written to:

- `runs/fireworks-rl-preview/metrics.jsonl`
- `runs/fireworks-rl-preview/reward_loss.png` if `matplotlib` is installed

## Notes

- Defaults use Qwen 3 8B full-parameter training:
  - `accounts/fireworks/models/qwen3-8b`
  - `Qwen/Qwen3-8B`
  - `accounts/fireworks/trainingShapes/qwen3-8b-128k`
- LoRA can be tested with `--lora-rank N`, but the validated Qwen3 8B training
  shape currently rejects LoRA mode on the `lorenss` preview account.
- The first checkpoint sync happens after step 0 and subsequent rollouts sample
  the updated weights through the same deployment.
- `--keep-trainer` and `--keep-deployment` are available for debugging. By
  default the trainer is cleaned up and the deployment scales to zero on exit.
