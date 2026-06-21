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

What matters for GRPO is **within-group** reward spread: advantages are computed
within each prompt group, so a group whose rollouts all score the same (all 0 or
all 1) produces zero advantage and no gradient — even if the *overall* mean looks
healthy. Calibration reports `within_group_reward_std` for exactly this; treat
it, not `reward_mean`, as the signal that training has something to learn.

Two backends:

- `--calibration-backend inference` (default): Fireworks' OpenAI-compatible API.
  Cheap, but samples `gpt-oss-120b` (`--inference-model`), not the training base —
  the small serverless catalog on the `lorenss` key has no Qwen3 8B. Use it only
  for a rough task sanity check.
- `--calibration-backend managed`: provisions the same deployment sampler that
  training uses and samples the **actual base model** (Qwen3 8B). This is the
  calibration that counts. It still skips the trainer and `optim_step`.

```bash
uv run train.py --calibrate-only --calibration-backend managed \
  --groups-per-step 6 --rollouts-per-prompt 6 --parallelism 18 --debug-samples 4
```

`--debug-samples N` prints the first N rollouts (reward, output-token count,
text) so you can see *why* a group scored the way it did. Tune the multiplication
range until `within_group_reward_std` is clearly above zero:

- Groups all-correct (`within_group_reward_std ~= 0`) → make it harder
  (`--min-a/--max-a/--min-b/--max-b`).
- Groups all-wrong → make it easier, or raise `--max-tokens` so the model can
  finish its working before the budget cuts it off.

The shipped defaults (3-digit × 3-digit, `--max-tokens 512`, thinking disabled)
calibrate to `reward_mean ~= 0.47`, `within_group_reward_std ~= 0.20` on Qwen3 8B:
a regime where the same problem is sometimes solved (when the model shows its
work) and sometimes slipped (when it answers directly) — so RL has a gradient to
follow.

### Reasoning models and the token budget

Qwen3 is a hybrid reasoning model: by default it opens a `<think>` block and, on
a tight `--max-tokens`, spends the whole budget reasoning and never emits the
answer (reward collapses to zero). This cookbook disables thinking by default
through the chat template so direct rollouts reach the integer. Pass
`--enable-thinking` to keep the reasoning block — and raise `--max-tokens`
accordingly so the answer still fits.

## Train

Once calibration has non-trivial rewards:

```bash
uv run train.py --steps 5 --groups-per-step 8 --rollouts-per-prompt 8 --parallelism 32
```

This uses the direct Training API managed service path. If you want calibration
to go through the managed deployment sampler too, pass
`--calibration-backend managed`; this provisions the same resources as training.

### Preview account constraints

On the `lorenss` preview account today:

- **Trainer creation works** end to end with a provisioned key: rollouts,
  `forward_backward_custom`, `optim_step`, checkpoint save, and sampler hotload
  all run, and multi-step training completes. (An earlier `unkey inference api id
  is not configured` 500 on trainer creation was an account-side provisioning gap,
  now resolved.)
- **LoRA is unavailable**: the validated `qwen3-8b-128k` shape only accepts
  full-parameter training, so `--lora-rank > 0` fails at trainer creation with
  `no validated training shape exists for ... trainer_mode=LORA_TRAINER`.
- **Hotloads sync full 8B weights** between steps and occasionally exceed the
  SDK's 600s hotload budget (`RuntimeError: Hotload failed for sampler snapshot
  ...`). This is transient preview-infra latency, not a loop bug — re-running the
  same command generally proceeds. There is no clean knob to extend the timeout
  on the managed sampler path.

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
