# RL Training

On-policy reinforcement learning with the HUD SDK: roll out a taskset with the
current weights, train on the resulting trajectories, and let the updated weights
serve the next rollout — all under one model string.

`hud.TrainingClient` targets one **trainable gateway model**. Training advances
the weights behind that string in place (the HUD training service checkpoints and
promotes them), so the *same* `model` you sample with is the one you train, and
each `optim_step` closes the on-policy loop.

| File | What it does |
|------|--------------|
| `env.py` | A tiny verifiable env: ask for `a + b`, reward 1.0 if correct (quickstart fallback) |
| `common.py` | Resolves the rollout source: a deployed taskset on remote boxes, or the local env |
| `simple_train.py` | The loop with a built-in server-side loss (`importance_sampling`) |
| `ppo_custom_loss.py` | The loop with a client-side custom loss (GLM-5.2 double-sided IS) |

## Run

Needs `HUD_API_KEY` and `HUD_MODEL` (a trainable gateway model).

**Train on a deployed taskset (the real flow).** You've built a taskset and
pushed it (`hud deploy` + `hud sync`); now train on it. Point `HUD_TASKSET` at it
and rollouts run on **remote HUD boxes** — nothing local:

```bash
HUD_MODEL=<trainable-model> HUD_TASKSET=<taskset-name-or-id> uv run simple_train.py --steps 10
HUD_MODEL=<trainable-model> HUD_TASKSET=<taskset-name-or-id> uv run ppo_custom_loss.py --steps 10
```

**Quickstart (self-contained).** Leave `HUD_TASKSET` unset and a tiny local
arithmetic taskset runs against the bundled `env.py`:

```bash
HUD_MODEL=<trainable-model> uv run simple_train.py --steps 10
```

The swap is `common.py`'s `load_taskset_and_runtime()` — `Taskset.from_api(name)`
+ `HUDRuntime()` for the deployed case, `Taskset(...)` + `LocalRuntime("env.py")`
for the local one. **The training code is identical either way.**

## The loop

Both scripts are the same five lines — the only difference is the training call:

```python
taskset, runtime = load_taskset_and_runtime()   # deployed+remote, or local
session = await Job.start("rl", group=8)         # one job spans the session
for step in range(steps):
    start = len(session.runs)
    await taskset.run(agent, runtime=runtime, job=session)   # roll out current weights
    batch = session.runs[start:]                             # this step's runs
    await trainer.step(batch, learning_rate=1e-5, group_size=8)   # train + promote
```

The loop only ever touches `job.runs`, so where the rollouts executed — a remote
leased box or your laptop — is irrelevant to training. Passing the `Run` is
enough either way:

- **Remote (`HUDRuntime`)** runs fold back only reward + `trace_id`; their full
  token-level trajectory lives on the platform (collected server-side during the
  rollout). The client sends the `trace_id` and the training service resolves the
  trajectory + reward from it.
- **Local (`LocalRuntime`)** runs carry the token-level `Sample` on each agent
  turn in `run.trace`, so the client sends the trajectory inline (works even with
  telemetry off).

You can also pass `trace_id` strings directly, and mix them with `Run`s.

## Two loss tiers

**Built-in (`simple_train.py`).** `trainer.step(...)` = one `forward_backward`
with a server-side loss, then one `optim_step`. The client stays dependency-light
(no torch). `loss_fn` mirrors Tinker's native set — `cross_entropy` (supervised),
`importance_sampling`, `ppo`, `cispo`, `dro`; the policy-gradient ones compute
advantages from rewards server-side (GRPO over each `group_size` chunk).

**Custom (`ppo_custom_loss.py`).** `trainer.forward_backward_custom(batch, loss_fn)`
splits the step so *you* write the loss:

1. `forward` (service) runs the current-policy pass and returns per-token tensors
   (`DatumTensors`: current-policy logprobs π_θ, rollout logprobs q, action mask,
   reward, group index).
2. your `loss_fn` builds a differentiable loss over the π_θ logprobs (torch, here).
3. `backward` (service) applies the resulting per-token gradients.

This mirrors Tinker's `forward_backward_custom` and its `weights = -dC/dlogprobs`
convention, split across the service boundary. Build the loss out of the
**provided** logprob tensors (don't re-wrap from `.data`) or gradients won't flow.

## What this supports (and what it doesn't)

The custom path expresses token-level methods whose only moving part is the
advantage / loss math over per-token tensors:

- **GLM-5.2 direct double-sided IS** (the worked example): reuse rollout logprobs
  as the behavior proxy, ratio `r = exp(logπ_θ − logπ_rollout)`, hard-mask tokens
  outside `[1 − ε_l, 1 + ε_h]`, token-level normalization.
- **Compaction** is free: a rollout is a variable-length list of variable-length
  turns, and training has no constraint on how many turns a trajectory has or
  their relative lengths — every turn's `Sample` is a trainable unit.
- Critic-free credit assignment (TEMPO-style tree-TD, MemPO per-segment,
  broadcast-advantage + token-level loss) is all advantage math you can write in
  `loss_fn`.

The one thing the Tinker backend cannot do natively is **train a value network**
(its loss API is over logprobs, not a value head). GLM-5.2's critic exists only to
produce token-level advantages, and advantages are an input — so for true
critic-PPO you host a decoupled critic in the training service (**Option A**:
value model + GAE, fed as the `advantages` input; deps beyond `tinker` such as a
small value model are expected there) rather than on Tinker. The examples here use
a critic-free group baseline as the stand-in.
