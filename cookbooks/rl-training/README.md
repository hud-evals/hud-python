# RL Training

On-policy reinforcement learning with the HUD SDK. The loop is the classic
one: roll out a taskset with the current weights, train on the trajectories
that come back, and let the updated weights serve the next rollout. What
makes it clean here is that all of this happens under one model string.

`hud.TrainingClient` targets a trainable gateway model. Training advances
the weights behind that string in place (the HUD training service
checkpoints and promotes them), so the model you sample with is the same
model you train, and every `optim_step` closes the on-policy loop.

| File | What it does |
|------|--------------|
| `env.py` | A tiny verifiable env: ask for `a + b`, reward 1.0 if correct (quickstart fallback) |
| `common.py` | Resolves the rollout source: a deployed taskset on remote boxes, or the local env |
| `simple_train.py` | The loop with a built-in server-side loss (`importance_sampling`) |
| `ppo_custom_loss.py` | The loop with a client-side custom loss (GLM-5.2 double-sided IS) |

## Run

You need a `HUD_API_KEY`, either in your environment or in a `.env` file.
List the gateway models on your account, pick one marked trainable, and set
it as the `MODEL` constant at the top of `simple_train.py` or
`ppo_custom_loss.py`:

```bash
hud models list          # Name | Model (API) | ID | Provider | Agent | Trainable
```

The real flow is training on a deployed taskset. You have built a taskset
and pushed it with `hud deploy` and `hud sync`, and now you want to train on
it. Set the `TASKSET` constant in `common.py` to its name or id, and the
rollouts run on remote HUD boxes with nothing executing locally:

```bash
uv run simple_train.py --steps 10
uv run ppo_custom_loss.py --steps 10
```

If you just want to see the loop move, leave `TASKSET` empty and a tiny
local arithmetic taskset runs against the bundled `env.py`:

```bash
uv run simple_train.py --steps 10
```

The swap between the two lives in `common.py`'s `load_taskset_and_runtime()`:
`Taskset.from_api(name)` with `HUDRuntime()` for the deployed case, and
`Taskset(...)` with `LocalRuntime("env.py")` for the local one. The training
code does not change either way.

## The loop

Both scripts are the same five lines, and the only difference between them
is the training call:

```python
taskset, runtime = load_taskset_and_runtime()   # deployed+remote, or local
session = await Job.start("rl", group=8)         # one job spans the session
for step in range(steps):
    start = len(session.runs)
    await taskset.run(agent, runtime=runtime, job=session)   # roll out current weights
    batch = session.runs[start:]                             # this step's runs
    await trainer.step(batch, learning_rate=1e-5, group_size=8)   # train + promote
```

The loop only ever touches `job.runs`, so it does not care whether the
rollouts executed on a leased remote box or on your laptop. Passing the
`Run` is enough in both cases, but what travels with it differs:

- Remote (`HUDRuntime`) runs fold back only the reward and a `trace_id`. The
  full token-level trajectory lives on the platform, collected server-side
  during the rollout, and the training service resolves the trajectory and
  reward from the `trace_id` the client sends.
- Local (`LocalRuntime`) runs carry the token-level `Sample` on each agent
  turn in `run.trace`, so the client sends the trajectory inline. This works
  even with telemetry off.

You can also pass `trace_id` strings directly, and mix them with `Run`s.

## Two loss tiers

`simple_train.py` uses a built-in loss. `trainer.step(...)` is one
`forward_backward` with a server-side loss followed by one `optim_step`, and
the client stays dependency-light (no torch). The `loss_fn` menu mirrors
Tinker's native set: `cross_entropy` for supervised data, then
`importance_sampling`, `ppo`, `cispo`, and `dro` for RL. The policy-gradient
ones compute advantages from rewards server-side, GRPO-style over each
`group_size` chunk.

`ppo_custom_loss.py` writes the loss by hand.
`trainer.forward_backward_custom(batch, loss_fn)` splits the step so the
loss math runs on your machine:

1. The service runs the current-policy forward pass and returns per-token
   tensors (`DatumTensors`: current-policy logprobs, rollout logprobs, the
   action mask, reward, group index).
2. Your `loss_fn` builds a differentiable loss over those logprobs, in torch
   here.
3. The service applies the resulting per-token gradients on the backward
   pass.

This mirrors Tinker's `forward_backward_custom` and its
`weights = -dC/dlogprobs` convention, split across the service boundary. One
thing to watch: build the loss out of the logprob tensors the service hands
you rather than re-wrapping them from `.data`, or gradients will not flow.

## What this supports (and what it doesn't)

The custom path covers token-level methods whose only moving part is the
advantage and loss math over per-token tensors:

- The worked example is GLM-5.2's direct double-sided importance sampling:
  reuse the rollout logprobs as the behavior proxy, form the ratio
  `r = exp(logπ_θ − logπ_rollout)`, hard-mask tokens outside
  `[1 − ε_l, 1 + ε_h]`, and normalize at the token level.
- Compaction comes for free. A rollout is a variable-length list of
  variable-length turns, and training puts no constraint on how many turns a
  trajectory has or how long each one is, since every turn's `Sample` is a
  trainable unit.
- Critic-free credit assignment (TEMPO-style tree TD, MemPO per-segment
  advantages, broadcast-advantage with a token-level loss) is all advantage
  math you can write inside `loss_fn`.

The one thing the Tinker backend cannot do natively is train a value
network, because its loss API is over logprobs rather than a value head.
GLM-5.2's critic exists only to produce token-level advantages, and
advantages are an input here, so true critic-PPO means hosting a decoupled
critic in the training service (a value model with GAE feeding the
`advantages` input, where dependencies beyond `tinker`, like a small value
model, are expected) rather than on Tinker. The examples here use a
critic-free group baseline as the stand-in.
