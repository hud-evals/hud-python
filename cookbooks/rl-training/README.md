# RL Training

This cookbook shows how to train a HUD gateway model on trajectories from a
HUD taskset. Each iteration collects grouped rollouts, grades them in the
environment, and updates the model behind the same gateway identifier.
Subsequent rollouts therefore use the latest weights.

Two implementations are included. `simple_train.py` uses a built-in
server-side loss, while `ppo_custom_loss.py` defines the policy-gradient
loss in PyTorch and sends the resulting per-token gradients to the training
service.

| File | Purpose |
|------|---------|
| `env.py` | Local arithmetic environment used by the example |
| `common.py` | Selects a deployed taskset or the local environment |
| `simple_train.py` | On-policy training with a built-in loss |
| `ppo_custom_loss.py` | On-policy training with a custom double-sided importance-sampling loss |

## Setup

Set `HUD_API_KEY` in the environment or in a local `.env` file. Then list
the gateway models available to the account:

```bash
hud models list
```

Choose a model marked **Trainable** and assign its identifier to `MODEL` in
the training script.

## Run

The default configuration uses the arithmetic taskset in `env.py` and runs
it through `LocalRuntime`:

```bash
uv run simple_train.py --steps 10
```

To train on a deployed taskset, set `TASKSET` in `common.py` to the taskset
name or id. `load_taskset_and_runtime()` will load it with
`Taskset.from_api(...)` and execute rollouts through `HUDRuntime`. The
training command remains the same:

```bash
uv run simple_train.py --steps 10
```

The custom-loss example uses the same task and rollout configuration:

```bash
uv run ppo_custom_loss.py --steps 10
```

Both scripts accept `--group`, `--learning-rate`, and `--max-concurrent`.
The defaults are a group size of 8, a learning rate of `1e-5`, and at most
8 concurrent rollouts.

## Training flow

A `Job` spans the training session and accumulates its runs. Each iteration
selects the runs added by the latest rollout and trains on that batch:

```python
batch_start = len(session.runs)
await taskset.run(agent, runtime=runtime, job=session)
batch = session.runs[batch_start:]

await trainer.step(batch, learning_rate=1e-5, group_size=8)
```

`trainer.step(...)` performs `forward_backward` followed by `optim_step`.
The optimizer step checkpoints and promotes the updated weights behind the
gateway model, so the next call to `taskset.run(...)` samples the new policy.

The trajectory representation depends on the runtime:

| Runtime | Data sent to training |
|---------|-----------------------|
| `HUDRuntime` | Reward and `trace_id`; the training service resolves the token-level trajectory stored by the platform |
| `LocalRuntime` | Reward and the token-level `Sample` recorded on each agent turn in `run.trace` |

`TrainingClient` also accepts `trace_id` strings directly. A training batch
may contain `Run` objects, trace ids, or both.

## Built-in losses

`simple_train.py` calls `forward_backward` with
`loss_fn="importance_sampling"`. The available server-side losses are:

| Loss | Use |
|------|-----|
| `cross_entropy` | Supervised training |
| `importance_sampling` | Group-relative policy-gradient training |
| `ppo` | PPO objective |
| `cispo` | CISPO objective |
| `dro` | Distributionally robust objective |

For the policy-gradient losses, the service computes group-relative
advantages from rewards using each consecutive `group_size` set of
trajectories. The built-in path does not require PyTorch on the client.

## Custom loss

`ppo_custom_loss.py` uses `forward_backward_custom` to implement GLM-5.2
direct double-sided importance sampling. The computation is split across
the client and training service:

1. The service runs the current-policy forward pass and returns
   `DatumTensors`, including policy logprobs, rollout logprobs, action masks,
   rewards, and group indices.
2. The client computes a differentiable PyTorch loss from the policy
   logprobs.
3. The service applies the per-token gradients, after which `optim_step`
   updates and promotes the model.

The example uses the rollout logprobs as the behavior-policy proxy, computes
`r = exp(logπ_θ - logπ_rollout)`, masks tokens outside
`[1 - ε_l, 1 + ε_h]`, and normalizes by the number of trained tokens.
The loss must use the policy logprob tensors returned by the service.
Constructing new tensors from `.data` disconnects the computation graph.

### Scope

The custom API supports objectives defined from per-token logprobs, masks,
rewards, group membership, and externally supplied advantages. It supports
variable-length, multi-turn trajectories because each turn's `Sample` is an
independent training datum. Critic-free methods such as grouped baselines,
tree-based credit assignment, and per-segment advantages can be expressed
in the client loss.

The backend does not train a value head. A critic-based PPO implementation
must run the value model separately and pass its token-level advantages into
the policy loss. The included example uses a group-mean baseline instead.
