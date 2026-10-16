# Robot LIBERO

LIBERO tabletop manipulation with a Franka Panda, on robosuite and MuJoCo. One
environment serves policies and tool-calling agents: the simulator runs behind
a `robot` capability that a VLA policy streams actions to, and in absolute mode
the same environment also serves a `move_to` tool, so any tool-calling LLM or
coding agent can drive the arm with no robot code of its own. Only the
interface changes; the simulator, tasks, and grading do not.

The five LIBERO suites are the templates: `libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`, and `libero_90` (130 tasks, 50 saved initial states
each). A task is a template plus `libero_task_id` and `init_state_id`, and is
graded by LIBERO's own success check. The simulator renders on the CPU
(OSMesa), so no GPU is needed.

Every control tick streams to the trace, so each episode replays with camera
video and state and action plots.

## Interfaces

`LIBERO_CONTROL` picks the interface the environment serves:

| Value | `robot` wire | Direct control | Use |
| --- | --- | --- | --- |
| `absolute` (default) | absolute end-effector targets | `move_to` | Tool-calling agents, from the web app or `hud eval` |
| `delta` | end-effector deltas | none | Policies trained on LIBERO, such as pi0.5 |

The platform runs the default. To run a policy against the delta interface,
serve the image locally and point the agent at it (see the
[robot benchmark cookbook](https://docs.hud.ai/v6/cookbooks/robot-benchmark)):

```bash
docker build -f Dockerfile.hud -t robot-libero .
docker run -p 8765:8765 -e LIBERO_CONTROL=delta robot-libero
```

## Results

`gpt-6-astra` at medium reasoning effort, up to 100 tool-calling turns, one
episode per task (`init_state_id` 0), on the absolute interface:

| Suite | Solved | Job |
| --- | --- | --- |
| `libero_spatial` | 9/10 | [job](https://hud.ai/jobs/955ef137fb88429c9a799633f793c705) |
| `libero_object` | 6/10 | [job](https://hud.ai/jobs/fa564b1889a249e8b638fa86de3cb2de) |
| `libero_goal` | 5/10 | [job](https://hud.ai/jobs/3e7cd8b167264b0c8c8de98c16c0b8e3) |
| `libero_10` | 3/10 | [job](https://hud.ai/jobs/55adc5df8824415e815dc24f16b5bb30) |

These are short checks, not benchmark scores. Failures are mostly the model
declaring the goal met when it was not, or the simulator's 1000-step horizon
ending the episode first.

Deploy from this directory with:

```bash
hud deploy
```
