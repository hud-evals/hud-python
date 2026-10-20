# Robot Gripper

A tabletop robotics environment powered by a Newton physics simulation. Agents
control a gripper through MCP tools, inspect rendered views and simulator state,
and receive rewards from the resulting physical state.

The environment includes tasks for opening a drawer, lifting and moving an
object, and maintaining a force-controlled grasp. The featured task asks the
agent to pick up a mug within a 40-step budget.

Grading measures continuous progress as well as task completion, so partial
movement receives a proportional score while reaching the target earns the
success bonus.

## Run the example tasks

Use Python 3.12. The bundled Newton wheel and the scene files ship with this
project; no GPU is needed for the example's software-rendered simulation.

```bash
uv sync
uv run hud eval tasks.py claude --task-ids pick-mug --runtime local -y
```

`tasks.json` is the task source for evaluation and taskset sync; `tasks.py` binds
it to the local environment. To adapt a task, change its scene, target, or
thresholds in `tasks.json`. To add a new goal, define its
setup and simulation-state grader in `environment/env.py`. Keep grading tied to
the physical state, including every requested hold step.

## Deploy

Deploy from this directory with:

```bash
hud deploy
```
