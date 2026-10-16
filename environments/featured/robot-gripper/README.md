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

Deploy from this directory with:

```bash
hud deploy
```
