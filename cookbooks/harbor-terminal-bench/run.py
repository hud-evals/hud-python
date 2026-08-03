"""Run Harbor task directories through the HUD runtime.

uv run run.py ./terminal-bench                    # every task, with the oracle
uv run run.py ./terminal-bench --agent claude-sonnet-4-5
uv run run.py ./terminal-bench --task qemu-startup --task regex-log
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hud.agents import create_agent
from hud.agents.base import Agent
from hud.eval import DockerRuntime
from hud.eval.run import Run
from hud.integrations import harbor

if TYPE_CHECKING:
    from hud.capabilities import SSHClient


def task_dirs(dataset: Path) -> list[Path]:
    """The Harbor tasks under *dataset* — or *dataset* itself, if it is one."""
    if (dataset / "task.toml").is_file():
        return [dataset]
    return sorted(child for child in dataset.iterdir() if (child / "task.toml").is_file())


class Oracle(Agent):
    """Run each task's ``solution/solve.sh`` instead of a model."""

    def __init__(self, tasks: list[Path]) -> None:
        self.solutions = {task.name: task / "solution" / "solve.sh" for task in tasks}

    async def __call__(self, run: Run) -> None:
        solution = self.solutions[run.task_id]
        ssh = cast("SSHClient", await run.client.open("ssh/2"))
        result = await ssh.conn.run(solution.read_text("utf-8"), check=True)
        if result.exit_status is None:
            raise RuntimeError("solution SSH channel closed without an exit status")
        run.trace.content = "solution completed"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path, help="a Harbor task dir, or a directory of them")
    parser.add_argument(
        "--task", action="append", default=[], metavar="ID", help="run only these task dirs"
    )
    parser.add_argument(
        "--agent",
        default="oracle",
        help="'oracle' to run each task's own solution, else a model name",
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument(
        "--hud-requirement",
        default="hud",
        help="the HUD package requirement installed into the adapted image",
    )
    args = parser.parse_args()

    wanted = set(args.task)
    tasks = [task for task in task_dirs(args.dataset) if not wanted or task.name in wanted]
    if not tasks:
        raise SystemExit(f"no Harbor tasks in {args.dataset}")
    if missing := wanted - {task.name for task in tasks}:
        raise SystemExit(f"no such task: {', '.join(sorted(missing))}")
    if args.agent == "oracle":
        missing = [task.name for task in tasks if not (task / "solution" / "solve.sh").is_file()]
        if missing:
            raise SystemExit(f"tasks without solution/solve.sh: {', '.join(missing)}")

    taskset = await harbor.adapt(
        args.dataset,
        task_ids=wanted or None,
        hud_requirement=args.hud_requirement,
    )

    agent = (
        Oracle(tasks)
        if args.agent == "oracle"
        else create_agent(args.agent, max_steps=args.max_steps)
    )
    job = await taskset.run(
        agent,
        runtime=DockerRuntime(),
        max_concurrent=args.max_concurrent,
    )

    print(f"\n{'task':<40} {'reward':>7}  status")
    for run in sorted(job.runs, key=lambda r: r.task_id):
        print(f"{run.task_id:<40} {run.reward:>7}  {run.trace.status}")

    errored = len(job.errors)
    scored = len(job.runs) - errored
    print(f"\nscored {scored}/{len(job.runs)} (errored {errored}), mean reward {job.reward:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
