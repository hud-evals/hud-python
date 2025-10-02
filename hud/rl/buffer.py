import random
from abc import ABC, abstractmethod
from collections import defaultdict

from hud.types import Task, Trace


def _reward_value(trace: Trace) -> float:
    value = getattr(trace, "reward", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class Buffer(ABC):
    """Base class that keeps track of tasks, traces, and group sampling."""

    def __init__(self, tasks: list[Task], group_size: int, shuffle_dataset: bool = False) -> None:
        self.group_size = group_size
        self._initialize_tasks(tasks, shuffle_dataset)
        self.task_traces: defaultdict[str, list[Trace]] = defaultdict(list)

    def _initialize_tasks(self, tasks: list[Task], shuffle: bool) -> None:
        if not tasks:
            raise ValueError("No tasks provided to buffer")

        self.tasks = list(tasks)
        if shuffle:
            random.shuffle(self.tasks)

        self.current_task_idx = 0
        self.current_taskset: list[Task] = []

    def _task_key(self, task: Task) -> str:
        task_id = getattr(task, "id", None)
        if task_id is None:
            raise ValueError("Task is missing required id")
        return str(task_id)

    def _trace_task_key(self, trace: Trace) -> str:
        task = getattr(trace, "task", None)
        assert task is not None and getattr(task, "id", None), "Trace must include a task with an id"
        return str(task.id)

    def _tasks_ready(self) -> list[str]:
        return [task_id for task_id, traces in self.task_traces.items() if len(traces) >= self.group_size]

    def add_traces(self, traces: list[Trace]) -> None:
        for trace in traces:
            if getattr(trace, "isError", False):
                continue
            task_key = self._trace_task_key(trace)
            self.task_traces[task_key].append(trace)

    def sample_tasks(self, n: int) -> list[Task]:
        if not self.current_taskset:
            requested = min(n, len(self.tasks))
            selected_keys = {self._task_key(task) for task in self.current_taskset}

            while len(self.current_taskset) < requested:
                task = self.tasks[self.current_task_idx]
                self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)

                task_key = self._task_key(task)
                if task_key in selected_keys:
                    continue
                selected_keys.add(task_key)
                self.current_taskset.append(task)

        incomplete = [
            task for task in self.current_taskset
            if len(self.task_traces[self._task_key(task)]) < self.group_size
        ]

        batched_tasks: list[Task] = []
        for task in incomplete:
            batched_tasks.extend([task] * self.group_size)
        return batched_tasks

    def reset(self) -> None:
        self.current_taskset.clear()

    def completed_groups(self) -> int:
        return sum(1 for traces in self.task_traces.values() if len(traces) >= self.group_size)

    @abstractmethod
    def sample_traces(self, n: int) -> list[Trace]:
        raise NotImplementedError

    def update(self, **kwargs) -> None:
        pass

    def __len__(self) -> int:
        return sum(len(traces) for traces in self.task_traces.values())

    @property
    def info(self) -> dict[str, int | float | str]:
        return {
            "total_traces": len(self),
            "completed_groups": self.completed_groups(),
            "total_tasks": len(self.tasks),
            "group_size": self.group_size,
        }


class SimpleBuffer(Buffer):
    """Buffer that always returns the most recent group for each task."""

    def sample_traces(self, n: int) -> list[Trace]:
        task_ids = self._tasks_ready()
        n = min(n, len(task_ids))
        sampled_traces: list[Trace] = []
        for task_id in task_ids[:n]:
            sampled_traces.extend(self.task_traces[task_id][-self.group_size:])
        return sampled_traces

    def reset(self) -> None:
        for task in self.current_taskset:
            task_key = self._task_key(task)
            self.task_traces[task_key].clear()
        super().reset()


class ReplayBuffer(Buffer):
    """Buffer that keeps a window of past traces and can replay them."""

    def __init__(
        self,
        tasks: list[Task],
        group_size: int,
        select_strategy: str,
        buffer_steps: int,
        shuffle_dataset: bool = False,
    ) -> None:
        super().__init__(tasks, group_size, shuffle_dataset)
        self.select_strategy = select_strategy
        self.buffer_steps = buffer_steps

    def _max_traces_per_task(self) -> int | None:
        if self.buffer_steps <= 0:
            return None
        return self.buffer_steps * self.group_size

    def add_traces(self, traces: list[Trace]) -> None:
        super().add_traces(traces)

        max_traces = self._max_traces_per_task()
        if max_traces is None:
            return

        for task_id, task_traces in list(self.task_traces.items()):
            if len(task_traces) > max_traces:
                self.task_traces[task_id] = task_traces[-max_traces:]

    def sample_traces(self, n: int) -> list[Trace]:
        available_task_ids = self._tasks_ready()
        if not available_task_ids:
            return []

        n = min(n, len(available_task_ids))

        if self.select_strategy == "random":
            chosen_task_ids = self._choose_random_task_ids(available_task_ids, n)
            return self._gather_random_traces(chosen_task_ids)

        return self._gather_high_variance_traces(available_task_ids, n)

    def _choose_random_task_ids(self, task_ids: list[str], n: int) -> list[str]:
        return random.sample(task_ids, n)

    def _gather_random_traces(self, task_ids: list[str]) -> list[Trace]:
        sampled: list[Trace] = []
        for task_id in task_ids:
            task_traces = self.task_traces[task_id]
            if len(task_traces) <= self.group_size:
                sampled.extend(task_traces[-self.group_size:])
            else:
                sampled.extend(random.sample(task_traces, self.group_size))
        return sampled

    def _gather_high_variance_traces(self, task_ids: list[str], n: int) -> list[Trace]:
        scored_ids = [
            (task_id, _variance([_reward_value(tr) for tr in self.task_traces[task_id]]))
            for task_id in task_ids
        ]
        scored_ids.sort(key=lambda item: item[1], reverse=True)

        top_task_ids = [task_id for task_id, _ in scored_ids[:n]]
        sampled: list[Trace] = []
        for task_id in top_task_ids:
            sampled.extend(self._select_with_earlier_injection(task_id))
        return sampled

    def _select_with_earlier_injection(self, task_id: str) -> list[Trace]:
        task_traces = self.task_traces[task_id]
        if len(task_traces) <= self.group_size:
            return list(task_traces)

        recent = list(task_traces[-self.group_size:])
        earlier = list(task_traces[:-self.group_size])
        if not earlier:
            return recent

        mean_reward = sum(_reward_value(tr) for tr in task_traces) / len(task_traces)
        earlier.sort(key=lambda tr: abs(_reward_value(tr) - mean_reward), reverse=True)

        inject_count = min(len(earlier), max(1, self.group_size // 2))
        selected = earlier[:inject_count]

        remaining = self.group_size - len(selected)
        if remaining > 0:
            selected.extend(recent[-remaining:])

        return selected

    @property
    def info(self) -> dict[str, int | float | str]:
        base = super().info
        base["select_strategy"] = self.select_strategy
        base["buffer_steps"] = self.buffer_steps
        return base


def create_buffer(
    tasks: list[Task],
    group_size: int,
    select_strategy: str,
    buffer_steps: int = 0,
    shuffle_dataset: bool = False,
) -> Buffer:
    if select_strategy == "recent":
        return SimpleBuffer(tasks, group_size, shuffle_dataset)
    if select_strategy in {"random", "variance"}:
        return ReplayBuffer(tasks, group_size, select_strategy, buffer_steps, shuffle_dataset)
    raise ValueError(f"Invalid select_strategy: {select_strategy}. Expected 'recent', 'random', or 'variance'.")
