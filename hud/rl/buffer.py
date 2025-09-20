import random
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from hud.rl.logger import console
from hud.types import Task, Trace

if TYPE_CHECKING:
    from hud.rl.config import Config


class Buffer(ABC):
    """Abstract base class for managing tasks and traces with sampling strategies."""

    @abstractmethod
    def add_tasks(self, tasks: list[Task]) -> None:
        """Add raw dataset tasks to the buffer."""

    @abstractmethod
    def add_traces(self, traces: list[Trace]) -> None:
        """Add completed rollouts to the buffer."""

    @abstractmethod
    def sample_tasks(self, n: int) -> list[Task]:
        """Sample tasks for the actor to execute."""

    @abstractmethod
    def sample_traces(self, batch_size: int | None = None) -> list[Trace]:
        """Sample traces for training."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update buffer state/strategy based on training progress."""

    def __len__(self) -> int:
        """Return the number of traces in the buffer."""
        return 0


class SimpleBuffer(Buffer):
    """Simple buffer with recent sampling strategy."""

    def __init__(self, tasks: list[Task], config: Config) -> None:
        self.config = config
        self.batch_size = config.training.batch_size
        self.group_size = config.training.group_size
        self.training_steps = config.training.training_steps

        # Validate group and batch sizes
        if self.group_size > self.batch_size:
            raise ValueError(
                f"Group size is greater than batch size, {self.group_size} > {self.batch_size}"
            )
        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"A batch cannot have irregular groups, {self.group_size} % {self.batch_size} != 0"
            )

        self.groups_per_batch = self.batch_size // self.group_size
        self.number_of_tasks = self.training_steps * self.groups_per_batch

        # Initialize task queue with provided tasks
        self.task_queue: deque[Task] = deque()
        self._initialize_tasks(tasks)

        # Initialize trace buffer
        self.trace_buffer: deque[Trace] = deque(
            maxlen=config.training.buffer_steps * config.training.batch_size
        )

    def _initialize_tasks(self, tasks: list[Task]) -> None:
        """Initialize task queue with repetition to fill training steps."""
        if not tasks:
            raise ValueError("No tasks provided to buffer")

        # Shuffle if configured
        if self.config.training.shuffle_dataset:
            random.shuffle(tasks)

        # Fill task queue to match number of training steps
        while len(self.task_queue) < self.number_of_tasks:
            for task in tasks:
                self.task_queue.append(task)
                if len(self.task_queue) >= self.number_of_tasks:
                    break

    def add_tasks(self, tasks: list[Task]) -> None:
        """Add tasks to the buffer."""
        for task in tasks:
            self.task_queue.append(task)

    def add_traces(self, traces: list[Trace]) -> None:
        """Add traces to the buffer."""
        for trace in traces:
            self.trace_buffer.append(trace)

    def sample_tasks(self, n: int | None = None) -> list[Task]:
        """Sample n tasks from the buffer, creating groups with repetition."""
        if n is None:
            n = self.groups_per_batch

        tasks = []
        for _ in range(n):
            if self.task_queue:
                tasks.append(self.task_queue.popleft())

        # Create groups where each task is repeated group_size times
        result = []
        for task in tasks:
            result.extend([task] * self.group_size)
        return result

    def sample_traces(self, batch_size: int | None = None) -> list[Trace]:
        """Sample most recent traces (FIFO)."""
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.trace_buffer) < batch_size:
            console.warning(f"Buffer has {len(self.trace_buffer)} traces, requested {batch_size}")
            return list(self.trace_buffer)

        # Return most recent traces
        return list(self.trace_buffer)[-batch_size:]

    def update(self, **kwargs) -> None:
        """Update buffer state (no-op for simple buffer)."""

    def __len__(self) -> int:
        """Return the number of remaining tasks in the buffer."""
        return len(self.task_queue)

    @property
    def info(self) -> dict[str, int | float | str]:
        """Get buffer statistics."""
        return {
            "remaining_tasks": len(self.task_queue),
            "total_traces": len(self.trace_buffer),
            "training_steps": self.training_steps,
            "group_size": self.group_size,
            "batch_size": self.batch_size,
        }


class ReplayBuffer(Buffer):
    """Replay buffer with configurable sampling strategy (recent, random, variance)."""

    def __init__(self, tasks: list[Task], config: Config) -> None:
        self.config = config
        self.batch_size = config.training.batch_size
        self.group_size = config.training.group_size
        self.training_steps = config.training.training_steps
        self.select_strategy = config.training.select_strategy

        # Validate group and batch sizes
        if self.group_size > self.batch_size:
            raise ValueError(
                f"Group size is greater than batch size, {self.group_size} > {self.batch_size}"
            )
        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"A batch cannot have irregular groups, {self.group_size} % {self.batch_size} != 0"
            )

        self.groups_per_batch = self.batch_size // self.group_size
        self.number_of_tasks = self.training_steps * self.groups_per_batch

        # Initialize task queue with provided tasks
        self.task_queue: deque[Task] = deque()
        self._initialize_tasks(tasks)

        # Initialize trace buffer
        self.trace_buffer: deque[Trace] = deque(
            maxlen=config.training.buffer_steps * config.training.batch_size
        )

    def _initialize_tasks(self, tasks: list[Task]) -> None:
        """Initialize task queue with repetition to fill training steps."""
        if not tasks:
            raise ValueError("No tasks provided to buffer")

        # Shuffle if configured
        if self.config.training.shuffle_dataset:
            random.shuffle(tasks)

        # Fill task queue to match number of training steps
        while len(self.task_queue) < self.number_of_tasks:
            for task in tasks:
                self.task_queue.append(task)
                if len(self.task_queue) >= self.number_of_tasks:
                    break

    def add_tasks(self, tasks: list[Task]) -> None:
        """Add tasks to the buffer."""
        for task in tasks:
            self.task_queue.append(task)

    def add_traces(self, traces: list[Trace]) -> None:
        """Add traces to the buffer."""
        for trace in traces:
            self.trace_buffer.append(trace)

    def sample_tasks(self, n: int | None = None) -> list[Task]:
        """Sample n tasks from the buffer, creating groups with repetition."""
        if n is None:
            n = self.groups_per_batch

        tasks = []
        for _ in range(n):
            if self.task_queue:
                tasks.append(self.task_queue.popleft())

        # Create groups where each task is repeated group_size times
        result = []
        for task in tasks:
            result.extend([task] * self.group_size)
        return result

    def sample_traces(self, batch_size: int | None = None) -> list[Trace]:
        """Sample traces using configured strategy."""
        if batch_size is None:
            batch_size = self.batch_size

        if self.select_strategy == "recent":
            if len(self.trace_buffer) < batch_size:
                return list(self.trace_buffer)
            return list(self.trace_buffer)[-batch_size:]
        elif self.select_strategy == "random":
            if len(self.trace_buffer) < batch_size:
                return list(self.trace_buffer)
            return random.sample(list(self.trace_buffer), batch_size)
        elif self.select_strategy == "variance":
            return self._sample_high_variance_traces()
        else:
            raise ValueError(f"Invalid select strategy: {self.select_strategy}")

    def update(self, **kwargs) -> None:
        """Update buffer state (can be used for curriculum learning, etc)."""

    def __len__(self) -> int:
        """Return the number of remaining tasks in the buffer."""
        return len(self.task_queue)

    @property
    def info(self) -> dict[str, int | float | str]:
        """Get buffer statistics."""
        return {
            "remaining_tasks": len(self.task_queue),
            "total_traces": len(self.trace_buffer),
            "training_steps": self.training_steps,
            "select_strategy": self.select_strategy,
            "group_size": self.group_size,
            "batch_size": self.batch_size,
        }

    def _extract_group_key(self, trace: Trace) -> tuple[str, str]:
        """Return a stable grouping key for a trace.

        Preference order:
        1) task.id when present (kind='id')
        2) task.prompt exact string (kind='prompt') when id is None
        3) 'NA' for missing/errored entries (kind='NA')
        """
        if getattr(trace, "isError", False):
            return ("NA", "NA")

        task = getattr(trace, "task", None)
        if task is None:
            return ("NA", "NA")

        tid = getattr(task, "id", None)
        if tid is not None:
            return ("id", str(tid))

        prompt = getattr(task, "prompt", None)
        if prompt:
            return ("prompt", str(prompt))

        return ("NA", "NA")

    def _validate_and_split_groups(
        self, recent_traces: list[Trace]
    ) -> tuple[list[list[Trace]], list[tuple[str, str]]]:
        """Validate and split recent traces into homogeneous groups by id or prompt.

        - Uses id when present; otherwise falls back to prompt equality.
        - Any NA/error traces are excluded and the group is filled by duplicating
          existing valid members in that group.
        - Always returns len == groups_per_batch groups of size == group_size.
        """
        from collections import Counter

        groups_per_batch = self.batch_size // self.group_size

        window_keys = [self._extract_group_key(t) for t in recent_traces]
        window_counter = Counter(k for k in window_keys if k[0] != "NA")

        validated_groups: list[list[Trace]] = []
        selected_keys: list[tuple[str, str]] = []

        for g_idx in range(groups_per_batch):
            start = g_idx * self.group_size
            end = start + self.group_size
            chunk = recent_traces[start:end]

            key_counts = Counter()
            per_item_keys: list[tuple[str, str]] = []
            for tr in chunk:
                k = self._extract_group_key(tr)
                per_item_keys.append(k)
                if k[0] != "NA":
                    key_counts[k] += 1

            if key_counts:
                best_key = key_counts.most_common(1)[0][0]
            elif window_counter:
                best_key = window_counter.most_common(1)[0][0]
            else:
                best_key = ("NA", "NA")

            homogeneous = [tr for tr, k in zip(chunk, per_item_keys, strict=False) if k == best_key]

            while len(homogeneous) < self.group_size:
                if homogeneous:
                    homogeneous.append(homogeneous[-1])
                else:
                    idx = next((i for i, wk in enumerate(window_keys) if wk[0] != "NA"), None)
                    if idx is not None:
                        homogeneous.append(recent_traces[idx])
                    elif chunk:
                        homogeneous.append(chunk[0])
                    else:
                        homogeneous.append(recent_traces[0])

            validated_groups.append(homogeneous)
            selected_keys.append(best_key)

        return validated_groups, selected_keys

    def _sample_high_variance_traces(self) -> list[Trace]:
        from collections import Counter, defaultdict, deque

        buf_list = list(self.trace_buffer)
        if len(buf_list) < self.batch_size:
            console.warning(
                f"[group-sampler] Buffer has only {len(buf_list)} traces, need {self.batch_size}"
            )
            while len(buf_list) < self.batch_size:
                take = min(len(buf_list) or 1, self.batch_size - len(buf_list))
                buf_list.extend(buf_list[:take])
        recent_traces = buf_list[-self.batch_size :]

        recent_keys = [self._extract_group_key(t) for t in recent_traces]
        console.info(f"[group-sampler] recent-window histogram: {Counter(recent_keys)}")

        console.info(
            f"[group-sampler] Building earlier traces lookup, buffer size: {len(buf_list)}"
        )
        earlier_traces_by_key: dict[tuple[str, str], deque[Trace]] = defaultdict(deque)
        for tr in buf_list[: -self.batch_size]:
            k = self._extract_group_key(tr)
            if k[0] != "NA":
                earlier_traces_by_key[k].append(tr)

        groups, group_keys = self._validate_and_split_groups(recent_traces)

        final_traces: list[Trace] = []
        for g_idx, (homogeneous, target_key) in enumerate(zip(groups, group_keys, strict=False)):

            def current_mean(h: list[Trace]) -> float:
                if not h:
                    return 0.0
                vals = [float(getattr(t, "reward", 0.0) or 0.0) for t in h]
                return sum(vals) / len(vals)

            pool = earlier_traces_by_key.get(target_key, deque())
            if pool:
                pool_vals = [float(getattr(tr, "reward", 0.0) or 0.0) for tr in list(pool)]
                if pool_vals:
                    pool_mean = sum(pool_vals) / len(pool_vals)
                    pool_var = sum((v - pool_mean) * (v - pool_mean) for v in pool_vals) / len(
                        pool_vals
                    )
                    console.info(
                        f"[group-sampler] Group {g_idx}: earlier-pool size={len(pool_vals)} "
                        f"mean={pool_mean:.4f} std={(pool_var**0.5):.4f}"
                    )

                replace_k = max(1, self.group_size // 4)
                replace_k = min(replace_k, len(pool), self.group_size)

                if replace_k > 0:
                    mu = current_mean(homogeneous)
                    pool_list = list(pool)
                    pool_indices = list(range(len(pool_list)))
                    pool_indices.sort(
                        key=lambda i: abs(
                            (float(getattr(pool_list[i], "reward", 0.0) or 0.0)) - mu
                        ),
                        reverse=True,
                    )
                    chosen_pool_idx = set(pool_indices[:replace_k])
                    replacements = [pool_list[i] for i in pool_indices[:replace_k]]

                    remaining = [tr for i, tr in enumerate(pool_list) if i not in chosen_pool_idx]
                    earlier_traces_by_key[target_key] = deque(remaining)

                    group_indices = list(range(len(homogeneous)))
                    mu = current_mean(homogeneous)
                    group_indices.sort(
                        key=lambda i: abs(
                            (float(getattr(homogeneous[i], "reward", 0.0) or 0.0)) - mu
                        )
                    )
                    target_positions = group_indices[:replace_k]

                    for pos, new_tr in zip(target_positions, replacements, strict=False):
                        homogeneous[pos] = new_tr

            if any(self._extract_group_key(t) != target_key for t in homogeneous):
                raise RuntimeError(f"Group {g_idx} is not homogeneous after sampling")
            final_traces.extend(homogeneous)

        for i in range(0, len(final_traces), self.group_size):
            block = final_traces[i : i + self.group_size]
            keys = {self._extract_group_key(t) for t in block}
            if len(keys) != 1:
                raise RuntimeError(f"Homogeneity validation failed for block starting at index {i}")

        console.info(
            f"[group-sampler] final histogram: "
            f"{Counter(self._extract_group_key(t) for t in final_traces)}"
        )
        return final_traces
