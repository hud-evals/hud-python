import torch
import torch.distributed as dist


class MetricsCollector:
    """maintains streaming (sum, sumsq, count) per metric key
    instead of storing raw tensors. Reduces triplets across ranks when
    `distributed` is enabled.
    """

    def __init__(self, distributed: bool = False) -> None:
        self.distributed = distributed
        self.reset()

    def reset(self) -> None:
        # Local accumulators: key -> (sum, sumsq, count)
        self._acc: dict[str, tuple[float, float, int]] = {}

    def log(self, **kwargs: torch.Tensor) -> None:
        """Update streaming stats for each key with a tensor of values.

        Accepts tensors on any device. Detaches tensors and computes
        sum/sumsq/count on their current device to avoid large host copies.
        """
        for key, tensor in kwargs.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"MetricsCollector.log expects tensors; got {type(tensor)} for key '{key}'"
                )

            t = tensor.detach()

            # DTensor support: prefer local shard if available
            to_local = getattr(t, "to_local", None)
            if callable(to_local):
                try:
                    t = to_local()
                except Exception:
                    pass

            # Compute stats on the tensor's device to minimize transfers
            x = t.reshape(-1).float()
            s = float(x.sum().item())
            # Use square() to avoid materializing an extra tensor via x * x
            s2 = float(x.square().sum().item())
            n = int(x.numel())

            if key in self._acc:
                s0, s20, n0 = self._acc[key]
                self._acc[key] = (s0 + s, s20 + s2, n0 + n)
            else:
                self._acc[key] = (s, s2, n)

    def _union_keys(self, local: set[str]) -> list[str]:
        if not self.distributed or not dist.is_initialized():
            return sorted(local)
        gathered: list[list[str]] = [None] * dist.get_world_size()  # type: ignore[assignment]
        dist.all_gather_object(gathered, sorted(local))
        union: set[str] = set()
        for keys in gathered:
            union.update(keys)
        return sorted(union)

    def _triplet_from_acc(self, key: str) -> tuple[float, float, int]:
        return self._acc.get(key, (0.0, 0.0, 0))

    def _reduce_triplet(self, triplet: tuple[float, float, int]) -> tuple[float, float, int]:
        if not self.distributed or not dist.is_initialized():
            return triplet
        world_size = dist.get_world_size()
        gathered: list[tuple[float, float, int]] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(gathered, triplet)
        s = sum(t[0] for t in gathered)
        s2 = sum(t[1] for t in gathered)
        n = sum(t[2] for t in gathered)
        return (float(s), float(s2), int(n))

    @staticmethod
    def _mean_std(s: float, s2: float, n: int) -> tuple[float, float]:
        if n <= 0:
            return (0.0, 0.0)
        mean = s / n
        var = max(0.0, (s2 / n) - (mean * mean))
        return (mean, var ** 0.5)

    def get_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        keys = self._union_keys(set(self._acc.keys()))
        for key in keys:
            s, s2, n = self._triplet_from_acc(key)
            s, s2, n = self._reduce_triplet((s, s2, n))
            mean, std = self._mean_std(s, s2, n)
            out[f"{key}_mean"] = float(mean)
            out[f"{key}_std"] = float(std)
        return out
