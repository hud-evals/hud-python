import torch

from hud.rl.types import Metric, TrainingMetrics


class MetricsCollector:
    """Simple metrics collector with distributed support."""

    def __init__(self, distributed: bool = False) -> None:
        """Initialize the metrics collector.

        Args:
            distributed: Whether to aggregate metrics across ranks
        """
        self.metrics: dict[str, list[float]] = {}
        self.distributed = distributed

    def log(self, **kwargs: float | torch.Tensor) -> None:
        """Log metrics values.

        Args:
            **kwargs: Metric name-value pairs to log
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []

            # Convert tensors to float
            if isinstance(value, torch.Tensor):
                value = value.item()

            self.metrics[key].append(float(value))

    def get_stats(self) -> TrainingMetrics:
        """Get aggregated metrics statistics.

        Returns:
            TrainingMetrics object with computed statistics
        """
        # Aggregate across ranks if distributed
        if self.distributed:
            metrics_dict = self._aggregate_across_ranks()
        else:
            metrics_dict = self.metrics

        # Create TrainingMetrics object
        training_metrics = TrainingMetrics()

        # Update each metric field with collected values
        for field_name in training_metrics.__dataclass_fields__:
            if field_name in metrics_dict:
                metric = Metric(name=field_name, values=metrics_dict[field_name])
                # Compute mean and std
                if metric.values:
                    metric.mean = sum(metric.values) / len(metric.values)
                    variance = sum((x - metric.mean) ** 2 for x in metric.values) / len(metric.values)
                    metric.std = variance ** 0.5
                setattr(training_metrics, field_name, metric)

        return training_metrics

    def _aggregate_across_ranks(self) -> dict[str, list[float]]:
        """Aggregate metrics across all distributed ranks.

        Returns:
            Dictionary with aggregated metrics from all ranks
        """
        from hud.rl.distributed import get_local_rank, get_world_size, is_main_process

        world_size = get_world_size()
        if world_size <= 1:
            return self.metrics

        # Get the union of all metric keys across ranks
        local_keys = set(self.metrics.keys())

        # For simplicity, we'll aggregate the last value from each rank
        # and update our metrics dict with the mean
        aggregated = {}

        for key in local_keys:
            if key in self.metrics and self.metrics[key]:
                # Get last value for this metric
                local_value = self.metrics[key][-1]
            else:
                local_value = 0.0

            # Create tensor for this metric
            value_tensor = torch.tensor(
                local_value,
                device=f"cuda:{get_local_rank()}",
                dtype=torch.float32
            )

            # Gather from all ranks
            gather_list = [torch.zeros_like(value_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, value_tensor)

            # Compute mean across all ranks
            all_values = torch.stack(gather_list).cpu().numpy()
            mean_value = float(all_values.mean())

            # Update our metrics with the aggregated value
            if key not in aggregated:
                aggregated[key] = []
            # Keep history but replace last with aggregated
            aggregated[key] = self.metrics[key][:-1] + [mean_value]

        # On non-main processes, just return local metrics
        # Only main process gets the aggregated values
        if is_main_process():
            return aggregated
        else:
            return self.metrics

    def reset(self) -> None:
        """Clear all collected metrics."""
        self.metrics = {}