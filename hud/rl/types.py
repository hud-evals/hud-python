import math
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass

try:
    import torch
except ImportError:
    raise ImportError("uv tool install hud-python[rl] to use this module") from None

class ProcessedInputs(TypedDict):
    """Tokenized inputs ready for model forward pass."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    assistant_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None


class TrainingSample(BaseModel):
    """A training sample for GRPO."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Tokenized inputs to the model (model.forward(**inputs))
    inputs: ProcessedInputs
    old_logprobs: torch.Tensor | None = Field(default=None)
    ref_logprobs: torch.Tensor | None = Field(default=None)
    advantage: torch.Tensor | None = Field(default=None)

    def to_device(self, device: torch.device) -> "TrainingSample":
        """Move sample to device."""
        self.inputs = {
            "input_ids": self.inputs["input_ids"].to(device),
            "attention_mask": self.inputs["attention_mask"].to(device),
            "assistant_mask": self.inputs["assistant_mask"].to(device),
            "pixel_values": self.inputs["pixel_values"].to(device) if self.inputs["pixel_values"] is not None else None,
            "image_grid_thw": self.inputs["image_grid_thw"].to(device) if self.inputs["image_grid_thw"] is not None else None,
        }
        self.advantage = self.advantage.to(device) if self.advantage is not None else None
        self.old_logprobs = self.old_logprobs.to(device) if self.old_logprobs is not None else None
        self.ref_logprobs = self.ref_logprobs.to(device) if self.ref_logprobs is not None else None
        return self


@dataclass
class Metric:
    """A tuple for metrics."""

    name: str = Field(default="")
    mean: float = Field(default=0.0)
    std: float = Field(default=0.0)
    values: list[float] = Field(default_factory=list)

    def update(
        self, value: float | torch.Tensor | list[float] | list[int] | list[torch.Tensor]
    ) -> None:
        """Update metric."""
        if isinstance(value, list):
            self.values.extend(value.item() if isinstance(value, torch.Tensor) else value)  # type: ignore
        else:
            self.values.append(value.item() if isinstance(value, torch.Tensor) else value)  # type: ignore
        mean_val = sum(self.values) / len(self.values)
        self.mean = mean_val.item() if isinstance(mean_val, torch.Tensor) else float(mean_val)  # type: ignore
        variance = sum((x - self.mean) ** 2 for x in self.values) / len(self.values)
        variance_val = variance.item() if isinstance(variance, torch.Tensor) else float(variance)  # type: ignore
        self.std = math.sqrt(variance_val)


@dataclass
class TrainingMetrics:
    """Metrics for GRPO training (per training step)."""

    # Learner metrics
    grad_norm: Metric = Field(default=Metric())
    loss: Metric = Field(default=Metric())
    kl: Metric = Field(default=Metric())
    reward: Metric = Field(default=Metric())
    advantage: Metric = Field(default=Metric())
    policy_ratio: Metric = Field(default=Metric())
    tokens: Metric = Field(default=Metric())
    entropy: Metric = Field(default=Metric())

    # Computation metrics
    gpu_util: Metric = Field(default=Metric())  # GPU utilization percentage
    gpu_memory: Metric = Field(default=Metric())  # GPU memory usage in GB
    episode_time: Metric = Field(default=Metric())  # Time to run episodes (actor)
    training_time: Metric = Field(default=Metric())  # Time for gradient updates (learner)
    samples_per_second: Metric = Field(default=Metric())  # Training throughput

    def update(self, metrics: dict[str, Any]) -> None:
        """Update metrics."""
        for key, value in metrics.items():
            if key in self.__dataclass_fields__:
                getattr(self, key).update(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        final_metrics = {}
        for key in self.__dataclass_fields__:
            final_metrics[f"{key}_mean"] = getattr(self, key).mean
            final_metrics[f"{key}_std"] = getattr(self, key).std
        return final_metrics
