import math
from typing import TypedDict

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
    temperature: torch.Tensor
    advantage: torch.Tensor
    old_logprobs: torch.Tensor | None = Field(default=None)
    ref_logprobs: torch.Tensor | None = Field(default=None)

    def to_device(self, device: torch.device) -> "TrainingSample":
        """Move sample to device."""
        self.inputs = {
            "input_ids": self.inputs["input_ids"].to(device),
            "attention_mask": self.inputs["attention_mask"].to(device),
            "assistant_mask": self.inputs["assistant_mask"].to(device),
            "pixel_values": self.inputs["pixel_values"].to(device) if self.inputs["pixel_values"] is not None else None,
            "image_grid_thw": self.inputs["image_grid_thw"].to(device) if self.inputs["image_grid_thw"] is not None else None,
        }
        self.temperature = self.temperature.to(device)
        self.advantage = self.advantage.to(device)
        self.old_logprobs = self.old_logprobs.to(device) if self.old_logprobs is not None else None
        self.ref_logprobs = self.ref_logprobs.to(device) if self.ref_logprobs is not None else None
        return self
