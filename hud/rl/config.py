"""Configuration for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# List of supported VL (Vision-Language) models
SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-14B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def validate_vl_model(model_name: str) -> None:
    """Validate that the model is a supported VL model.

    Args:
        model_name: The model name to validate

    Raises:
        ValueError: If the model is not a supported VL model
    """
    if not any(model_name.startswith(supported) for supported in SUPPORTED_MODELS):
        raise ValueError(
            f"Model '{model_name}' is not a supported VL model. "
            f"Only VL (Vision-Language) models are supported for RL training.\n"
            f"Supported models: {', '.join(SUPPORTED_MODELS)}\n"
            f"Note: '{model_name}' appears to be a text-only model."
        )


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""

    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    attn_implementation: str = "flash_attention_2"
    use_liger: bool = True
    gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # GPU parameters
    gpu_type: str = "A100"
    num_gpus: int = 2

    # Training parameters
    training_steps: int = 100
    shuffle_dataset: bool = False
    save_every_batches: int = 1

    # Batching parameters
    epochs: int = 1
    batch_size: int = 16
    group_size: int = 8
    mini_batch_size: int = 1
    update_after_group: bool = True  # Whether to update the policy after each task group
    accumulate_over_minibatches: bool = False  # Whether to accumulate over minibatches

    # Advantage calculation parameters
    batch_level: Literal["group", "batch"] = "group"
    no_std: bool = False
    leave_one_out: bool = True

    # Replay buffer parameters
    buffer_steps: int = 8
    select_strategy: Literal["recent", "variance", "random"] = "variance"

    # Aggregation parameters
    ppo_mode: Literal["per_token", "per_trace"] = "per_token"
    token_agg: Literal["mean", "sum"] = "mean"  # noqa: S105

    # Regularization parameters
    kl_beta: float = 0.001
    entropy_beta: float = 0.001
    top_eps: float = 0.2
    bottom_eps: float = 0.1

    # Training hyperparameters
    lr: float = 3e-5
    grad_clip: float = 1.0

    # Adam hyperparameters
    use_8bit_optimizer: bool = True
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8


@dataclass
class ActorConfig:
    """Actor/episode collection configuration."""

    # Execution parameters
    max_steps_per_episode: int = 5
    max_parallel_episodes: int = 48
    max_new_tokens: int = 1024
    force_tool_choice: bool = True
    allowed_tools: list[str] | None = None

    # Model parameters
    temperature: float = 0.7

    # Hud agent parameters
    system_prompt: str = "You are an expert agent. Complete the task efficiently."
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-abc123"

    # Episode execution timeout (seconds)
    episode_timeout_sec: int = 600


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)

    # Telemetry configuration
    job_name: str = "RL Training"
    job_id: str | None = None  # Use existing job ID if provided
    stats_interval: int = 1
    verbose: bool = False
    very_verbose: bool = False

    # Paths
    out_dir: str = "./checkpoints"
    adapter_prefix: str = "cua-grpo-step"

    # Misc
    seed: int = 1234

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Create config from dictionary."""
        model = ModelConfig(**d.get("model", {}))
        training = TrainingConfig(**d.get("training", {}))
        actor = ActorConfig(**d.get("actor", {}))

        return cls(
            model=model,
            training=training,
            actor=actor,
            job_name=d.get("job_name", "RL Training"),
            job_id=d.get("job_id"),
            stats_interval=d.get("stats_interval", 1),
            verbose=d.get("verbose", False),
            very_verbose=d.get("very_verbose", False),
            out_dir=d.get("out_dir", "./checkpoints"),
            adapter_prefix=d.get("adapter_prefix", "cua-grpo-step"),
            seed=d.get("seed", 1234),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "actor": self.actor.__dict__,
            "job_name": self.job_name,
            "job_id": self.job_id,
            "stats_interval": self.stats_interval,
            "verbose": self.verbose,
            "very_verbose": self.very_verbose,
            "out_dir": self.out_dir,
            "adapter_prefix": self.adapter_prefix,
            "seed": self.seed,
        }
