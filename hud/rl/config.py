from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator

import re

PATTERN = re.compile(r"^Qwen/Qwen2\.5.*", re.IGNORECASE)

def validate_model(model_name: str) -> None:
    if not PATTERN.match(model_name):
        raise ValueError(f"Model '{model_name}' is not supported. Only Qwen2.5 models are supported.")

class BaseConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    @classmethod
    def from_file(cls, config_path: str | Path) -> "BaseConfig":
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            import json
            config_data = json.load(f)
        
        return cls.model_validate(config_data)
    
    def save_to_file(self, config_path: str | Path) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            import json
            json.dump(self.model_dump(), f, indent=2)
    
    def validate_config(self) -> None:
        pass
    
    def model_post_init(self, __context: object) -> None:
        self.validate_config()


class ProcessorConfig(BaseConfig): 
    min_pixels: int = Field(default=256 * 28 * 28, ge=1, description="Minimum number of pixels for image processing")
    max_pixels: int = Field(default=512 * 28 * 28, ge=1, description="Maximum number of pixels for image processing")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code from HuggingFace")
    
    def validate_config(self) -> None:
        super().validate_config()
        
        if self.min_pixels >= self.max_pixels:
            raise ValueError(
                f"min_pixels ({self.min_pixels}) must be less than max_pixels ({self.max_pixels})"
            )


class ModelConfig(BaseConfig):
    base_model: str = ""
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = Field(default="flash_attention_2")
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code from HuggingFace"
    )
    use_liger: bool = Field(
        default=True,
        description="Whether to use Liger kernel optimizations for the model"
    )
    freeze_vision_tower: bool = Field(
        default=True,
        description="Whether to freeze the vision tower parameters during training"
    )

    @model_validator(mode='after')
    def validate_dependencies(self) -> 'ModelConfig':
        if self.attn_implementation == "flash_attention_2":
            try:
                import flash_attn
            except ImportError:
                # Silent fallback to eager attention
                self.attn_implementation = "eager"
        
        if self.use_liger:
            try:
                import liger_kernel.transformers
            except ImportError:
                # Silent fallback to disable liger
                self.use_liger = False
        
        return self

    def validate_config(self) -> None:
        super().validate_config()


class OptimizerConfig(BaseConfig):
    lr: float = Field(default=3e-5, gt=0.0, description="Learning rate")
    use_8bit_optimizer: bool = Field(default=True, description="Use 8-bit Adam optimizer")
    adam_betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Adam beta parameters")
    adam_eps: float = Field(default=1e-8, gt=0.0, description="Adam epsilon")

    @field_validator("adam_betas")
    @classmethod
    def validate_adam_betas(cls, v: tuple[float, float]) -> tuple[float, float]:
        if len(v) != 2:
            raise ValueError("adam_betas must be a tuple of 2 floats")
        if not (0.0 <= v[0] < 1.0 and 0.0 <= v[1] < 1.0):
            raise ValueError("adam_betas values must be in [0, 1)")
        return v


class CheckpointConfig(BaseConfig):
    out_dir: str = Field(default="./checkpoints", description="Output directory for checkpoints")
    checkpoint_prefix: str = Field(default="cua-grpo-step", description="Prefix for checkpoint directories")


class TrainingConfig(BaseConfig):
    # Model and processor configuration
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig, description="Image processor configuration")

    # Checkpoint configuration
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Checkpoint configuration")

    # Parallelization
    dp_replicate: int = Field(default=1, ge=1, description="Data parallel replicate. To run DDP, set to num_devices")
    dp_shard: int = Field(default=1, ge=1, description="Data parallel shard. To run FSDP, set to num_devices")

    # Training parameters
    training_steps: int = Field(default=100, ge=1, description="Number of training steps")
    shuffle_dataset: bool = Field(default=False, description="Whether to shuffle the dataset")
    save_every_batches: int = Field(default=1, ge=1, description="Save checkpoint every N batches")

    # Batching parameters
    epochs: int = Field(default=2, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=24, ge=1, description="Batch size for training")
    group_size: int = Field(default=4, ge=1, description="Group size for batching")
    mini_batch_size: int = Field(default=1, ge=1, description="Mini-batch size")
    update_after_group: bool = Field(default=True, description="Whether to update the policy after each task group")
    accumulate_over_minibatches: bool = Field(default=False, description="Whether to accumulate over minibatches")

    # Advantage calculation parameters
    scale_rewards: Literal["group", "batch", "none"] = Field(default="group", description="Reward scaling strategy")
    leave_one_out: bool = Field(default=False, description="RLOO scaling factor G/(G-1), only applies when scale_rewards='none'")

    # Replay buffer parameters
    buffer_steps: int = Field(default=4, ge=0, description="Number of buffer steps")
    select_strategy: Literal["recent", "variance", "random"] = Field(default="variance", description="Buffer selection strategy")

    # Aggregation parameters
    ppo_mode: Literal["per_token", "per_trace"] = Field(default="per_token", description="PPO mode")
    token_agg: Literal["mean", "sum"] = Field(default="mean", description="Token aggregation method")

    # Regularization parameters
    kl_beta: float = Field(default=0.0, ge=0.0, description="KL divergence coefficient")
    entropy_beta: float = Field(default=0.0, ge=0.0, description="Entropy coefficient")
    top_eps: float = Field(default=0.2, ge=0.0, le=1.0, description="Top epsilon for PPO clipping")
    bottom_eps: float = Field(default=0.1, ge=0.0, le=1.0, description="Bottom epsilon for PPO clipping")

    # Gradient clipping
    grad_clip: float = Field(default=1.0, gt=0.0, description="Gradient clipping value")

    # Optimizer configuration
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer configuration")

    @model_validator(mode='after')
    def validate_model(self) -> 'TrainingConfig':
        if self.mini_batch_size > self.batch_size:
            raise ValueError(
                f"mini_batch_size ({self.mini_batch_size}) cannot be greater than "
                f"batch_size ({self.batch_size})"
            )

        if self.top_eps < self.bottom_eps:
            raise ValueError(
                f"top_eps ({self.top_eps}) must be >= bottom_eps ({self.bottom_eps})"
            )

        return self

class ActorConfig(BaseConfig):
    base_model: str = ""

    verbose: bool = Field(default=False, description="Enable verbose logging")

    # Execution parameters
    max_steps_per_episode: int = Field(default=5, ge=1, description="Maximum steps per episode")
    max_parallel_episodes: int = Field(default=48, ge=1, description="Maximum parallel episodes")

    # Agent parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    vllm_base_url: str = Field(default="http://localhost:8000/v1", description="vLLM server base URL")
    vllm_api_key: str = Field(default="token-abc123", description="vLLM API key")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum new tokens to generate")
    force_tool_choice: bool = Field(default=True, description="Force tool choice when available")
    allowed_tools: list[str] | None = Field(default=None, description="List of allowed tools (None = all)")

    # Timeouts
    request_timeout: int = Field(default=45, ge=1, le=300, description="Request timeout in seconds")
    episode_timeout_sec: int = Field(default=600, ge=1, le=3600, description="Episode timeout in seconds")


    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("allowed_tools cannot be an empty list, use None to allow all tools")
        return v

    def validate_config(self) -> None:
        super().validate_config()
        
        if self.episode_timeout_sec <= self.request_timeout:
            raise ValueError(
                f"episode_timeout_sec ({self.episode_timeout_sec}) must be greater than "
                f"request_timeout ({self.request_timeout})"
            )


class Config(BaseConfig):
    """Main configuration combining all sub-configs."""

    # Main model specification - shared across model and actor configs
    base_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Base model name shared across model and actor configurations"
    )

    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    actor: ActorConfig = Field(default_factory=ActorConfig, description="Actor configuration")

    # Telemetry configuration
    job_name: str = Field(default="RL Training", description="Job name for telemetry")
    job_id: str | None = Field(default=None, description="Use existing job ID if provided")
    stats_interval: int = Field(default=1, ge=1, description="Statistics collection interval")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    # Misc
    seed: int = Field(default=1234, description="Random seed for reproducibility")

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        validate_model(v)
        return v

    @model_validator(mode='after')
    def propagate_base_model(self) -> 'Config':
        """Propagate base_model to sub-configs."""
        self.training.model.base_model = self.base_model
        self.actor.base_model = self.base_model
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from dictionary."""
        training = TrainingConfig.model_validate(d.get("training", {}))
        actor = ActorConfig.model_validate(d.get("actor", {}))

        return cls(
            base_model=d.get("base_model", "Qwen/Qwen2.5-VL-3B-Instruct"),
            training=training,
            actor=actor,
            job_name=d.get("job_name", "RL Training"),
            job_id=d.get("job_id"),
            stats_interval=d.get("stats_interval", 1),
            verbose=d.get("verbose", False),
            seed=d.get("seed", 1234),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "base_model": self.base_model,
            "training": self.training.model_dump(),
            "actor": self.actor.model_dump(),
            "job_name": self.job_name,
            "job_id": self.job_id,
            "stats_interval": self.stats_interval,
            "verbose": self.verbose,
            "seed": self.seed,
        }
