from dataclasses import dataclass, field
from typing import Literal, TypeAlias
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator

# Simple type alias for attention implementations
AttnImplementation: TypeAlias = Literal["eager", "flash_attention_2", "sdpa"]

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


class LoRAConfig(BaseConfig):
    r: int = Field(default=8, ge=1, le=64, description="LoRA rank parameter - controls the size of the adaptation")
    alpha: int = Field(default=16, ge=1, description="LoRA alpha parameter - scaling factor for the adaptation")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    target_modules: tuple[str, ...] = Field(
        default=(
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        description="List of modules to apply LoRA to"
    )
    
    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: int) -> int:
        # This will be validated after r is set in validate_config
        return v
    
    def validate_config(self) -> None:
        super().validate_config()
        
        if self.alpha < self.r:
            raise ValueError(
                f"LoRA alpha ({self.alpha}) should be >= LoRA rank ({self.r})"
            )
        
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")


class ProcessorConfig(BaseConfig): 
    min_pixels: int = Field(default=256 * 28 * 28, ge=1, description="Minimum number of pixels for image processing")
    max_pixels: int = Field(default=512 * 28 * 28, ge=1, description="Maximum number of pixels for image processing")
    
    def validate_config(self) -> None:
        super().validate_config()
        
        if self.min_pixels >= self.max_pixels:
            raise ValueError(
                f"min_pixels ({self.min_pixels}) must be less than max_pixels ({self.max_pixels})"
            )


class ModelConfig(BaseConfig):
    base_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Base model name for fine-tuning"
    )
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA configuration"
    )
    processor: ProcessorConfig = Field(
        default_factory=ProcessorConfig,
        description="Image processor configuration"
    )
    attn_implementation: AttnImplementation = Field(default="flash_attention_2")
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code from HuggingFace"
    )
    use_liger: bool = Field(
        default=True,
        description="Whether to use Liger kernel optimizations for the model"
    )

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        validate_model(v)
        return v


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


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Training parameters
    training_steps: int = 100
    shuffle_dataset: bool = False
    save_every_batches: int = 1

    # Batching parameters
    epochs: int = 2
    batch_size: int = 24
    group_size: int = 4
    mini_batch_size: int = 1
    update_after_group: bool = True  # Whether to update the policy after each task group
    accumulate_over_minibatches: bool = False  # Whether to accumulate over minibatches

    # Advantage calculation parameters
    batch_level: Literal["group", "batch"] = "group"
    no_std: bool = False
    leave_one_out: bool = True

    # Replay buffer parameters
    buffer_steps: int = 4
    select_strategy: Literal["recent", "variance", "random"] = "variance"

    # Aggregation parameters
    ppo_mode: Literal["per_token", "per_trace"] = "per_token"
    token_agg: Literal["mean", "sum"] = "mean"  # noqa: S105

    # Regularization parameters
    kl_beta: float = 0.0
    entropy_beta: float = 0.0
    top_eps: float = 0.2
    bottom_eps: float = 0.1

    # Training hyperparameters
    lr: float = 3e-5
    grad_clip: float = 1.0

    # Adam hyperparameters
    use_8bit_optimizer: bool = True
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8


class ActorConfig(BaseConfig):
    """Actor/episode collection configuration."""

    # Model and logging
    base_model: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct", description="Base model name")
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

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        validate_model(v)
        return v

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

    # Paths
    out_dir: str = "./checkpoints"
    adapter_prefix: str = "cua-grpo-step"

    # Misc
    seed: int = 1234

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
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
            "out_dir": self.out_dir,
            "adapter_prefix": self.adapter_prefix,
            "seed": self.seed,
        }
