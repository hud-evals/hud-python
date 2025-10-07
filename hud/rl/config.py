import re
import uuid
import math
from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator

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


class LossConfig(BaseConfig):
    apply_length_norm: bool = Field(default=False, description="Apply length norm to the importance ratio; only applies when importance_sampling_level is 'sequence'")
    importance_sampling_level: Literal["token", "sequence"] = Field(default="token", description="Importance sampling level")
    kl_beta: float = Field(default=0.0, ge=0.0, description="KL divergence coefficient")
    clip_ratio: float = Field(default=10.0, description="Clip ratio for importance sampling")

class TrainingConfig(BaseConfig):
    # Mode and Parallelization
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    dp_replicate: int = Field(default=1, ge=1, description="Data parallel replicate.")
    dp_shard: int = Field(default=1, ge=1, description="Data parallel shard.")

    # Checkpointing
    output_dir: str = ""
    save_last_n: int = Field(default=1, ge=1, description="Save last N checkpoints")

    # Loss configuration
    loss: LossConfig = Field(default_factory=LossConfig, description="Loss configuration")

    # Optimizer configuration
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer configuration")
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Maximum gradient norm")

class RewardConfig(BaseConfig):
    scale_rewards: Literal["group", "batch", "none"] = Field(default="group", description="Reward scaling strategy")
    leave_one_out: bool = Field(default=False, description="RLOO scaling factor G/(G-1), only applies when scale_rewards='none'")

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

    # Timeouts
    request_timeout: int = Field(default=45, ge=1, le=300, description="Request timeout in seconds")
    episode_timeout_sec: int = Field(default=600, ge=1, le=3600, description="Episode timeout in seconds")

    def validate_config(self) -> None:
        super().validate_config()
        
        if self.episode_timeout_sec <= self.request_timeout:
            raise ValueError(
                f"episode_timeout_sec ({self.episode_timeout_sec}) must be greater than "
                f"request_timeout ({self.request_timeout})"
            )


class Config(BaseConfig):
    """Top-level RL configuration with rollout and trainer settings."""

    base_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Base model name shared across model and actor configurations"
    )

    processor: ProcessorConfig = Field(default_factory=ProcessorConfig, description="Processor configuration")

    num_gpus: int = Field(default=1, ge=1, description="Number of GPUs to use for training")
    training_steps: int = Field(default=100, ge=1, description="Number of training steps")
    shuffle_dataset: bool = Field(default=False, description="Whether to shuffle the dataset")
    batch_size: int = Field(default=24, ge=1, description="Global batch size for training")
    mini_batch_size: int = Field(default=1, ge=1, description="Mini-batch size")
    group_size: int = Field(default=4, ge=1, description="Group size i.e. number of rollouts per prompt")
    rewards: RewardConfig = Field(default_factory=RewardConfig, description="Reward scaling configuration")
    buffer_steps: int = Field(default=4, ge=0, description="Number of buffer steps")
    select_strategy: Literal["recent", "variance", "random"] = Field(default="variance", description="Buffer selection strategy")

    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Trainer configuration")
    actor: ActorConfig = Field(default_factory=ActorConfig, description="Actor configuration")

    job_name: str = Field(default="RL Training", description="Job name for telemetry")
    job_id: str = Field(default=str(uuid.uuid4()), description="Job ID for telemetry")
    stats_interval: int = Field(default=1, ge=1, description="Statistics collection interval")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    seed: int = Field(default=1234, description="Random seed for reproducibility")

    output_dir: str = Field(default="./outputs", description="Output directory for batches/checkpoints")

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

    @model_validator(mode='after')
    def propagate_output_dir(self) -> 'Config':
        """Propagate output_dir to sub-configs."""
        self.training.output_dir = self.output_dir
        return self

    @model_validator(mode='after')
    def validate_batching(self) -> 'Config':
        """Ensure batch sizing aligns with group collection and distributed layout."""
        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by group_size ({self.group_size})"
            )

        world_size = self.expected_world_size
        if world_size <= 0:
            raise ValueError("Expected world size must be at least 1")

        return self

    @property
    def expected_world_size(self) -> int:
        return max(1, self.training.dp_replicate * self.training.dp_shard)

    @property
    def grad_accumulation_steps(self) -> int:
        """Number of microbatches each rank processes before an optimizer step."""
        world_size = self.expected_world_size
        total_microbatches = max(1, math.ceil(self.batch_size / self.mini_batch_size))
        return max(1, math.ceil(total_microbatches / world_size))
