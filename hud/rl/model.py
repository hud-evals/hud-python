import torch
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl  # type: ignore

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

from hud.rl.distributed import get_local_rank, get_world_size
from hud.rl.logger import console

if TYPE_CHECKING:
    from .config import Config


def load_processor(config: Config) -> Any:
    """Load the appropriate processor/tokenizer for the model."""
    model_cfg = config.model
    is_vl_model = "VL" in model_cfg.base_model

    if is_vl_model:
        # Some environments require remote code for Qwen2.5-VL processors
        processor = AutoProcessor.from_pretrained(
            model_cfg.base_model,
            min_pixels=model_cfg.min_pixels,
            max_pixels=model_cfg.max_pixels,
            trust_remote_code=True,
        )
    else:
        processor = AutoTokenizer.from_pretrained(model_cfg.base_model)

    return processor


def load_policy_model(config: Config) -> Any:
    """Load and configure the policy model with LoRA."""
    model_cfg = config.model
    is_vl_model = "VL" in model_cfg.base_model
    model_type = "Vision-Language" if is_vl_model else "Text"
    console.info_log(f"Loading {model_type} model: {model_cfg.base_model}")

    # Apply Liger kernel optimizations if available and enabled
    if model_cfg.use_liger and LIGER_AVAILABLE:
        if is_vl_model:
            console.info_log("Applying Liger kernel optimizations to Qwen2.5-VL")
            apply_liger_kernel_to_qwen2_5_vl(
                rope=True,  # Optimized RoPE
                rms_norm=True,  # Optimized RMSNorm
                swiglu=True,  # Optimized SwiGLU
                fused_linear_cross_entropy=True,  # Fused Linear+CrossEntropy for memory
            )
    elif model_cfg.use_liger and not LIGER_AVAILABLE:
        console.warning(
            "Liger kernel requested but not installed. Install with: pip install liger-kernel"
        )

    # Use attention implementation from config
    attn_implementation = model_cfg.attn_implementation

    # Choose the appropriate model class
    model_class = Qwen2_5_VLForConditionalGeneration if is_vl_model else AutoModelForCausalLM

    try:
        policy = model_class.from_pretrained(
            model_cfg.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        console.info_log(f"Using {attn_implementation} for attention")
    except (ImportError, ValueError) as e:
        # Only fallback if explicitly using flash_attention_2 and it's not available
        if attn_implementation == "flash_attention_2":
            console.warning(f"Flash Attention 2 not available ({e}), using eager attention")
            policy = model_class.from_pretrained(
                model_cfg.base_model,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        else:
            raise  # Re-raise if it's a different error

    # Move model to device
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)  # type: ignore

    # Enable gradient checkpointing for memory efficiency
    if model_cfg.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        console.info_log("Gradient checkpointing enabled for memory efficiency")

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        task_type="CAUSAL_LM",
        bias="none",
        target_modules=list(model_cfg.target_modules),
    )
    policy.config.use_cache = False
    policy = get_peft_model(policy, lora_config)

    # Wrap with DDP if in distributed mode
    world_size = get_world_size()
    if world_size > 1:
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        console.info_log("Wrapped model (find_unused_parameters=True)")

    return policy


def load_models(config: Config) -> tuple[Any, Any, Any]:
    """Load all models.

    Returns:
        Tuple of (processor, policy, reference_model)
    """
    processor = load_processor(config)
    policy = load_policy_model(config)

    # Reference model is not used in this implementation
    reference_model = None

    return processor, policy, reference_model
