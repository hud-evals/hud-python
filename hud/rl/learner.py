"""GRPO learner for vision-language and text models."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

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

try:
    import bitsandbytes as bnb  # type: ignore

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from contextlib import nullcontext

from hud.rl.distributed import (
    get_local_rank,
    get_world_size,
    is_main_process,
)
from hud.rl.utils import (
    batch_training_samples,
    entropy_from_logits,
    get_gpu_utilization,
    get_memory_usage,
    prepare_inputs,
)
from hud.utils.hud_console import HUDConsole

from .types import TrainingMetrics, TrainingSample

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)

if TYPE_CHECKING:
    from .config import Config


class GRPOLearner:
    """GRPO learning algorithm for Vision-Language Models (VLMs) and Text Models."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Detect model type
        self.is_vl_model = "VL" in config.model.base_model

        # Load models and processor
        self.processor, self.policy, self.ref, self.optimizer = self._load_models()
        self.metrics: list[TrainingMetrics] = []

    def log(self, message: str) -> None:
        hud_console.info_log(f"[{self.local_rank}] {message}")

    def _load_models(self) -> tuple[Any, Any, Any, Any]:
        """Load policy, reference models and optimizer."""
        model_cfg = self.config.model

        # Detect if this is a VL model or standard text model
        is_vl_model = "VL" in model_cfg.base_model
        model_type = "Vision-Language" if is_vl_model else "Text"
        self.log(f"Loading {model_type} model: {model_cfg.base_model}")

        # Apply Liger kernel optimizations if available and enabled
        if model_cfg.use_liger and LIGER_AVAILABLE:
            if is_vl_model:
                self.log("Applying Liger kernel optimizations to Qwen2.5-VL")
                apply_liger_kernel_to_qwen2_5_vl(
                    rope=True,  # Optimized RoPE
                    rms_norm=True,  # Optimized RMSNorm
                    swiglu=True,  # Optimized SwiGLU
                    fused_linear_cross_entropy=True,  # Fused Linear+CrossEntropy for memory
                )
        elif model_cfg.use_liger and not LIGER_AVAILABLE:
            self.log(
                "Liger kernel requested but not installed. Install with: pip install liger-kernel"
            )

        # Load processor/tokenizer based on model type
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

        # Load policy model with LoRA
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
            self.log(f"Using {attn_implementation} for attention")
        except (ImportError, ValueError) as e:
            # Only fallback if explicitly using flash_attention_2 and it's not available
            if attn_implementation == "flash_attention_2":
                self.log(f"Flash Attention 2 not available ({e}), using eager attention")
                policy = model_class.from_pretrained(
                    model_cfg.base_model,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
            else:
                raise  # Re-raise if it's a different error

        # Move model to device
        policy = policy.to(self.device)  # type: ignore
        # Enable gradient checkpointing for memory efficiency
        if model_cfg.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            self.log("Gradient checkpointing enabled for memory efficiency")

        # Add LoRA adapters or load existing adapter
        policy.config.use_cache = False

        if model_cfg.adapter_path:
            # Load existing adapter as baseline
            self.log(f"Loading existing LoRA adapter from: {model_cfg.adapter_path}")
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, model_cfg.adapter_path)
            # Enable adapter training
            policy.train()
        else:
            # Create new LoRA adapter
            lora_config = LoraConfig(
                r=model_cfg.lora_r,
                lora_alpha=model_cfg.lora_alpha,
                lora_dropout=model_cfg.lora_dropout,
                task_type="CAUSAL_LM",
                bias="none",
                target_modules=list(model_cfg.target_modules),
            )
            policy = get_peft_model(policy, lora_config)

        # Wrap with DDP if in distributed mode
        if self.world_size > 1:
            policy = DDP(
                policy,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            self.log("Wrapped model (find_unused_parameters=True)")

        # Create optimizer - need to access underlying model if DDP
        base_model = policy.module if hasattr(policy, "module") else policy
        trainable_params = [p for _, p in base_model.named_parameters() if p.requires_grad]  # type: ignore

        # Use 8-bit optimizer if configured
        if self.config.training.use_8bit_optimizer and BNB_AVAILABLE:
            hud_console.info("Using 8-bit AdamW optimizer from bitsandbytes")
            optimizer = bnb.optim.AdamW8bit(  # type: ignore
                trainable_params,
                lr=self.config.training.lr,
                betas=self.config.training.adam_betas,
                eps=self.config.training.adam_eps,
            )
        else:
            self.log("Using standard FP32 AdamW optimizer")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.lr,
                betas=self.config.training.adam_betas,
                eps=self.config.training.adam_eps,
            )

        # Log optimizer info
        self.log(f"Optimizer: {type(optimizer).__name__}")
        num_params = sum(p.numel() for p in trainable_params)
        self.log(f"Number of trainable parameters: {num_params:,}")

        return processor, policy, None, optimizer

    def prepare_groups(
        self,
        samples: list[TrainingSample],
    ) -> list[list[TrainingSample]]:
        """Prepare groups of samples for training."""
        # Prepare inputs with messages
        batch = []
        for sample in samples:
            inputs = prepare_inputs(sample, self.processor)
            # If inputs are invalid, create dummy inputs to maintain batch size
            if (
                not inputs
                or "input_ids" not in inputs
                or inputs.get("input_ids", torch.tensor([])).numel() == 0
            ):
                hud_console.warning_log("Sample has invalid inputs, using dummy values")
                # Create minimal dummy inputs to keep batch size consistent
                inputs = {
                    "input_ids": torch.zeros(1, 2, dtype=torch.long),  # Minimal sequence
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "assistant_mask": torch.zeros(1, 1, dtype=torch.bool),  # T-1 length
                }
            elif "assistant_mask" not in inputs:
                hud_console.warning_log("Sample missing assistant_mask, creating zero mask")
                seq_len = inputs["input_ids"].shape[-1]
                inputs["assistant_mask"] = torch.zeros(
                    inputs["input_ids"].shape[0], seq_len - 1, dtype=torch.bool
                )

            new_sample = TrainingSample(**sample.model_dump())
            new_sample.inputs = inputs
            new_sample.advantage = sample.advantage
            batch.append(new_sample)

        with hud_console.progress("Processing batch of traces...") as progress, torch.no_grad():
            for i, sample in enumerate(batch):
                if is_main_process():
                    progress.update(f"Processing batch of traces... {i}/{len(batch)}")
                if sample.inputs:
                    sample = sample.to_device(self.device)
                    sample.old_logprobs, _ = self.compute_logprobs(self.policy, sample.inputs)
                    # Free GPU memory for this sample immediately
                    sample.to_device(torch.device("cpu"))

            policy_module = self.policy.module if hasattr(self.policy, "module") else self.policy
            with policy_module.disable_adapter():
                for i, sample in enumerate(batch):
                    if is_main_process():
                        progress.update(f"Processing batch of traces... {i}/{len(batch)}")
                    if sample.inputs:
                        # Move back to GPU for reference computation, then free
                        sample = sample.to_device(self.device)
                        sample.ref_logprobs, _ = self.compute_logprobs(self.policy, sample.inputs)
                        sample.to_device(torch.device("cpu"))

        hud_console.info_log("Creating mini-batches...")
        group_size = self.config.training.group_size
        processed_batch = []
        if not self.config.training.accumulate_over_minibatches:
            # Find minibatches and group them via batch_training_samples
            # Minibatches control the batch size of the forward pass to the model
            mb_size = self.config.training.mini_batch_size
            group_size = group_size // mb_size
            for i in range(0, len(batch), mb_size):
                processed_batch.extend(batch_training_samples(batch[i : i + mb_size]))
        else:
            processed_batch = batch

        for sample in processed_batch:
            sample.to_device(torch.device("cpu"))

        # Convert to grouped batches (if updating the model after each task group)
        if self.config.training.update_after_group:
            return [
                processed_batch[i : i + group_size]
                for i in range(0, len(processed_batch), group_size)
            ]
        else:
            return [processed_batch]

    def update(self, samples: list[TrainingSample]) -> TrainingMetrics:
        """Perform a gradient update on a batch."""
        import time

        training_start_time = time.time()

        # Always create metrics for synchronization
        self.metrics.append(TrainingMetrics())
        metrics = self.metrics[-1]

        # Prepare groups for GRPO training
        groups = self.prepare_groups(samples)
        self.log(f"Updating over {len(groups)} groups")

        # Update over mini batch size
        with hud_console.progress("Gradient update...") as progress:
            for epoch in range(self.config.training.epochs):  # Do not accumulate across epochs
                progress.update(f"Training epoch {epoch + 1}/{self.config.training.epochs}")
                for group_idx, group in enumerate(groups):  # Do not accumulate across "groups"
                    self.optimizer.zero_grad(set_to_none=True)

                    debug_per_group = ""
                    grad_accum_steps = len(group)
                    # Tensor for distributed sync
                    global_skip = torch.zeros(1, device=self.device)

                    for s_idx, sample_minibatch in enumerate(group):
                        # self.log(f"{group_idx} {sample_minibatch.inputs['assistant_mask'].sum()}")
                        # mini_updated = sample_minibatch.inputs["assistant_mask"].sum() > 0

                        # Update mini_updated globally
                        # self.log(f"{group_idx} Mini updated: {mini_updated}")

                        # Do not sync until the last minibatch
                        if s_idx < len(group) - 1 and self.world_size > 1:
                            ddp_ctx = self.policy.no_sync()
                        else:
                            ddp_ctx = nullcontext()

                        with ddp_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            try:
                                # if mini_updated:
                                loss = self.compute_loss(sample_minibatch) / grad_accum_steps
                                debug_per_group += f"l{s_idx}:{round(loss.item(), 3)!s} "
                                loss.backward()
                                # else: # Dummy backward that touches all params, produces zero g
                                #     dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
                                #     debug_per_group += f"d{s_idx}:{str(round(dummy.item(), 3))} "
                                #     dummy.backward()
                                # self.log(f"{group_idx} GPU Backward: {get_gpu_utilization():.1f}% | Memory: {get_memory_usage():.2f} GB") # noqa: E501
                            except torch.cuda.OutOfMemoryError:
                                hud_console.warning_log(
                                    f"{group_idx} CUDA OOM for {sample_minibatch.inputs['input_ids'].numel()} tokens; skipping minibatch"  # noqa: E501
                                )
                                # Dummy backward to keep DDP happy
                                dummy = torch.sum(p.sum() for p in self.policy.parameters()) * 0.0  # type: ignore
                                debug_per_group += f"o{s_idx}:{round(dummy.item(), 3)!s} "
                                dummy.backward()
                                # mark global skip if OOM
                                global_skip.fill_(1)
                                continue

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # After minibatches loop, sync skip across ranks
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_skip, op=torch.distributed.ReduceOp.MAX)
                    skip_any = bool(global_skip.item())

                    if skip_any:
                        self.log(f"G[{group_idx}] {debug_per_group} N/A (skipped)")
                        continue

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.training.grad_clip,
                        error_if_nonfinite=True,
                    )
                    self.optimizer.step()

                    debug_per_group += f"g:{round(grad_norm.item(), 3)!s}"
                    self.log(f"G[{group_idx}] {debug_per_group}")

                    metrics.update(
                        {
                            "grad_norm": grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else float(grad_norm),
                        }
                    )

        # Calculate training time and throughput
        training_time = time.time() - training_start_time
        total_samples = (
            len(groups) * self.config.training.group_size * self.config.training.mini_batch_size
        )
        samples_per_second = total_samples / training_time if training_time > 0 else 0.0

        metrics.update(
            {
                "training_time": training_time,
                "samples_per_second": samples_per_second,
            }
        )

        return metrics

    def compute_loss(self, sample: TrainingSample) -> torch.Tensor:
        """Compute GRPO loss for a batch of samples."""
        training_cfg = self.config.training
        metrics = self.metrics[-1] if len(self.metrics) > 0 else TrainingMetrics()

        sample.to_device(self.device)

        pol_logp, pol_entropy = self.compute_logprobs(
            self.policy,
            sample.inputs,
        )

        sanity_check(sample, pol_logp, sample.old_logprobs, sample.ref_logprobs)

        metrics.update(
            {
                "gpu_util": get_gpu_utilization(),  # Track peak utilization
                "gpu_memory": get_memory_usage(),  # Track memory usage
            }
        )
        self.log(f"GPU Util: {get_gpu_utilization():.1f}% | Memory: {get_memory_usage():.2f} GB")

        old_logp = sample.old_logprobs
        ref_logp = sample.ref_logprobs

        if old_logp is None or ref_logp is None or sample.advantage is None:
            raise ValueError("old_logp, ref_logp, or sample.advantage is None")

        # Use assistant mask to remove non-assistant tokens
        m = sample.inputs["assistant_mask"]

        # Aggregate per trace or per token
        if training_cfg.ppo_mode == "per_trace":
            counts = m.sum(dim=1).clamp_min(1.0)
            pol_logp = (pol_logp * m.float()).sum(dim=1) / counts
            pol_entropy = (pol_entropy * m.float()).sum(dim=1) / counts
            old_logp = (old_logp * m.float()).sum(dim=1) / counts
            ref_logp = (ref_logp * m.float()).sum(dim=1) / counts

        # Clip log probability differences
        log_ratio = torch.where(m, pol_logp - old_logp, torch.zeros_like(pol_logp))
        ratio_tok = torch.exp(log_ratio.clamp(-20.0, 20.0))

        # Ensure advantage shape matches ratio_tok for broadcasting
        advantage = (
            sample.advantage.view(-1, 1) if ratio_tok.dim() == 2 else sample.advantage.squeeze(-1)
        )

        unclipped = ratio_tok * advantage
        clipped = (
            torch.clamp(ratio_tok, 1 - training_cfg.top_eps, 1 + training_cfg.bottom_eps)
            * advantage
        )

        policy_term = -torch.minimum(unclipped, clipped)

        # Clip log probability differences in KL
        log_rho = torch.where(m, pol_logp - ref_logp, torch.zeros_like(pol_logp))
        rho_tok = torch.exp(log_rho.clamp(-20.0, 20.0))
        kl_approx = rho_tok - torch.log(rho_tok) - 1

        total_loss = (
            policy_term + training_cfg.kl_beta * kl_approx + training_cfg.entropy_beta * pol_entropy
        )

        # Aggregate loss
        if training_cfg.ppo_mode == "per_trace":
            total_loss = total_loss.mean() if training_cfg.token_agg == "mean" else total_loss.sum()  # noqa: S105
        else:
            if training_cfg.token_agg == "mean":  # noqa: S105
                total_loss = (total_loss * m).sum() / m.sum().clamp_min(1.0)
            else:
                total_loss = (total_loss * m).sum()

        # Compute metrics only over masked (assistant) tokens
        mask_count = m.sum().clamp_min(1.0)
        metrics.update(
            {
                "policy_ratio": (ratio_tok * m).sum().item() / mask_count.item()
                if mask_count.item() > 0
                else 1.0,
                "kl": (kl_approx * m).sum().item() / mask_count.item()
                if mask_count.item() > 0
                else 0.0,
                "entropy": (pol_entropy * m).sum().item() / mask_count.item()
                if mask_count.item() > 0
                else 0.0,
                "tokens": sample.inputs["input_ids"].numel(),
                "loss": total_loss.item(),
            }
        )

        sample.to_device(torch.device("cpu"))

        return total_loss

    def compute_logprobs(self, model: Any, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked per-token log probabilities via the model.

        Returns log probabilities for the actual next tokens.
        """
        try:
            model_inputs = {k: v for k, v in inputs.items() if k != "assistant_mask"}
            out = model(**model_inputs)

            logits = out.logits / self.config.actor.temperature

            targets = inputs["input_ids"][:, 1:]

            # Align logits to predict next token: use logits[:, :-1, :]
            next_logits = logits[:, :-1, :]

            token_log_probs = _selective_log_softmax(next_logits, targets)

            # Compute entropy only for assistant tokens to save memory
            assistant_mask = inputs["assistant_mask"]
            entropy = torch.zeros_like(token_log_probs)
            if assistant_mask.any():
                entropy[assistant_mask] = entropy_from_logits(logits[:, :-1][assistant_mask])

            return token_log_probs, entropy
        except (IndexError, RuntimeError) as e:
            # Handle empty inputs or DDP errors
            hud_console.warning_log(f"Error in compute_logprobs: {e}. Returning dummy values.")
            # Return dummy values that match expected shapes
            seq_len = inputs["input_ids"].shape[1] - 1 if "input_ids" in inputs else 0
            batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
            # Create dummy tensors that still participate in autograd so backward doesn't fail
            try:
                # Touch params to build a graph
                param_sum = torch.sum(next(self.policy.parameters()))
                base = param_sum * 0.0
            except StopIteration:
                base = torch.tensor(0.0, device=self.device)
            dummy_logprobs = (
                base + torch.zeros(batch_size, seq_len, device=self.device)
            ).requires_grad_(True)
            dummy_entropy = (
                base + torch.zeros(batch_size, seq_len, device=self.device)
            ).requires_grad_(True)
            return dummy_logprobs, dummy_entropy

    def save(self, path: str) -> None:
        """Save the current policy checkpoint (only on rank 0)."""
        if is_main_process():
            os.makedirs(path, exist_ok=True)
            # Unwrap DDP model if needed
            model_to_save = self.policy.module if hasattr(self.policy, "module") else self.policy
            model_to_save.save_pretrained(path)
            self.log(f"Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load a policy checkpoint."""
        # Would need to reload LoRA weights
        self.log(f"Loading checkpoint from {path}")
        # Implementation depends on PEFT version


def sanity_check(
    sample: TrainingSample,
    pol_logp: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
) -> None:
    assert "assistant_mask" in sample.inputs
    m = sample.inputs["assistant_mask"]
    if old_logp is None or ref_logp is None:
        return
    with torch.no_grad():
        B, K = pol_logp.shape
        assert old_logp.shape == (B, K), "old_logp shape mismatch"
        assert ref_logp.shape == (B, K), "ref_logp shape mismatch"
        assert m.shape == (B, K), "assistant_mask shape mismatch"

        # Check mask is subset of attention_mask[:, 1:]
        att = sample.inputs.get("attention_mask", None)
        if att is not None and att.dim() == 2:
            att_shift = att[:, 1:].bool()
            bad = (m & ~att_shift).sum().item()
            if bad > 0:
                hud_console.warning_log(f"assistant_mask overlaps padding: {bad} tokens")

        # Finiteness on masked entries only
        def _stats(name: str, t: torch.Tensor) -> None:
            sel = t[m]
            if sel.numel() == 0:
                hud_console.warning_log(f"{name} empty under mask")
                return
            finite = torch.isfinite(sel)
            if finite.sum() < sel.numel():
                hud_console.warning_log(
                    f"{name} non-finite: {((~finite).sum().item())}/{sel.numel()}"
                )
            sel = sel[finite].float()

        _stats("pol_logp", pol_logp)
        _stats("old_logp", old_logp)
        _stats("ref_logp", ref_logp)

        # Log-probabilities should be <= 0 (log-softmax)
        if (pol_logp[m] > 1e-6).any():
            hud_console.warning_log("pol_logp has positive values under mask")

        # Precompute masked deltas and ratios for diagnostics (before exp)
        masked_log_ratio = torch.zeros_like(pol_logp)
        masked_log_ratio[m] = (pol_logp - old_logp)[m]
        masked_log_rho = torch.zeros_like(pol_logp)
        masked_log_rho[m] = (pol_logp - ref_logp)[m]

        _stats("log_ratio(masked)", masked_log_ratio)
        _stats("log_rho(masked)", masked_log_rho)

        # Ratios after clamp (diagnostic only)
        ratio_diag = torch.zeros_like(pol_logp)
        rho_diag = torch.zeros_like(pol_logp)
        ratio_diag[m] = torch.exp(masked_log_ratio[m].clamp(-20.0, 20.0))
        rho_diag[m] = torch.exp(masked_log_rho[m].clamp(-20.0, 20.0))
        _stats("ratio_tok(masked)", ratio_diag)
        _stats("rho_tok(masked)", rho_diag)


def _selective_log_softmax(
    logits_bt_v: torch.Tensor,
    index_bt: torch.Tensor,
) -> torch.Tensor:
    """Gather log softmax for selected indices with reduced peak memory.

    Uses logsumexp subtraction for float32/64; falls back to per-row
    log_softmax for bf16/fp16.
    logits_bt_v: [B, T, V]
    index_bt:    [B, T]
    Returns:     [B, T]
    """
    if logits_bt_v.dtype in (torch.float32, torch.float64):
        # Compute logsumexp per [B, T] in a loop over batch to reduce
        # peak from B*T*V to T*V
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits_bt_v])
        selected_logits = torch.gather(logits_bt_v, dim=-1, index=index_bt.unsqueeze(-1)).squeeze(
            -1
        )
        return selected_logits - logsumexp_values
    # Reduced precision: numerically stable route using per-row log_softmax
    token_logprobs_rows: list[torch.Tensor] = []
    for logits_row, index_row in zip(logits_bt_v, index_bt, strict=True):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_rows.append(
            torch.gather(logprobs_row, dim=-1, index=index_row.unsqueeze(-1)).squeeze(-1)
        )
    return torch.stack(token_logprobs_rows)
