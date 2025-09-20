from typing import Any

import torch
import torch.nn.functional as F

from hud.rl.logger import console
from hud.rl.utils import get_gpu_utilization, get_memory_usage


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits in a memory-efficient way."""
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
    return entropy


def compute_logprobs(
    model: Any, inputs: Any, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked per-token log probabilities via the model.

    Returns log probabilities for the actual next tokens.
    """
    try:
        model_inputs = {k: v for k, v in inputs.items() if k != "assistant_mask"}
        out = model(**model_inputs)

        logits = out.logits / temperature
        log_probs = F.log_softmax(logits, dim=-1)

        targets = inputs["input_ids"][:, 1:]
        token_log_probs = log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Compute entropy only for assistant tokens to save memory
        assistant_mask = inputs["assistant_mask"]
        entropy = torch.zeros_like(token_log_probs)
        if assistant_mask.any():
            entropy[assistant_mask] = entropy_from_logits(logits[:, :-1][assistant_mask])

        return token_log_probs, entropy
    except (IndexError, RuntimeError) as e:
        # Handle empty inputs or DDP errors
        console.warning_log(f"Error in compute_logprobs: {e}. Returning dummy values.")
        # Return dummy values that match expected shapes
        seq_len = inputs["input_ids"].shape[1] - 1 if "input_ids" in inputs else 0
        batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
        device = inputs["input_ids"].device if "input_ids" in inputs else torch.device("cpu")
        dummy_logprobs = torch.zeros(batch_size, seq_len, device=device)
        dummy_entropy = torch.zeros(batch_size, seq_len, device=device)
        return dummy_logprobs, dummy_entropy


def compute_grpo_loss(
    sample: Any,
    pol_logp: torch.Tensor,
    pol_entropy: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
    config: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss for a batch of samples.

    Args:
        sample: Training sample with inputs and advantage
        pol_logp: Policy log probabilities
        pol_entropy: Policy entropy
        old_logp: Old policy log probabilities
        ref_logp: Reference policy log probabilities
        config: Training configuration

    Returns:
        Tuple of (loss tensor, metrics dictionary)
    """
    training_cfg = config.training

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
        torch.clamp(ratio_tok, 1 - training_cfg.top_eps, 1 + training_cfg.bottom_eps) * advantage
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
        total_loss = total_loss.mean() if training_cfg.token_agg == "mean" else total_loss.sum()
    else:
        if training_cfg.token_agg == "mean":
            total_loss = (total_loss * m).sum() / m.sum().clamp_min(1.0)
        else:
            total_loss = (total_loss * m).sum()

    # Compute metrics only over masked (assistant) tokens
    mask_count = m.sum().clamp_min(1.0)
    metrics_dict = {
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
        "gpu_util": get_gpu_utilization(),
        "gpu_memory": get_memory_usage(),
    }

    return total_loss, metrics_dict


def sanity_check(
    sample: Any,
    pol_logp: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
) -> None:
    """Sanity check for loss computation inputs.

    Args:
        sample: Training sample
        pol_logp: Policy log probabilities
        old_logp: Old policy log probabilities
        ref_logp: Reference policy log probabilities
    """
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
                console.warning_log(f"assistant_mask overlaps padding: {bad} tokens")

        # Finiteness on masked entries only
        def _stats(name: str, t: torch.Tensor) -> None:
            sel = t[m]
            if sel.numel() == 0:
                console.warning_log(f"{name} empty under mask")
                return
            finite = torch.isfinite(sel)
            if finite.sum() < sel.numel():
                console.warning_log(f"{name} non-finite: {((~finite).sum().item())}/{sel.numel()}")
            sel = sel[finite].float()

        _stats("pol_logp", pol_logp)
        _stats("old_logp", old_logp)
        _stats("ref_logp", ref_logp)

        # Log-probabilities should be <= 0 (log-softmax)
        if (pol_logp[m] > 1e-6).any():
            console.warning_log("pol_logp has positive values under mask")

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
