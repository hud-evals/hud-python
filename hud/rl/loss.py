import torch
import torch.nn.functional as F

from hud.rl.config import LossConfig


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(log_probs * logits, dim=-1)

    return entropy


def get_per_token_logps(
    logits: torch.Tensor,  # (B, seq, vocab)
    input_ids: torch.Tensor,  # (B, seq)
    temperature: torch.Tensor,  # (B,)
) -> torch.Tensor:
    # Shift logits
    logits = logits[:, :-1, :]  # (B, seq-1, vocab)
    logits = torch.cat([torch.zeros_like(logits[:, :1, :]), logits], dim=1)  # (B, seq, vocab)

    # Scale by temperature
    logits = logits / temperature.view(-1, 1, 1)

    # Compute log probabilities for input tokens
    # The less-efficient selective log softmax but stable for bfloat16
    per_token_logps = []
    for row_logits, row_input_ids in zip(logits, input_ids):
        row_logprobs = F.log_softmax(row_logits, dim=-1)
        row_per_token_logps = row_logprobs.gather(dim=-1, index=row_input_ids.unsqueeze(-1)).squeeze(-1)
        per_token_logps.append(row_per_token_logps)
    per_token_logps = torch.stack(per_token_logps)

    return per_token_logps


def compute_loss(
    log_probs: torch.Tensor, # B, seq
    old_log_probs: torch.Tensor, # B, seq
    ref_log_probs: torch.Tensor | None, # B, seq
    advantages: torch.Tensor, # (B,) 
    assistant_mask: torch.Tensor, # (B, seq)
    config: LossConfig,
    loss_norm: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute loss for a batch of samples, use per-token loss."""
    log_ratio = log_probs - old_log_probs

    if config.importance_sampling_level == "sequence":
        seq_log_ratio = (log_ratio[assistant_mask]).sum()
        # GSPO normalizes the importance ratio by the sequence length
        if config.apply_length_norm:
            seq_log_ratio = seq_log_ratio / torch.clamp_min(assistant_mask.sum(), 1)
        log_ratio = torch.clamp(seq_log_ratio.unsqueeze(0), max=10.0)

    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, max=config.clip_ratio)

    loss = - clipped_ratio * advantages

    clipped = (ratio > config.clip_ratio).float()

    loss = (loss[assistant_mask]).sum()

    if config.importance_sampling_level == "sequence":
        loss = loss / torch.clamp(assistant_mask.sum(), 1)

    loss = loss / max(loss_norm, 1)

    return loss, {
        "loss": loss,
        "ratio": ratio,
        "clipped": clipped,
    }