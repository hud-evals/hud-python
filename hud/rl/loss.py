import torch
import torch.nn.functional as F

from hud.rl.config import LossConfig


def entropy_from_logits(
    logits: torch.Tensor,  # (B, seq, vocab)
    mask: torch.Tensor,    # (B, seq)
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    - Flattens all leading dims: (N, V)
    - Selects indices where mask is True
    - Processes selected rows in chunks with log_softmax for numerical stability
    - Returns a 1D tensor of entropies corresponding to masked rows (in
      the same order as a flattened mask would visit them)
    """
    num_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, num_classes)
    flat_mask = mask.reshape(-1)
    idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return torch.empty(0, device=logits.device, dtype=logits.dtype)

    entropies: list[torch.Tensor] = []
    for start in range(0, idx.numel(), chunk_size):
        sel = idx[start:start + chunk_size]
        rows = flat_logits.index_select(0, sel)
        logps = F.log_softmax(rows.float(), dim=-1)
        chunk_entropy = -(logps.exp() * logps).sum(dim=-1)
        entropies.append(chunk_entropy)

    return torch.cat(entropies, dim=0).to(logits.dtype)

def get_per_token_logps(
    logits: torch.Tensor,  # (B, seq, vocab)
    input_ids: torch.Tensor,  # (B, seq)
) -> torch.Tensor:
    """
    The less-efficient selective log softmax but stable for bfloat16.
    Assumes logits are already shifted and scaled by temperature.
    """
    per_token_logps: list[torch.Tensor] = []
    for row_logits, row_input_ids in zip(logits, input_ids):  # (seq, vocab)
        row_logprobs = F.log_softmax(row_logits, dim=-1)
        row_per_token = row_logprobs.gather(dim=-1, index=row_input_ids.unsqueeze(-1)).squeeze(-1)
        per_token_logps.append(row_per_token)
    return torch.stack(per_token_logps)


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
        seq_log_ratio = (log_ratio * assistant_mask).sum(dim=-1)  # (B,)
        # GSPO normalizes the importance ratio by the sequence length
        if config.apply_length_norm:
            seq_log_ratio = seq_log_ratio / torch.clamp_min(assistant_mask.sum(dim=-1), 1)
        seq_log_ratio = torch.clamp(seq_log_ratio, max=10.0)
        log_ratio = seq_log_ratio.unsqueeze(-1).expand_as(log_ratio)  # (B, seq_len)

    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, max=config.clip_ratio)

    loss = - clipped_ratio * advantages.unsqueeze(-1)

    clipped = (ratio > config.clip_ratio).float()

    loss = (loss * assistant_mask).sum(dim=-1)  # (B,)

    if config.importance_sampling_level == "sequence":
        loss = loss / torch.clamp_min(assistant_mask.sum(dim=-1), 1)

    loss = loss.sum() / max(loss_norm, 1)

    out_tensors: dict[str, torch.Tensor] = {
        "importance_ratio": ratio.detach(),
        "clipped_importance_ratio": clipped.detach(),
        "is_clipped": (ratio > config.clip_ratio).float().detach(),
    }

    if ref_log_probs is not None:
        log_rho = (log_probs - ref_log_probs).detach()
        rho = torch.exp(log_rho.clamp(-20.0, 20.0))
        kl_approx = rho - torch.log(rho) - 1
        out_tensors["kl"] = kl_approx.detach()

    return loss, out_tensors
