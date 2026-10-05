"""Metrics used by the ESM language-model objective."""

import math

import torch
import torch.distributed as dist


@torch.no_grad()
def calculate_bpb_score(next_token_indices, per_token_loss, token_bytes):
    """Return bits-per-byte, total nats, and total token bytes."""

    if (next_token_indices < 0).any():
        valid = next_token_indices >= 0
        safe_indices = torch.where(
            valid, next_token_indices, torch.zeros_like(next_token_indices)
        )
        num_bytes = torch.where(
            valid,
            token_bytes[safe_indices],
            torch.zeros_like(next_token_indices, dtype=token_bytes.dtype),
        )
    else:
        num_bytes = token_bytes[next_token_indices]

    total_nats = (per_token_loss * (num_bytes > 0)).sum()
    total_bytes = num_bytes.sum().to(torch.int64)
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_nats_value = total_nats.item()
    total_bytes_value = total_bytes.item()
    bpb = (
        total_nats_value / (math.log(2) * total_bytes_value)
        if total_bytes_value > 0
        else float("inf")
    )
    return bpb, total_nats_value, total_bytes_value
