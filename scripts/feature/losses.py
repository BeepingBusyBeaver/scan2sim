from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from scripts.feature.heads import HeadSpec


def _adjacent_soft_targets(target: torch.Tensor, num_classes: int, epsilon: float) -> torch.Tensor:
    # target: [B] long index
    batch_size = int(target.shape[0])
    out = torch.zeros((batch_size, num_classes), dtype=torch.float32, device=target.device)
    if epsilon <= 0.0:
        out.scatter_(1, target.unsqueeze(1), 1.0)
        return out

    for row, cls_idx in enumerate(target.tolist()):
        neighbors = []
        if cls_idx - 1 >= 0:
            neighbors.append(cls_idx - 1)
        if cls_idx + 1 < num_classes:
            neighbors.append(cls_idx + 1)

        if not neighbors:
            out[row, cls_idx] = 1.0
            continue

        neighbor_mass = epsilon / float(len(neighbors))
        out[row, cls_idx] = 1.0 - epsilon
        for neighbor_idx in neighbors:
            out[row, neighbor_idx] = neighbor_mass
    return out


def _soft_cross_entropy(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return -(soft_target * log_prob).sum(dim=-1).mean()


def compute_multitask_loss(
    logits_by_head: Dict[str, torch.Tensor],
    target_indices: torch.Tensor,
    heads: Sequence[HeadSpec],
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    if target_indices.ndim != 2:
        raise ValueError(f"target_indices must be [B,H], got shape={tuple(target_indices.shape)}")
    if target_indices.shape[1] != len(heads):
        raise ValueError("target_indices head dimension mismatch.")

    total_loss: torch.Tensor | None = None
    loss_detail: Dict[str, float] = {}
    acc_detail: Dict[str, float] = {}

    for head_idx, head in enumerate(heads):
        logits = logits_by_head[head.name]
        target = target_indices[:, head_idx]
        if logits.shape[0] != target.shape[0]:
            raise ValueError(f"Batch size mismatch for head '{head.name}'.")

        relax = head.boundary_relax
        if relax.enabled and relax.epsilon > 0.0:
            soft_target = _adjacent_soft_targets(target, logits.shape[1], float(relax.epsilon))
            head_loss = _soft_cross_entropy(logits, soft_target)
        else:
            head_loss = F.cross_entropy(logits, target)

        weighted = head_loss * float(head.weight)
        total_loss = weighted if total_loss is None else (total_loss + weighted)
        loss_detail[head.name] = float(head_loss.detach().cpu().item())

        pred = torch.argmax(logits, dim=1)
        acc = (pred == target).float().mean()
        acc_detail[head.name] = float(acc.detach().cpu().item())

    assert total_loss is not None
    return total_loss, loss_detail, acc_detail

