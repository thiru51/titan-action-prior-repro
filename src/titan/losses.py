from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.schema import ACTION_GROUPS


def masked_trajectory_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Smooth-L1 over future boxes, averaged over valid timesteps only.

    Both tensors are in normalised box space. beta=0.01 puts the errors that
    actually matter (the paper's best FDE is ~20 px, i.e. ~0.01 normalised)
    inside the quadratic region, while badly tracked or occluded agents fall in
    the linear region instead of dominating the update.
    """
    per_el = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
    m = mask.unsqueeze(-1).to(per_el.dtype)
    return (per_el * m).sum() / m.sum().clamp_min(1.0).mul(pred.shape[-1])


def ego_motion_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


class UncertaintyWeightedActionLoss(nn.Module):
    """The paper's L_a = sum_i [ ce_i / sigma_i^2 + log sigma_i ].

    Parameterised by log variance for numerical stability, so the term becomes
    ce_i * exp(-s_i) + 0.5 * s_i with s_i = log sigma_i^2.
    """

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: torch.Tensor,
        agent_mask: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        if not logits:
            return torch.zeros((), device=agent_mask.device)

        total = torch.zeros((), device=agent_mask.device)
        for i, group in enumerate(ACTION_GROUPS):
            lg = logits.get(group.key)
            if lg is None:
                continue
            tgt = targets[..., i]
            # -1 marks unlabelled; padded agent slots must not contribute either.
            valid = (tgt >= 0) & agent_mask.bool()
            if not valid.any():
                continue
            ce = F.cross_entropy(lg[valid], tgt[valid])
            total = total + ce * torch.exp(-log_var[i]) + 0.5 * log_var[i]
        return total
