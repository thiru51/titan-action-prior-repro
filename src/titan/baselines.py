"""Constant-velocity baseline.

The paper's weakest reference row (102.47 px FDE) is a constant-velocity
extrapolation. It needs no training, so it is the one number in the whole table
this repo can compute honestly the moment real data arrives -- which makes it
the natural first sanity check that the eval protocol is wired up correctly.

The paper reports it both with and without box scaling; both are here because
they share an ADE and differ only in FIOU, which is a useful check that the
box-size channels are being handled the way the paper handled them.
"""

from __future__ import annotations

import torch


def constant_velocity(
    obs_boxes: torch.Tensor, pred_len: int, scale_boxes: bool = False
) -> torch.Tensor:
    """obs_boxes: (B, A, T, 4) -> (B, A, pred_len, 4).

    Velocity is the last observed step difference. With scale_boxes the box
    extent is extrapolated too; otherwise it is frozen at the last observation.
    """
    if obs_boxes.shape[-2] < 2:
        raise ValueError("need at least two observed steps to estimate velocity")

    last = obs_boxes[..., -1, :]
    vel = obs_boxes[..., -1, :] - obs_boxes[..., -2, :]
    if not scale_boxes:
        vel = vel.clone()
        vel[..., 2:] = 0

    steps = torch.arange(1, pred_len + 1, device=obs_boxes.device, dtype=obs_boxes.dtype)
    return last.unsqueeze(-2) + vel.unsqueeze(-2) * steps.view(-1, 1)


def linear_least_squares(obs: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Least-squares straight-line extrapolation. obs: (..., T, D) -> (..., pred_len, D).

    Social-GAN's "Linear" row fits its line by minimising squared error over the
    whole observed window, not by differencing the last two steps. Both are
    "linear", they are not the same estimator, so both are here and results are
    labelled with which one produced them.
    """
    t = obs.shape[-2]
    if t < 2:
        raise ValueError("need at least two observed steps to fit a line")

    steps = torch.arange(t, device=obs.device, dtype=obs.dtype)
    mean_t = steps.mean()
    centred = steps - mean_t
    denom = (centred * centred).sum().clamp_min(1e-12)

    mean_xy = obs.mean(dim=-2, keepdim=True)
    slope = (centred.unsqueeze(-1) * (obs - mean_xy)).sum(dim=-2) / denom
    intercept = mean_xy.squeeze(-2) - slope * mean_t

    future_t = torch.arange(t, t + pred_len, device=obs.device, dtype=obs.dtype)
    return intercept.unsqueeze(-2) + slope.unsqueeze(-2) * future_t.view(-1, 1)
