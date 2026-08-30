"""ADE / FDE / FIOU as reported in the TITAN paper.

Boxes are the paper's parameterisation x = {c_u, c_v, l_u, l_v}: centre in
pixels plus box width and height. Errors are reported in pixels at the native
1920x1200 resolution, so anything normalised has to be scaled back before it
gets here.
"""

from __future__ import annotations

import torch


def _check(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.shape[-1] < 2:
        raise ValueError("last dim must hold at least (c_u, c_v)")


def displacement_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ADE and FDE in pixels.

    pred/target: (..., T, >=2) with centre coords first.
    mask: (..., T) bool, True where the timestep is a real observation.
    """
    _check(pred, target)
    dist = torch.linalg.vector_norm(pred[..., :2] - target[..., :2], dim=-1)

    if mask is None:
        ade = dist.mean()
        fde = dist[..., -1].mean()
        return ade, fde

    mask = mask.to(dist.dtype)
    denom = mask.sum().clamp_min(1.0)
    ade = (dist * mask).sum() / denom

    # FDE is the error at each track's *last valid* step, which is not always
    # index -1 once tracks drop out mid-horizon.
    idx = _last_valid_index(mask)
    fde_per_track = torch.gather(dist, -1, idx.unsqueeze(-1)).squeeze(-1)
    has_any = mask.sum(-1) > 0
    fde = fde_per_track[has_any].mean() if has_any.any() else dist.new_zeros(())
    return ade, fde


def _last_valid_index(mask: torch.Tensor) -> torch.Tensor:
    steps = torch.arange(mask.shape[-1], device=mask.device)
    shape = [1] * (mask.dim() - 1) + [mask.shape[-1]]
    steps = steps.view(shape).expand_as(mask)
    return torch.where(mask > 0, steps, torch.zeros_like(steps)).max(dim=-1).values


def to_corners(box: torch.Tensor) -> torch.Tensor:
    """(c_u, c_v, l_u, l_v) -> (u1, v1, u2, v2)."""
    cu, cv, lu, lv = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return torch.stack([cu - lu / 2, cv - lv / 2, cu + lu / 2, cv + lv / 2], dim=-1)


def box_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    a, b = to_corners(pred), to_corners(target)
    iu1 = torch.maximum(a[..., 0], b[..., 0])
    iv1 = torch.maximum(a[..., 1], b[..., 1])
    iu2 = torch.minimum(a[..., 2], b[..., 2])
    iv2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (iu2 - iu1).clamp_min(0) * (iv2 - iv1).clamp_min(0)
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_b = (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)
    union = (area_a + area_b - inter).clamp_min(1e-6)
    return inter / union


def final_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """FIOU: IoU between the predicted and true box at the final step."""
    if pred.shape[-1] < 4:
        raise ValueError("FIOU needs the full (c_u, c_v, l_u, l_v) box")
    if mask is None:
        return box_iou(pred[..., -1, :], target[..., -1, :]).mean()

    mask = mask.to(pred.dtype)
    idx = _last_valid_index(mask)
    gather_idx = idx.view(*idx.shape, 1, 1).expand(*idx.shape, 1, pred.shape[-1])
    p = torch.gather(pred, -2, gather_idx).squeeze(-2)
    t = torch.gather(target, -2, gather_idx).squeeze(-2)
    iou = box_iou(p, t)
    has_any = mask.sum(-1) > 0
    return iou[has_any].mean() if has_any.any() else pred.new_zeros(())


class ForecastMetrics:
    """Accumulates pixel-space errors over a whole evaluation pass."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._dist_sum = 0.0
        self._dist_count = 0
        self._fde_sum = 0.0
        self._iou_sum = 0.0
        self._track_count = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        _check(pred, target)
        pred = pred.detach().float()
        target = target.detach().float()
        dist = torch.linalg.vector_norm(pred[..., :2] - target[..., :2], dim=-1)

        if mask is None:
            mask = torch.ones_like(dist, dtype=torch.bool)
        m = mask.to(dist.dtype)

        self._dist_sum += float((dist * m).sum())
        self._dist_count += int(m.sum())

        idx = _last_valid_index(m)
        fde_per_track = torch.gather(dist, -1, idx.unsqueeze(-1)).squeeze(-1)
        valid = m.sum(-1) > 0
        n = int(valid.sum())
        if n:
            self._fde_sum += float(fde_per_track[valid].sum())
            if pred.shape[-1] >= 4:
                gather_idx = idx.view(*idx.shape, 1, 1).expand(*idx.shape, 1, pred.shape[-1])
                p = torch.gather(pred, -2, gather_idx).squeeze(-2)
                t = torch.gather(target, -2, gather_idx).squeeze(-2)
                self._iou_sum += float(box_iou(p, t)[valid].sum())
            self._track_count += n

    def compute(self) -> dict[str, float]:
        tracks = max(self._track_count, 1)
        return {
            "ADE": self._dist_sum / max(self._dist_count, 1),
            "FDE": self._fde_sum / tracks,
            "FIOU": self._iou_sum / tracks,
            "num_tracks": float(self._track_count),
        }
