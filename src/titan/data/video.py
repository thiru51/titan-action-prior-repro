"""Per-agent tube cropping for the action branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Kinetics statistics that torchvision's video weights were trained with.
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)

_EXTS = (".png", ".jpg", ".jpeg")


def _find_frame(img_dir: Path, frame: int) -> Path | None:
    for width in (10, 8, 6, 5, 4, 0):
        stem = str(frame).zfill(width) if width else str(frame)
        for ext in _EXTS:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def _read_image(path: Path) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def crop_agent_tubes(
    img_dir: Path,
    frames: np.ndarray,
    boxes: np.ndarray,
    agent_mask: np.ndarray,
    num_frames: int = 16,
    size: int = 112,
    context: float = 1.25,
) -> torch.Tensor:
    """Returns (A, 3, num_frames, size, size), normalised for the video backbone.

    Missing frames yield zero tubes rather than raising: a handful of dropped
    images should not kill a training run.
    """
    A = boxes.shape[0]
    out = torch.zeros(A, 3, num_frames, size, size)

    # Resample the observation window to the clip length the backbone wants.
    picks = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)

    cache: dict[int, torch.Tensor | None] = {}
    for t_out, t_in in enumerate(picks):
        fno = int(frames[t_in])
        if fno not in cache:
            path = _find_frame(img_dir, fno)
            cache[fno] = _read_image(path).float().div_(255.0) if path is not None else None
        img = cache[fno]
        if img is None:
            continue
        _, H, W = img.shape

        for a in range(A):
            if not agent_mask[a]:
                continue
            cu, cv, lu, lv = boxes[a, t_in]
            if lu <= 1 or lv <= 1:
                continue
            half_u, half_v = lu * context / 2, lv * context / 2
            u1 = int(max(0, round(cu - half_u)))
            v1 = int(max(0, round(cv - half_v)))
            u2 = int(min(W, round(cu + half_u)))
            v2 = int(min(H, round(cv + half_v)))
            if u2 <= u1 or v2 <= v1:
                continue
            patch = img[:, v1:v2, u1:u2].unsqueeze(0)
            patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
            out[a, :, t_out] = (patch.squeeze(0) - MEAN) / STD

    return out
