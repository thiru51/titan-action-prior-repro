from __future__ import annotations

import torch

from .schema import IMAGE_HEIGHT, IMAGE_WIDTH

# Networks train on normalised boxes; every reported number is in pixels at the
# native 1920x1200, so the conversion lives in one place.
_SCALE = torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT], dtype=torch.float32)


def normalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    return boxes / _SCALE.to(boxes.device, boxes.dtype)


def denormalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    return boxes * _SCALE.to(boxes.device, boxes.dtype)


def ltwh_to_cxcywh(top: float, left: float, height: float, width: float) -> tuple[float, float, float, float]:
    """TITAN stores top/left/height/width; the paper models centre + extent."""
    return left + width / 2.0, top + height / 2.0, width, height


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("empty batch")
    out: dict[str, torch.Tensor] = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out
