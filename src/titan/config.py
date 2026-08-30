from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .data import schema


@dataclass
class PriorConfig:
    """The paper's ablation switches: ego, interaction and action priors."""

    ego: bool = True
    interaction: bool = True
    action: bool = True

    @property
    def tag(self) -> str:
        parts = [n for n, on in (("EP", self.ego), ("IP", self.interaction), ("AP", self.action)) if on]
        return "+".join(parts) if parts else "vanilla"

    @classmethod
    def from_tag(cls, tag: str) -> "PriorConfig":
        t = tag.strip().lower()
        if t in ("vanilla", "none", ""):
            return cls(False, False, False)
        parts = {p.strip() for p in t.split("+")}
        known = {"ep", "ip", "ap"}
        unknown = parts - known
        if unknown:
            raise ValueError(f"unknown prior(s) {sorted(unknown)} in tag {tag!r}")
        return cls("ep" in parts, "ip" in parts, "ap" in parts)


@dataclass
class DataConfig:
    root: str = "data/titan"
    annotations_subdir: str = "titan_0_4"
    images_subdir: str = "images_anonymized"
    imu_subdir: str = "imu_data"
    splits_subdir: str = "splits"
    obs_len: int = schema.OBS_LEN
    pred_len: int = schema.PRED_LEN
    max_agents: int = 16
    clip_frames: int = 16
    crop_size: int = 112
    load_video: bool = True
    strict_split_sizes: bool = True
    num_workers: int | str = "auto"
    prefetch_factor: int = 4


@dataclass
class ModelConfig:
    traj_hidden: int = 128
    ego_hidden: int = 64
    action_feat_dim: int = 128
    interaction_dim: int = 128
    decoder_hidden: int = 128
    dropout: float = 0.1
    action_backbone: str = "r3d_18"
    pretrained_backbone: bool = True
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int | str = "auto"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    action_loss_weight: float = 1.0
    ego_loss_weight: float = 1.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    compile: bool = False
    out_dir: str = "checkpoints"
    log_every: int = 20


@dataclass
class SyntheticConfig:
    """Config for the clearly-fake data used only to smoke-test the pipeline."""

    enabled: bool = False
    num_clips: int = 12
    agents_per_clip: int = 6
    seed: int = 0


@dataclass
class Config:
    priors: PriorConfig = field(default_factory=PriorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        sections = {
            "priors": PriorConfig,
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "synthetic": SyntheticConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            kwargs[name] = klass(**(raw.get(name) or {}))
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
