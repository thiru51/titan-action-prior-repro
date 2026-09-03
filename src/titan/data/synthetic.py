"""SYNTHETIC data. Not TITAN. Not real.

This exists for exactly one purpose: proving the pipeline runs end to end
(loader -> action branch -> interaction -> AIM -> GRU decoder -> FDE) before
real TITAN access is granted. Any loss or FDE measured on this is a number
about random walks with noise, not about pedestrians, and must never be
reported as a reproduction of the paper.

Trajectories are constant-velocity walks with a per-agent turn rate plus
Gaussian noise, over the same box parameterisation and pixel range as TITAN.
Tubes are noise, so the action branch produces a well-formed but meaningless
signal -- which is what a plumbing test needs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import schema
from .common import normalize_boxes

BANNER = "SYNTHETIC DATA -- pipeline smoke test only, results are meaningless"


class SyntheticTitanDataset(Dataset):
    is_synthetic = True

    def __init__(self, cfg, split: str, synth_cfg, load_video: bool = True) -> None:
        self.obs_len = cfg.obs_len
        self.pred_len = cfg.pred_len
        self.seq_len = cfg.obs_len + cfg.pred_len
        self.max_agents = cfg.max_agents
        self.clip_frames = cfg.clip_frames
        self.crop_size = cfg.crop_size
        self.load_video = load_video
        self.agents_per_clip = min(synth_cfg.agents_per_clip, cfg.max_agents)

        # Distinct offsets so train/val/test are different draws, and a fixed
        # base seed so a smoke test is reproducible run to run.
        offset = {"train": 0, "val": 10_000, "test": 20_000}.get(split, 0)
        self.base_seed = synth_cfg.seed + offset
        self.length = synth_cfg.num_clips

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + idx)
        A, T = self.max_agents, self.seq_len
        n = self.agents_per_clip

        boxes = np.zeros((A, T, 4), dtype=np.float32)
        agent_mask = np.zeros((A,), dtype=bool)
        valid = np.zeros((A, T), dtype=bool)

        for a in range(n):
            cu = rng.uniform(200, schema.IMAGE_WIDTH - 200)
            cv = rng.uniform(300, schema.IMAGE_HEIGHT - 200)
            speed = rng.uniform(2.0, 14.0)
            heading = rng.uniform(0, 2 * np.pi)
            turn = rng.normal(0, 0.03)
            lu = rng.uniform(30, 90)
            lv = lu * rng.uniform(1.8, 2.6)

            for t in range(T):
                boxes[a, t] = (cu, cv, lu, lv)
                heading += turn
                cu += speed * np.cos(heading) + rng.normal(0, 0.6)
                cv += speed * np.sin(heading) * 0.35 + rng.normal(0, 0.4)
                # Scale grows as the agent nears the camera; keeps the box-size
                # channels from being trivially constant.
                lu *= 1.0 + rng.normal(0.002, 0.004)
                lv *= 1.0 + rng.normal(0.002, 0.004)

            agent_mask[a] = True
            valid[a] = True

        boxes[..., 0] = np.clip(boxes[..., 0], 0, schema.IMAGE_WIDTH)
        boxes[..., 1] = np.clip(boxes[..., 1], 0, schema.IMAGE_HEIGHT)

        ego = np.stack(
            [rng.normal(0, 1.2, size=T), rng.normal(0, 0.08, size=T)], axis=1
        ).astype(np.float32)

        actions = np.full((A, len(schema.ACTION_GROUPS)), -1, dtype=np.int64)
        for a in range(n):
            for g, group in enumerate(schema.ACTION_GROUPS):
                actions[a, g] = rng.integers(0, group.num_classes)

        sample = {
            "obs_boxes": normalize_boxes(torch.from_numpy(boxes[:, : self.obs_len])),
            "fut_boxes": torch.from_numpy(boxes[:, self.obs_len :]),
            "valid_fut": torch.from_numpy(valid[:, self.obs_len :]),
            "agent_mask": torch.from_numpy(agent_mask),
            "actions": torch.from_numpy(actions),
            "obs_ego": torch.from_numpy(ego[: self.obs_len]),
            "fut_ego": torch.from_numpy(ego[self.obs_len :]),
        }
        if self.load_video:
            tubes = torch.zeros(A, 3, self.clip_frames, self.crop_size, self.crop_size)
            g = torch.Generator().manual_seed(int(self.base_seed + idx))
            tubes[:n] = torch.randn(
                n, 3, self.clip_frames, self.crop_size, self.crop_size, generator=g
            ) * 0.5
            sample["tubes"] = tubes
        return sample
