"""ETH/UCY loader, kept separate from the TITAN path.

Why this exists: TITAN itself is access-gated, so nothing in this repo has been
run on the data the paper used. ETH/UCY is public, and it is the benchmark that
TITAN's own two baselines (Social-LSTM, Social-GAN) publish on. Running the
repo's decoder and its metric code here is a check that both are correct on
real human trajectories, on a benchmark whose numbers anyone can look up.

Format: tab-separated `frame_id  ped_id  x  y`, positions in world metres.
Frames step by 10 at 25 fps, so one row per person every 0.4 s (2.5 Hz).
Standard protocol is 8 observed steps (3.2 s) and 12 predicted (4.8 s), with
leave-one-scene-out: train on four scenes, test on the fifth.

This module does NOT touch the TITAN loader, model or config. It is additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# The five leave-one-out folds. Each directory holds train/val/test subdirs
# where `test` is the held-out scene and `train` is the other four.
SCENES: tuple[str, ...] = ("eth", "hotel", "univ", "zara1", "zara2")

OBS_LEN = 8
PRED_LEN = 12
FRAME_HZ = 2.5


@dataclass
class EthUcyConfig:
    root: str = "data/datasets"
    obs_len: int = OBS_LEN
    pred_len: int = PRED_LEN
    # Sliding-window stride, in frames. 1 is the standard setting and gives the
    # densest possible sample set.
    stride: int = 1
    # Windows are kept only if at least this many people are tracked through
    # all 20 steps. The widely-copied Social-GAN loader uses 2 here, which
    # silently drops every window holding a single person; see RESULTS.md.
    min_agents: int = 1


def read_trajectory_file(path: str | Path) -> np.ndarray:
    """-> (N, 4) array of (frame, ped_id, x, y)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace("\t", " ").split()
        rows.append([float(p) for p in parts[:4]])
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def split_dir(cfg: EthUcyConfig, scene: str, split: str) -> Path:
    return Path(cfg.root) / scene / split


def _windows_from_file(
    data: np.ndarray, obs_len: int, pred_len: int, stride: int, min_agents: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut one scene file into fixed-length windows.

    A person is kept in a window only if they are present at every one of the
    obs_len + pred_len steps. Partially tracked people are dropped rather than
    interpolated, which is what the published protocol does.
    """
    seq_len = obs_len + pred_len
    if data.shape[0] == 0:
        return []

    frames = np.unique(data[:, 0])
    # Position in the sorted frame list, not the raw frame number. Recordings
    # contain gaps where nobody is in view; indexing by list position is what
    # the reference loaders do, so it is what makes numbers comparable.
    frame_index = {f: i for i, f in enumerate(frames)}
    rows_by_frame: list[np.ndarray] = [data[data[:, 0] == f] for f in frames]

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(frames) - seq_len + 1, stride):
        chunk = np.concatenate(rows_by_frame[start : start + seq_len], axis=0)
        obs_list, fut_list = [], []
        for ped in np.unique(chunk[:, 1]):
            track = chunk[chunk[:, 1] == ped]
            first = frame_index[track[0, 0]] - start
            last = frame_index[track[-1, 0]] - start + 1
            if last - first != seq_len or track.shape[0] != seq_len:
                continue
            xy = track[:, 2:4]
            obs_list.append(xy[:obs_len])
            fut_list.append(xy[obs_len:])
        if len(obs_list) >= max(min_agents, 1):
            out.append(
                (
                    np.stack(obs_list).astype(np.float32),
                    np.stack(fut_list).astype(np.float32),
                )
            )
    return out


class EthUcyDataset(Dataset):
    """One item is one time window holding every fully-tracked person in it.

    Keeping a window together (rather than flattening to one person per item)
    is what lets a social-pooling model see a person's neighbours.
    """

    is_synthetic = False

    def __init__(self, cfg: EthUcyConfig, scene: str, split: str) -> None:
        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene!r}; expected one of {SCENES}")
        directory = split_dir(cfg, scene, split)
        files = sorted(directory.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"no .txt trajectory files under {directory}")

        self.cfg = cfg
        self.scene = scene
        self.split = split
        self.files = [str(f) for f in files]

        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        for f in files:
            self.windows.extend(
                _windows_from_file(
                    read_trajectory_file(f),
                    cfg.obs_len,
                    cfg.pred_len,
                    cfg.stride,
                    cfg.min_agents,
                )
            )

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def num_people(self) -> int:
        """Total person-windows, which is what ADE/FDE average over."""
        return sum(w[0].shape[0] for w in self.windows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        obs, fut = self.windows[i]
        return {"obs": torch.from_numpy(obs), "fut": torch.from_numpy(fut)}


def ethucy_collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate people across windows and record where each window starts.

    Padding to a fixed agent count would either waste memory on the crowded
    univ scene or silently drop people from it, so windows are concatenated and
    `seq_start_end` marks the boundaries instead.
    """
    obs = torch.cat([it["obs"] for it in items], dim=0)
    fut = torch.cat([it["fut"] for it in items], dim=0)
    counts = [it["obs"].shape[0] for it in items]
    ends = torch.tensor(counts).cumsum(0)
    starts = ends - torch.tensor(counts)
    return {
        "obs": obs,
        "fut": fut,
        "seq_start_end": torch.stack([starts, ends], dim=1),
    }
