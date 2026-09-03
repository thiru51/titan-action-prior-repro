"""Loader for the released TITAN annotations.

Expected layout under data.root, matching the tarballs HRI distributes:

    <root>/titan_0_4/clip_<n>.csv
    <root>/images_anonymized/clip_<n>/images/<frame>.png
    <root>/imu_data/clip_<n>/synced_sensors.csv
    <root>/splits/{train,val,test}_set.txt

This has been written against the published schema but has NOT been run against
the real tarballs -- dataset access is still pending. Treat the first real run
as a debugging session, not a training run.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import schema
from .common import ltwh_to_cxcywh, normalize_boxes


@dataclass
class Track:
    track_id: int
    frames: np.ndarray
    boxes: np.ndarray            # (T, 4) as (c_u, c_v, l_u, l_v) in pixels
    actions: np.ndarray          # (T, num_action_groups) int, -1 = unlabelled


class TitanForecastDataset(Dataset):
    def __init__(self, cfg, split: str, transform=None) -> None:
        import pandas as pd

        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.root = Path(cfg.root)
        self.obs_len = cfg.obs_len
        self.pred_len = cfg.pred_len
        self.seq_len = cfg.obs_len + cfg.pred_len
        self.max_agents = cfg.max_agents
        self._pd = pd

        self.clips = self._read_split(split)
        self.tracks: dict[str, list[Track]] = {}
        self.ego: dict[str, np.ndarray] = {}
        self.windows: list[tuple[str, int]] = []

        for clip in self.clips:
            csv = self.root / cfg.annotations_subdir / f"{clip}.csv"
            if not csv.exists():
                continue
            tracks = self._read_clip(csv)
            if not tracks:
                continue
            self.tracks[clip] = tracks
            self.ego[clip] = self._read_imu(clip)
            self.windows.extend((clip, t) for t in self._window_starts(tracks))

        if not self.windows:
            raise FileNotFoundError(
                f"no usable TITAN windows found under {self.root}. "
                "Dataset access must be granted and the tarballs extracted first; "
                "use synthetic.enabled=true to exercise the pipeline without data."
            )

    def _read_split(self, split: str) -> list[str]:
        fname = schema.SPLIT_FILES[split]
        path = self.root / self.cfg.splits_subdir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing split file {path}")
        clips = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        expected = schema.EXPECTED_SPLIT_SIZES[split]
        if self.cfg.strict_split_sizes and len(clips) != expected:
            raise ValueError(
                f"{split} split has {len(clips)} clips, the release has {expected}; "
                "the download is probably incomplete (set strict_split_sizes=false to override)"
            )
        return clips

    def _read_clip(self, csv: Path) -> list[Track]:
        df = self._pd.read_csv(csv)
        cols = schema.resolve_columns(list(df.columns))
        required = ("frames", "label", "obj_track_id", "top", "left", "height", "width")
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"{csv.name} is missing expected column(s): {missing}")

        df = df[df[cols["label"]].astype(str).str.strip().str.lower() == schema.PERSON_LABEL]
        if df.empty:
            return []

        # 'frames' is an image filename like 00000123.png in the released CSVs.
        frame_idx = (
            df[cols["frames"]].astype(str).str.split(".").str[0].str.extract(r"(\d+)")[0].astype(int)
        )
        df = df.assign(_frame=frame_idx)

        tracks: list[Track] = []
        for tid, grp in df.groupby(df[cols["obj_track_id"]].astype(int)):
            grp = grp.sort_values("_frame")
            boxes = np.stack(
                [
                    np.asarray(
                        ltwh_to_cxcywh(t, l, h, w), dtype=np.float32
                    )
                    for t, l, h, w in zip(
                        grp[cols["top"]].astype(float),
                        grp[cols["left"]].astype(float),
                        grp[cols["height"]].astype(float),
                        grp[cols["width"]].astype(float),
                    )
                ]
            )
            actions = np.stack(
                [self._encode_actions(row, cols) for _, row in grp.iterrows()]
            ) if len(grp) else np.zeros((0, len(schema.ACTION_GROUPS)), dtype=np.int64)
            tracks.append(
                Track(int(tid), grp["_frame"].to_numpy(dtype=np.int64), boxes, actions)
            )
        return tracks

    @staticmethod
    def _encode_actions(row, cols) -> np.ndarray:
        vals = []
        for group in schema.ACTION_GROUPS:
            col = cols.get(group.column)
            vals.append(schema.index_of(group, row[col]) if col is not None else -1)
        return np.asarray(vals, dtype=np.int64)

    def _read_imu(self, clip: str) -> np.ndarray:
        """Returns (num_frames, 2) of (longitudinal accel, yaw rate)."""
        path = self.root / self.cfg.imu_subdir / clip / "synced_sensors.csv"
        if not path.exists():
            return np.zeros((0, 2), dtype=np.float32)
        df = self._pd.read_csv(path)
        cols = {c.strip().lower(): c for c in df.columns}

        def pick(*names):
            for n in names:
                if n in cols:
                    return df[cols[n]].to_numpy(dtype=np.float32)
            return None

        accel = pick("accel_x", "accel x", "ax")
        yaw = pick("ang_vel_z", "gyro_z", "yaw_rate", "ang vel z")
        if accel is None or yaw is None:
            # Column naming in the IMU dump is the part of the schema documented
            # least precisely; fall back to positional rather than crash.
            num = df.select_dtypes("number").to_numpy(dtype=np.float32)
            if num.shape[1] < 2:
                return np.zeros((0, 2), dtype=np.float32)
            accel, yaw = num[:, 0], num[:, -1]
        n = min(len(accel), len(yaw))
        return np.stack([accel[:n], yaw[:n]], axis=1)

    def _window_starts(self, tracks: list[Track]) -> list[int]:
        starts: set[int] = set()
        for tr in tracks:
            f = tr.frames
            for i in range(len(f) - self.seq_len + 1):
                # Only contiguous stretches at the annotation rate are usable.
                if f[i + self.seq_len - 1] - f[i] == self.seq_len - 1:
                    starts.add(int(f[i]))
        return sorted(starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip, start = self.windows[idx]
        tracks = self.tracks[clip]
        frames = np.arange(start, start + self.seq_len)

        A = self.max_agents
        boxes = np.zeros((A, self.seq_len, 4), dtype=np.float32)
        valid = np.zeros((A, self.seq_len), dtype=bool)
        actions = np.full((A, len(schema.ACTION_GROUPS)), -1, dtype=np.int64)
        agent_mask = np.zeros((A,), dtype=bool)

        slot = 0
        for tr in tracks:
            if slot >= A:
                break
            pos = self._align(tr.frames, frames)
            if pos is None:
                continue
            boxes[slot] = tr.boxes[pos]
            valid[slot] = True
            agent_mask[slot] = True
            # Action prior is read at the last observed frame: at inference time
            # nothing past the observation window is available.
            actions[slot] = tr.actions[pos[self.obs_len - 1]]
            slot += 1

        ego = self._ego_window(clip, start)

        sample = {
            "obs_boxes": normalize_boxes(torch.from_numpy(boxes[:, : self.obs_len])),
            "fut_boxes": torch.from_numpy(boxes[:, self.obs_len :]),
            "valid_fut": torch.from_numpy(valid[:, self.obs_len :]),
            "agent_mask": torch.from_numpy(agent_mask),
            "actions": torch.from_numpy(actions),
            "obs_ego": torch.from_numpy(ego[: self.obs_len]),
            "fut_ego": torch.from_numpy(ego[self.obs_len :]),
        }
        if self.cfg.load_video:
            sample["tubes"] = self._load_tubes(clip, frames[: self.obs_len], boxes, agent_mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def _align(track_frames: np.ndarray, want: np.ndarray) -> np.ndarray | None:
        """Indices into the track covering `want`, or None if it does not."""
        lo = bisect.bisect_left(track_frames.tolist(), int(want[0]))
        hi = lo + len(want)
        if hi > len(track_frames):
            return None
        idx = np.arange(lo, hi)
        if not np.array_equal(track_frames[idx], want):
            return None
        return idx

    def _ego_window(self, clip: str, start: int) -> np.ndarray:
        ego = self.ego.get(clip)
        out = np.zeros((self.seq_len, 2), dtype=np.float32)
        if ego is None or len(ego) == 0:
            return out
        end = min(start + self.seq_len, len(ego))
        if start < len(ego):
            chunk = ego[start:end]
            out[: len(chunk)] = chunk
        return out

    def _load_tubes(
        self, clip: str, frames: np.ndarray, boxes: np.ndarray, agent_mask: np.ndarray
    ) -> torch.Tensor:
        from .video import crop_agent_tubes

        img_dir = self.root / self.cfg.images_subdir / clip / "images"
        return crop_agent_tubes(
            img_dir, frames, boxes[:, : self.obs_len], agent_mask,
            num_frames=self.cfg.clip_frames, size=self.cfg.crop_size,
        )
