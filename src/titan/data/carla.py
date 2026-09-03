"""CARLA loader: turns recorded simulator episodes into forecasting windows.

Why this exists. TITAN's central claim is that an **action prior** helps
short-horizon pedestrian forecasting more than an ego-motion prior or an
interaction prior does. TITAN itself is access-gated, so that claim cannot be
retested on TITAN. CARLA is a driving simulator, so its labels are exact rather
than hand-annotated: whether a pedestrian is standing on a driving lane is a
map query, not an annotator's opinion. That makes it a place where each prior
can be switched off cleanly and the ordering measured.

**What this is not.** These are simulated pedestrians. Nothing measured here is
an accuracy on real people, and nothing here is a TITAN number. The episodes
were recorded by a separate project (`pedestrian-intent-forecast`, its
`scripts/collect_carla.py`); the JSON files are copied into `data/carla_run/`
so this repo runs standalone. No code is imported across the two projects.

**The honest limit on the action prior.** A CARLA walker has no gaze target, no
hands to wave and no phone. TITAN's action prior covers five label groups, and
two of them -- communicative and transportive -- simply do not exist in a
simulator recording. What CARLA does know exactly is the *contextual* group:
where the pedestrian is with respect to the road. So the action prior here is
**restricted to the contextual group**, and every result must be read as a test
of that weaker claim. See RESULTS.md.

Recording format, one JSON per episode:

    fps                20.0, a fixed 0.05 s simulator step
    width / height     1280 x 720 camera image
    ego                per frame: OBD_speed, speed_mps, action
    walkers            per track: frames, bbox (x1,y1,x2,y2), visible_fraction,
                       cross ("crossing" / "not-crossing")
    traffic            parked vehicles, same frames/bbox layout

Two fields the recordings do NOT have, which restrict the other two priors just
as much as the missing gesture labels restrict the action prior:

    no ego pose        `ego` holds forward speed only. There is no position and
                       no yaw rate, so the ego prior here is the longitudinal
                       speed profile and nothing else.
    almost no crowd    every episode holds exactly one walker, and only some
                       hold a single *parked* (stationary) vehicle. The
                       interaction prior therefore has next to nothing to
                       encode. RESULTS.md says so plainly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Written into every recording by the collector. A file without it is from an
# older collector and is rejected rather than silently misread.
FORMAT = "pif-carla-recording"
FORMAT_VERSION = 1

RECORDED_HZ = 20.0

# Per-frame `cross` string, exactly as the collector writes it.
CROSSING = "crossing"

# The restricted action prior: the contextual group only, as a causal state
# machine over the per-frame "is this pedestrian on a driving lane" map query.
# Causal matters. A label like "waiting to cross" that is assigned by looking
# at whether the person crosses *later* would leak the future trajectory into
# the observation window, which is the very thing being predicted.
ACTION_APPROACHING = 0  # off the road, has not been on it yet
ACTION_CROSSING = 1  # on a driving lane right now
ACTION_CROSSED = 2  # off the road again, having been on it earlier
ACTION_PAD = 3  # padded agent slot, no label

CONTEXTUAL_ACTIONS: tuple[str, ...] = ("approaching", "crossing", "crossed", "pad")
NUM_CONTEXTUAL_ACTIONS = len(CONTEXTUAL_ACTIONS)

# Agent classes. The interaction prior needs to tell a person from a car.
AGENT_PERSON = 0
AGENT_VEHICLE = 1


@dataclass
class CarlaConfig:
    root: str = "data/carla_run"
    # Keep the native 20 Hz. Subsampling to TITAN's 10 Hz and then reusing its
    # 10/20 step counts would ask for a 3 s span, and a scripted crossing
    # episode is only 5-10 s long with the crossing near the end, so 99.7% of
    # the windows would sit in the approach phase and the action label would be
    # a constant. See RESULTS.md for the measured numbers behind that choice.
    subsample: int = 1
    obs_len: int = 10
    pred_len: int = 20
    # Sliding-window stride in *subsampled* steps. Windows from one episode
    # overlap, which is fine because the train/test split is by episode. At
    # 20 Hz a stride of 1 puts windows 0.05 s apart, which is near-duplicate
    # work; 2 is 0.1 s apart.
    stride: int = 2
    max_agents: int = 4
    # A box touching the image border has been clipped by the camera, so its
    # centre no longer tracks the pedestrian. Those frames are dropped.
    border_margin: float = 0.5
    # Frames where the ray-cast visibility is exactly zero are dropped: the
    # collector still writes a projected box, but nothing is actually visible.
    require_visible: bool = True


def _validated(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    if raw.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    version = int(raw.get("version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} is recording format v{version}, this loader reads v{FORMAT_VERSION}"
        )
    fps = float(raw.get("fps", 0.0))
    if abs(fps - RECORDED_HZ) > 1e-6:
        raise ValueError(f"{path} was captured at {fps} Hz, this loader assumes {RECORDED_HZ}")
    return raw


def episode_ids(root: str | Path) -> list[str]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"no CARLA recordings directory at {directory}")
    ids = sorted(p.stem for p in directory.glob("*.json"))
    if not ids:
        raise FileNotFoundError(f"no *.json recordings under {directory}")
    return ids


def read_episode(root: str | Path, episode_id: str) -> dict[str, Any]:
    path = Path(root) / f"{episode_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no CARLA recording at {path}")
    return _validated(json.loads(path.read_text()), path)


def to_centre_box(corners: list[float]) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) -> TITAN's (c_u, c_v, l_u, l_v)."""
    x1, y1, x2, y2 = corners
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def contextual_actions(cross: list[str]) -> np.ndarray:
    """Per-frame `cross` strings -> the restricted contextual action label.

    Causal: the label at step t depends only on steps 0..t.
    """
    out = np.empty(len(cross), dtype=np.int64)
    been_on_road = False
    for i, value in enumerate(cross):
        if value == CROSSING:
            been_on_road = True
            out[i] = ACTION_CROSSING
        else:
            out[i] = ACTION_CROSSED if been_on_road else ACTION_APPROACHING
    return out


def _usable(box: list[float], visible: float, width: int, height: int, cfg: CarlaConfig) -> bool:
    m = cfg.border_margin
    inside = box[0] > m and box[1] > m and box[2] < width - m and box[3] < height - m
    return inside and (visible > 0.0 or not cfg.require_visible)


@dataclass
class Track:
    """One agent's frames, boxes and action labels, indexed by frame number."""

    agent_class: int
    boxes: dict[int, tuple[float, float, float, float]]
    actions: dict[int, int]


def _walker_track(entry: dict[str, Any], width: int, height: int, cfg: CarlaConfig) -> Track:
    actions = contextual_actions([str(c) for c in entry["cross"]])
    visible = entry.get("visible_fraction") or [1.0] * len(entry["frames"])
    boxes: dict[int, tuple[float, float, float, float]] = {}
    labels: dict[int, int] = {}
    for i, frame in enumerate(entry["frames"]):
        box = [float(v) for v in entry["bbox"][i]]
        if not _usable(box, float(visible[i]), width, height, cfg):
            continue
        boxes[int(frame)] = to_centre_box(box)
        labels[int(frame)] = int(actions[i])
    return Track(AGENT_PERSON, boxes, labels)


def _traffic_track(entry: dict[str, Any], width: int, height: int, cfg: CarlaConfig) -> Track:
    boxes: dict[int, tuple[float, float, float, float]] = {}
    labels: dict[int, int] = {}
    for i, frame in enumerate(entry.get("frames", [])):
        box = [float(v) for v in entry["bbox"][i]]
        if not _usable(box, 1.0, width, height, cfg):
            continue
        boxes[int(frame)] = to_centre_box(box)
        # A parked car is not a pedestrian and has no contextual action. It is
        # given the pad class so the action embedding never invents one.
        labels[int(frame)] = ACTION_PAD
    return Track(AGENT_VEHICLE, boxes, labels)


def ego_signal(ego: dict[str, Any], frames: list[int]) -> np.ndarray:
    """(T, 2) = forward speed and its step-to-step change, in m/s and m/s/step.

    TITAN's ego prior is (longitudinal acceleration, yaw rate) from vehicle IMU.
    A CARLA recording has neither: it stores forward speed only. Acceleration is
    recovered by differencing; yaw rate cannot be recovered at all, so the ego
    prior here is longitudinal motion and nothing more.
    """
    speed = np.array(
        [float((ego.get(str(f)) or {}).get("speed_mps", 0.0)) for f in frames],
        dtype=np.float32,
    )
    accel = np.zeros_like(speed)
    accel[1:] = speed[1:] - speed[:-1]
    return np.stack([speed, accel], axis=-1)


@dataclass
class Window:
    episode_id: str
    start_frame: int
    obs_boxes: np.ndarray  # (A, obs_len, 4) pixels
    fut_boxes: np.ndarray  # (A, pred_len, 4) pixels
    obs_actions: np.ndarray  # (A, obs_len) int64
    agent_class: np.ndarray  # (A,) int64
    agent_mask: np.ndarray  # (A,) bool
    target_mask: np.ndarray  # (A,) bool, True for agents ADE/FDE is scored on
    obs_ego: np.ndarray  # (obs_len, 2)


def windows_from_episode(
    raw: dict[str, Any], episode_id: str, cfg: CarlaConfig
) -> list[Window]:
    """Cut one episode into fixed-length windows.

    An agent is kept in a window only if it has a usable box at every one of
    the obs_len + pred_len subsampled steps. Partial tracks are dropped rather
    than interpolated, matching the ETH/UCY path in this repo.
    """
    width, height = int(raw["width"]), int(raw["height"])
    seq_len = cfg.obs_len + cfg.pred_len

    people = [_walker_track(w, width, height, cfg) for w in raw.get("walkers") or []]
    vehicles = [_traffic_track(t, width, height, cfg) for t in raw.get("traffic") or []]
    if not people:
        return []

    # The subsampling grid is anchored at frame 0 with a single phase. Using
    # both phases of a 20 Hz recording would double the window count with
    # near-duplicates that carry almost no new information.
    grid = list(range(0, int(raw["num_frames"]), cfg.subsample))
    ego = raw.get("ego") or {}

    out: list[Window] = []
    for i in range(0, len(grid) - seq_len + 1, cfg.stride):
        frames = grid[i : i + seq_len]
        present = [t for t in people if all(f in t.boxes for f in frames)]
        if not present:
            continue
        # Vehicles are context for the interaction prior, never a target, so a
        # vehicle only has to be present for the frames the model observes.
        obs_frames = frames[: cfg.obs_len]
        context = [t for t in vehicles if all(f in t.boxes for f in obs_frames)]

        agents = present + context
        if len(agents) > cfg.max_agents:
            agents = agents[: cfg.max_agents]
            present = [t for t in agents if t.agent_class == AGENT_PERSON]

        a = cfg.max_agents
        obs_boxes = np.zeros((a, cfg.obs_len, 4), dtype=np.float32)
        fut_boxes = np.zeros((a, cfg.pred_len, 4), dtype=np.float32)
        obs_actions = np.full((a, cfg.obs_len), ACTION_PAD, dtype=np.int64)
        agent_class = np.zeros(a, dtype=np.int64)
        agent_mask = np.zeros(a, dtype=bool)
        target_mask = np.zeros(a, dtype=bool)

        for k, track in enumerate(agents):
            obs_boxes[k] = np.array([track.boxes[f] for f in obs_frames], dtype=np.float32)
            obs_actions[k] = np.array([track.actions[f] for f in obs_frames], dtype=np.int64)
            agent_class[k] = track.agent_class
            agent_mask[k] = True
            if track.agent_class == AGENT_PERSON:
                fut_boxes[k] = np.array(
                    [track.boxes[f] for f in frames[cfg.obs_len :]], dtype=np.float32
                )
                target_mask[k] = True
            else:
                # Held at the last observed box so the padded target is never
                # a wild value; it is masked out of the loss and the metric.
                fut_boxes[k] = obs_boxes[k, -1]

        out.append(
            Window(
                episode_id=episode_id,
                start_frame=frames[0],
                obs_boxes=obs_boxes,
                fut_boxes=fut_boxes,
                obs_actions=obs_actions,
                agent_class=agent_class,
                agent_mask=agent_mask,
                target_mask=target_mask,
                obs_ego=ego_signal(ego, obs_frames),
            )
        )
    return out


class CarlaDataset(Dataset):
    """Windows drawn from a fixed list of episodes.

    The episode list is the unit of splitting. Windows inside one episode share
    frames, so splitting by window would put near-identical samples on both
    sides of the split; `assert_disjoint_episodes` below is the guard.
    """

    is_synthetic = True  # simulator data, not real pedestrians

    def __init__(self, cfg: CarlaConfig, episodes: list[str]) -> None:
        self.cfg = cfg
        self.episodes = list(episodes)
        self.windows: list[Window] = []
        for episode_id in self.episodes:
            raw = read_episode(cfg.root, episode_id)
            self.windows.extend(windows_from_episode(raw, episode_id, cfg))

        # Pixel scale, applied once. The network sees boxes in [0, 1]; ADE and
        # FDE are reported back in pixels at the recorded 1280x720.
        first = read_episode(cfg.root, self.episodes[0]) if self.episodes else {}
        self.width = int(first.get("width", 1280))
        self.height = int(first.get("height", 720))
        self._scale = np.array(
            [self.width, self.height, self.width, self.height], dtype=np.float32
        )

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def scale(self) -> torch.Tensor:
        return torch.from_numpy(self._scale)

    @property
    def episodes_with_windows(self) -> set[str]:
        return {w.episode_id for w in self.windows}

    @property
    def num_targets(self) -> int:
        """Pedestrian-windows, which is what ADE/FDE average over."""
        return int(sum(int(w.target_mask.sum()) for w in self.windows))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        w = self.windows[i]
        return {
            "obs_boxes": torch.from_numpy(w.obs_boxes / self._scale),
            "fut_boxes": torch.from_numpy(w.fut_boxes / self._scale),
            "obs_actions": torch.from_numpy(w.obs_actions),
            "agent_class": torch.from_numpy(w.agent_class),
            "agent_mask": torch.from_numpy(w.agent_mask),
            "target_mask": torch.from_numpy(w.target_mask),
            "obs_ego": torch.from_numpy(w.obs_ego),
        }


def carla_collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def assert_disjoint_episodes(*splits: list[str]) -> None:
    """Fail loudly if any episode appears in more than one split.

    Windows from one episode overlap in time, so an episode landing on both
    sides of a split leaks almost-identical samples into the test set. This is
    the same check the recording project makes when it builds its own splits.
    """
    seen: dict[str, int] = {}
    for i, split in enumerate(splits):
        if len(set(split)) != len(split):
            raise ValueError(f"split {i} lists an episode twice")
        for episode_id in split:
            if episode_id in seen:
                raise ValueError(
                    f"episode {episode_id} appears in split {seen[episode_id]} and split {i}"
                )
            seen[episode_id] = i


def usable_episodes(cfg: CarlaConfig) -> list[str]:
    """Episodes that yield at least one complete window under `cfg`."""
    keep = []
    for episode_id in episode_ids(cfg.root):
        raw = read_episode(cfg.root, episode_id)
        if windows_from_episode(raw, episode_id, cfg):
            keep.append(episode_id)
    return keep


def scenario_of(cfg: CarlaConfig, episode_id: str) -> str:
    return str(read_episode(cfg.root, episode_id).get("scenario", "unknown"))


def episode_folds(cfg: CarlaConfig, episodes: list[str], num_folds: int) -> list[list[str]]:
    """Split episodes into folds, balanced across the recorded scenario types.

    Scenarios differ a lot in how the pedestrian moves (`no_cross` never enters
    the road, `hesitate` stops at the kerb), so dealing them round-robin inside
    each scenario keeps every fold representative instead of leaving one fold
    made mostly of one scenario.
    """
    if num_folds < 2:
        raise ValueError("need at least two folds")
    by_scenario: dict[str, list[str]] = {}
    for episode_id in episodes:
        by_scenario.setdefault(scenario_of(cfg, episode_id), []).append(episode_id)

    folds: list[list[str]] = [[] for _ in range(num_folds)]
    for scenario in sorted(by_scenario):
        for i, episode_id in enumerate(sorted(by_scenario[scenario])):
            folds[i % num_folds].append(episode_id)
    return [sorted(f) for f in folds]
