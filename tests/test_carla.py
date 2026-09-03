"""Tests for the CARLA path: loader, episode splitting and the ablation net.

These do not touch the TITAN path or the ETH/UCY path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from titan.config import PriorConfig
from titan.data import carla as C
from titan.models.carla_net import CarlaTitanNet

REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "carla_run"


def make_recording(
    episode_id: str = "ep_test",
    num_frames: int = 60,
    cross_from: int | None = None,
    cross_until: int | None = None,
    with_vehicle: bool = False,
    clip_from: int | None = None,
) -> dict:
    """A minimal recording in the format the collector writes."""
    frames = list(range(num_frames))
    bbox, cross, visible = [], [], []
    for f in frames:
        x = 100.0 + f
        if clip_from is not None and f >= clip_from:
            x = 0.0  # a box pushed against the image border, i.e. clipped
        bbox.append([x, 200.0, x + 20.0, 260.0])
        visible.append(1.0)
        on_road = cross_from is not None and cross_from <= f < (cross_until or num_frames)
        cross.append("crossing" if on_road else "not-crossing")

    raw = {
        "format": C.FORMAT,
        "version": C.FORMAT_VERSION,
        "episode_id": episode_id,
        "fps": 20.0,
        "width": 1280,
        "height": 720,
        "num_frames": num_frames,
        "scenario": "straight",
        "ego": {str(f): {"speed_mps": float(f) * 0.1} for f in frames},
        "walkers": [
            {
                "ped_id": f"{episode_id}_w0",
                "frames": frames,
                "bbox": bbox,
                "visible_fraction": visible,
                "cross": cross,
            }
        ],
        "traffic": [],
    }
    if with_vehicle:
        raw["traffic"] = [
            {
                "obj_id": f"{episode_id}_parked",
                "obj_class": "vehicle",
                "frames": frames,
                "bbox": [[600.0, 300.0, 700.0, 380.0] for _ in frames],
            }
        ]
    return raw


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for i in range(4):
        raw = make_recording(f"ep_{i:04d}", cross_from=40 if i % 2 else None, with_vehicle=i == 3)
        (tmp_path / f"ep_{i:04d}.json").write_text(json.dumps(raw))
    return tmp_path


def test_contextual_actions_are_causal():
    labels = C.contextual_actions(
        ["not-crossing"] * 3 + ["crossing"] * 2 + ["not-crossing"] * 3
    )
    assert labels.tolist() == [
        C.ACTION_APPROACHING,
        C.ACTION_APPROACHING,
        C.ACTION_APPROACHING,
        C.ACTION_CROSSING,
        C.ACTION_CROSSING,
        C.ACTION_CROSSED,
        C.ACTION_CROSSED,
        C.ACTION_CROSSED,
    ]


def test_contextual_actions_do_not_leak_the_future():
    """A prefix must label identically whether or not a crossing follows it."""
    prefix = ["not-crossing"] * 5
    a = C.contextual_actions(prefix + ["crossing"] * 5)[:5]
    b = C.contextual_actions(prefix + ["not-crossing"] * 5)[:5]
    assert a.tolist() == b.tolist()


def test_to_centre_box():
    assert C.to_centre_box([10.0, 20.0, 30.0, 60.0]) == (20.0, 40.0, 20.0, 40.0)


def test_ego_signal_is_speed_and_its_difference():
    ego = {"0": {"speed_mps": 1.0}, "1": {"speed_mps": 3.0}, "2": {"speed_mps": 4.0}}
    out = C.ego_signal(ego, [0, 1, 2])
    assert out.shape == (3, 2)
    assert out[:, 0].tolist() == [1.0, 3.0, 4.0]
    assert out[:, 1].tolist() == [0.0, 2.0, 1.0]


def test_rejects_a_recording_from_another_format(tmp_path: Path):
    (tmp_path / "bad.json").write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError):
        C.read_episode(tmp_path, "bad")


def test_clipped_boxes_never_enter_a_window(tmp_path: Path):
    raw = make_recording(num_frames=80, clip_from=50)
    cfg = C.CarlaConfig(root=str(tmp_path), obs_len=10, pred_len=20, stride=1)
    windows = C.windows_from_episode(raw, "ep_test", cfg)
    assert windows
    # The last usable frame is 49, so no window may reach past it.
    assert max(w.start_frame + (cfg.obs_len + cfg.pred_len - 1) for w in windows) <= 49


def test_dataset_shapes_and_masks(root: Path):
    cfg = C.CarlaConfig(root=str(root), obs_len=10, pred_len=20, stride=5, max_agents=4)
    ds = C.CarlaDataset(cfg, C.episode_ids(root))
    assert len(ds) > 0
    item = ds[0]
    assert item["obs_boxes"].shape == (4, 10, 4)
    assert item["fut_boxes"].shape == (4, 20, 4)
    assert item["obs_actions"].shape == (4, 10)
    assert item["obs_ego"].shape == (10, 2)
    # Only pedestrians are scored, and every scored agent is a real agent.
    for w in ds.windows:
        assert not (w.target_mask & ~w.agent_mask).any()
        assert (w.agent_class[w.target_mask] == C.AGENT_PERSON).all()
        assert (w.obs_actions[~w.agent_mask] == C.ACTION_PAD).all()
    assert ds.num_targets == sum(int(w.target_mask.sum()) for w in ds.windows)


def test_vehicles_are_context_and_never_targets(root: Path):
    cfg = C.CarlaConfig(root=str(root), obs_len=10, pred_len=20, stride=5)
    ds = C.CarlaDataset(cfg, ["ep_0003"])
    classes = np.concatenate([w.agent_class[w.agent_mask] for w in ds.windows])
    assert (classes == C.AGENT_VEHICLE).any(), "the fixture episode has a parked vehicle"
    for w in ds.windows:
        assert not w.target_mask[w.agent_class == C.AGENT_VEHICLE].any()


def test_windows_carry_their_episode_id(root: Path):
    cfg = C.CarlaConfig(root=str(root), stride=5)
    ds = C.CarlaDataset(cfg, ["ep_0000", "ep_0001"])
    assert ds.episodes_with_windows <= {"ep_0000", "ep_0001"}


def test_assert_disjoint_episodes():
    C.assert_disjoint_episodes(["a", "b"], ["c"], [])
    with pytest.raises(ValueError):
        C.assert_disjoint_episodes(["a", "b"], ["b"])
    with pytest.raises(ValueError):
        C.assert_disjoint_episodes(["a", "a"])


def test_episode_folds_are_disjoint_and_complete(root: Path):
    cfg = C.CarlaConfig(root=str(root))
    episodes = C.episode_ids(root)
    folds = C.episode_folds(cfg, episodes, 2)
    C.assert_disjoint_episodes(*folds)
    assert sorted(e for f in folds for e in f) == sorted(episodes)


def test_episode_folds_reject_a_single_fold(root: Path):
    with pytest.raises(ValueError):
        C.episode_folds(C.CarlaConfig(root=str(root)), C.episode_ids(root), 1)


def _batch(root: Path) -> dict[str, torch.Tensor]:
    cfg = C.CarlaConfig(root=str(root), obs_len=10, pred_len=20, stride=5)
    ds = C.CarlaDataset(cfg, C.episode_ids(root))
    return C.carla_collate([ds[i] for i in range(min(4, len(ds)))])


@pytest.mark.parametrize(
    "tag", ["vanilla", "AP", "EP", "IP", "EP+AP", "EP+IP", "EP+IP+AP"]
)
def test_every_ablation_configuration_runs(root: Path, tag: str):
    batch = _batch(root)
    net = CarlaTitanNet(PriorConfig.from_tag(tag), pred_len=20)
    out = net(batch)
    assert out.shape == (batch["obs_boxes"].shape[0], 4, 20, 4)
    assert torch.isfinite(out).all()


def test_action_prior_needs_action_labels(root: Path):
    batch = _batch(root)
    del batch["obs_actions"]
    net = CarlaTitanNet(PriorConfig.from_tag("AP"), pred_len=20)
    with pytest.raises(KeyError):
        net(batch)


def test_ego_prior_needs_ego_signal(root: Path):
    batch = _batch(root)
    del batch["obs_ego"]
    net = CarlaTitanNet(PriorConfig.from_tag("EP"), pred_len=20)
    with pytest.raises(KeyError):
        net(batch)


def test_vanilla_ignores_the_priors_entirely(root: Path):
    """The vanilla row must not read a prior even when the batch supplies one."""
    batch = _batch(root)
    net = CarlaTitanNet(PriorConfig.from_tag("vanilla"), pred_len=20).eval()
    with torch.no_grad():
        before = net(batch)
        batch["obs_actions"] = torch.full_like(batch["obs_actions"], C.ACTION_CROSSING)
        batch["obs_ego"] = batch["obs_ego"] + 5.0
        after = net(batch)
    assert torch.allclose(before, after)


@pytest.mark.skipif(not REAL_ROOT.is_dir(), reason="CARLA recordings are not present")
def test_real_recordings_load():
    cfg = C.CarlaConfig(root=str(REAL_ROOT))
    episodes = C.episode_ids(cfg.root)
    assert len(episodes) == 222
    ds = C.CarlaDataset(cfg, episodes[:5])
    assert len(ds) > 0
    for w in ds.windows:
        assert np.isfinite(w.obs_boxes).all() and np.isfinite(w.fut_boxes).all()
        assert (w.obs_boxes[w.agent_mask] > 0).all()
