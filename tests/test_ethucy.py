from __future__ import annotations

import numpy as np
import torch

from titan.baselines import constant_velocity, linear_least_squares
from titan.data.ethucy import (
    EthUcyConfig,
    EthUcyDataset,
    _windows_from_file,
    ethucy_collate,
    read_trajectory_file,
)
from titan.metrics import ForecastMetrics
from titan.models.traj_lstm import TrajLSTM

OBS, PRED = 8, 12
SEQ = OBS + PRED


def _track(ped_id: int, n_frames: int, start_frame: int = 0) -> list[str]:
    return [
        f"{start_frame + 10 * t}\t{float(ped_id)}\t{float(t)}\t{float(2 * t)}"
        for t in range(n_frames)
    ]


def test_read_trajectory_file_parses_tabs(tmp_path):
    f = tmp_path / "scene.txt"
    f.write_text("\n".join(_track(1, 3)) + "\n")
    data = read_trajectory_file(f)
    assert data.shape == (3, 4)
    assert data[2].tolist() == [20.0, 1.0, 2.0, 4.0]


def test_window_keeps_only_fully_tracked_people():
    # Person 1 spans the whole window; person 2 stops one step early and must
    # be dropped rather than padded, which is what the published protocol does.
    rows = []
    for t in range(SEQ):
        rows.append([10 * t, 1.0, float(t), 0.0])
        if t < SEQ - 1:
            rows.append([10 * t, 2.0, float(t), 5.0])
    windows = _windows_from_file(np.asarray(rows), OBS, PRED, stride=1, min_agents=1)
    assert len(windows) == 1
    obs, fut = windows[0]
    assert obs.shape == (1, OBS, 2)
    assert fut.shape == (1, PRED, 2)


def test_window_drops_person_with_a_hole_in_the_middle():
    rows = []
    for t in range(SEQ):
        rows.append([10 * t, 1.0, float(t), 0.0])
        if t != 5:
            rows.append([10 * t, 2.0, float(t), 5.0])
    windows = _windows_from_file(np.asarray(rows), OBS, PRED, stride=1, min_agents=1)
    assert windows[0][0].shape[0] == 1


def test_min_agents_filters_sparse_windows():
    rows = [[10 * t, 1.0, float(t), 0.0] for t in range(SEQ)]
    data = np.asarray(rows)
    assert len(_windows_from_file(data, OBS, PRED, 1, min_agents=1)) == 1
    assert len(_windows_from_file(data, OBS, PRED, 1, min_agents=2)) == 0


def test_collate_marks_window_boundaries():
    items = [
        {"obs": torch.zeros(3, OBS, 2), "fut": torch.zeros(3, PRED, 2)},
        {"obs": torch.zeros(2, OBS, 2), "fut": torch.zeros(2, PRED, 2)},
    ]
    batch = ethucy_collate(items)
    assert batch["obs"].shape == (5, OBS, 2)
    assert batch["seq_start_end"].tolist() == [[0, 3], [3, 5]]


def test_linear_least_squares_is_exact_on_a_straight_line():
    t = torch.arange(OBS, dtype=torch.float32)
    obs = torch.stack([2.0 * t + 1.0, -0.5 * t], dim=-1).unsqueeze(0)
    pred = linear_least_squares(obs, PRED)
    future = torch.arange(OBS, OBS + PRED, dtype=torch.float32)
    expected = torch.stack([2.0 * future + 1.0, -0.5 * future], dim=-1).unsqueeze(0)
    assert torch.allclose(pred, expected, atol=1e-4)


def test_constant_velocity_accepts_two_dimensional_positions():
    t = torch.arange(OBS, dtype=torch.float32)
    obs = torch.stack([t, t], dim=-1).unsqueeze(0)
    pred = constant_velocity(obs, PRED)
    assert pred.shape == (1, PRED, 2)
    assert torch.allclose(pred[0, -1], torch.tensor([19.0, 19.0]))


def test_metrics_on_positions_give_metres():
    # A fixed 3 m offset at every step means ADE and FDE are both 3 m. This is
    # the check that the shared metric code means what the benchmark means.
    target = torch.zeros(4, PRED, 2)
    pred = target + torch.tensor([3.0, 0.0])
    m = ForecastMetrics()
    m.update(pred, target)
    res = m.compute()
    assert abs(res["ADE"] - 3.0) < 1e-5
    assert abs(res["FDE"] - 3.0) < 1e-5


def test_traj_lstm_shapes_and_anchoring():
    torch.manual_seed(0)
    obs = torch.randn(5, OBS, 2)
    sse = torch.tensor([[0, 3], [3, 5]])
    for social in (False, True):
        net = TrajLSTM(OBS, PRED, social=social)
        out = net(obs, sse)
        assert out.shape == (5, PRED, 2)
        assert torch.isfinite(out).all()


def test_real_dataset_loads_and_has_the_expected_horizon():
    cfg = EthUcyConfig()
    ds = EthUcyDataset(cfg, "zara1", "test")
    assert len(ds) > 0
    item = ds[0]
    assert item["obs"].shape[1:] == (OBS, 2)
    assert item["fut"].shape[1:] == (PRED, 2)
