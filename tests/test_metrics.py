import torch

from titan.baselines import constant_velocity
from titan.metrics import ForecastMetrics, box_iou, displacement_errors, final_iou


def test_fde_is_error_at_last_step():
    pred = torch.zeros(1, 1, 3, 4)
    target = torch.zeros(1, 1, 3, 4)
    target[0, 0, :, 0] = torch.tensor([1.0, 2.0, 10.0])

    ade, fde = displacement_errors(pred, target)
    assert fde.item() == 10.0
    assert abs(ade.item() - (1 + 2 + 10) / 3) < 1e-6


def test_fde_uses_last_valid_step_not_last_index():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :, 0] = torch.tensor([1.0, 5.0, 99.0, 99.0])
    mask = torch.tensor([[[True, True, False, False]]])

    _, fde = displacement_errors(pred, target, mask)
    assert fde.item() == 5.0


def test_ade_ignores_masked_steps():
    pred = torch.zeros(1, 1, 3, 4)
    target = torch.zeros(1, 1, 3, 4)
    target[0, 0, :, 0] = torch.tensor([2.0, 4.0, 1000.0])
    mask = torch.tensor([[[True, True, False]]])

    ade, _ = displacement_errors(pred, target, mask)
    assert abs(ade.item() - 3.0) < 1e-6


def test_iou_identical_boxes_is_one():
    box = torch.tensor([[100.0, 100.0, 50.0, 80.0]])
    assert abs(box_iou(box, box).item() - 1.0) < 1e-5


def test_iou_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[500.0, 500.0, 10.0, 10.0]])
    assert box_iou(a, b).item() == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes offset by 5 in u: intersection 50, union 150.
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 10.0, 10.0]])
    assert abs(box_iou(a, b).item() - 50.0 / 150.0) < 1e-5


def test_final_iou_matches_manual():
    pred = torch.zeros(1, 1, 2, 4)
    target = torch.zeros(1, 1, 2, 4)
    pred[0, 0, -1] = torch.tensor([0.0, 0.0, 10.0, 10.0])
    target[0, 0, -1] = torch.tensor([5.0, 0.0, 10.0, 10.0])
    assert abs(final_iou(pred, target).item() - 50.0 / 150.0) < 1e-5


def test_accumulator_matches_single_shot():
    torch.manual_seed(0)
    pred = torch.randn(4, 3, 20, 4) * 10
    target = torch.randn(4, 3, 20, 4) * 10
    mask = torch.ones(4, 3, 20, dtype=torch.bool)

    ade, fde = displacement_errors(pred, target, mask)
    acc = ForecastMetrics()
    acc.update(pred, target, mask)
    res = acc.compute()

    assert abs(res["ADE"] - ade.item()) < 1e-4
    assert abs(res["FDE"] - fde.item()) < 1e-4
    assert res["num_tracks"] == 12


def test_accumulator_fiou_matches_single_shot():
    torch.manual_seed(1)
    pred = torch.randn(2, 3, 5, 4).abs() * 20 + 50
    target = torch.randn(2, 3, 5, 4).abs() * 20 + 50
    mask = torch.ones(2, 3, 5, dtype=torch.bool)

    acc = ForecastMetrics()
    acc.update(pred, target, mask)
    assert abs(acc.compute()["FIOU"] - final_iou(pred, target, mask).item()) < 1e-4


def test_accumulator_fiou_is_one_for_perfect_boxes():
    boxes = torch.rand(2, 3, 5, 4) * 100 + 20
    acc = ForecastMetrics()
    acc.update(boxes, boxes.clone(), torch.ones(2, 3, 5, dtype=torch.bool))
    assert abs(acc.compute()["FIOU"] - 1.0) < 1e-4


def test_metrics_stay_fp32_when_predictions_arrive_in_half_precision():
    # The training loop hands eval bf16/fp16 tensors. FDE is a reported number,
    # so the accumulator has to widen them rather than measure in half.
    pred = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    target = torch.zeros(1, 1, 2, 4)
    target[0, 0, -1, 0] = 1234.0

    acc = ForecastMetrics()
    acc.update(pred, target, torch.ones(1, 1, 2, dtype=torch.bool))
    assert acc.compute()["FDE"] == 1234.0


def test_constant_velocity_is_exact_on_linear_motion():
    # A perfectly linear track must give zero error, which is the sanity check
    # that the baseline and the metric agree on time indexing.
    obs = torch.zeros(1, 1, 10, 4)
    obs[0, 0, :, 0] = torch.arange(10, dtype=torch.float32) * 3.0
    obs[0, 0, :, 2:] = 20.0

    pred = constant_velocity(obs, pred_len=20)
    fut = torch.zeros(1, 1, 20, 4)
    fut[0, 0, :, 0] = (torch.arange(20, dtype=torch.float32) + 10) * 3.0
    fut[0, 0, :, 2:] = 20.0

    _, fde = displacement_errors(pred, fut)
    assert fde.item() < 1e-4
