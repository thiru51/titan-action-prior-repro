import dataclasses

import pytest
import torch

from titan.config import Config, PriorConfig
from titan.data.synthetic import SyntheticTitanDataset
from titan.data.common import collate
from titan.models.aim import AgentImportance
from titan.models.titan_net import TitanNet

TAGS = ["vanilla", "AP", "EP", "IP", "EP+AP", "EP+IP", "EP+IP+AP"]


def _tiny_cfg(tag: str) -> Config:
    cfg = Config()
    cfg.priors = PriorConfig.from_tag(tag)
    cfg.data = dataclasses.replace(
        cfg.data, max_agents=4, obs_len=10, pred_len=20, clip_frames=4, crop_size=32
    )
    cfg.model = dataclasses.replace(
        cfg.model,
        traj_hidden=16,
        ego_hidden=8,
        action_feat_dim=16,
        interaction_dim=16,
        decoder_hidden=16,
        pretrained_backbone=False,
    )
    cfg.synthetic = dataclasses.replace(cfg.synthetic, num_clips=2, agents_per_clip=3)
    return cfg


def _batch(cfg: Config):
    ds = SyntheticTitanDataset(cfg.data, "train", cfg.synthetic, load_video=cfg.priors.action)
    return collate([ds[0], ds[1]])


def test_prior_tag_roundtrip():
    for tag in TAGS:
        assert PriorConfig.from_tag(tag).tag == tag


def test_unknown_prior_tag_rejected():
    with pytest.raises(ValueError):
        PriorConfig.from_tag("EP+XP")


@pytest.mark.parametrize("tag", TAGS)
def test_every_ablation_forward_and_backward(tag):
    cfg = _tiny_cfg(tag)
    model = TitanNet(cfg)
    out = model(_batch(cfg))

    assert out["pred_boxes"].shape == (2, cfg.data.max_agents, cfg.data.pred_len, 4)
    assert out["ego_pred"].shape == (2, cfg.data.pred_len, 2)
    assert torch.isfinite(out["pred_boxes"]).all()

    out["pred_boxes"].sum().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_action_prior_missing_tubes_raises():
    cfg = _tiny_cfg("AP")
    model = TitanNet(cfg)
    batch = _batch(cfg)
    del batch["tubes"]
    with pytest.raises(KeyError):
        model(batch)


def test_importance_weights_sum_to_one_over_real_agents():
    aim = AgentImportance(8)
    feats = torch.randn(3, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    pooled, w = aim(feats, mask)

    assert torch.allclose(w.sum(-1), torch.ones(3), atol=1e-5)
    # Padded slots must get exactly zero, otherwise the pooled scene vector is
    # contaminated by whatever garbage sits in the padding.
    assert w[0, 3:].abs().max() == 0
    assert w[1, 1:].abs().max() == 0
    assert pooled.shape == (3, 8)


def test_importance_handles_empty_frame():
    aim = AgentImportance(8)
    feats = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    pooled, w = aim(feats, mask)

    assert torch.isfinite(pooled).all()
    assert w.detach().sum().item() == 0.0


def test_interaction_changes_prediction():
    # If IP is wired up, moving a neighbour must change the target agent's
    # forecast; otherwise the attention is silently a no-op.
    cfg = _tiny_cfg("IP")
    torch.manual_seed(0)
    model = TitanNet(cfg).eval()
    batch = _batch(cfg)

    with torch.no_grad():
        a = model(batch)["pred_boxes"][:, 0].clone()
        moved = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        moved["obs_boxes"][:, 1] += 0.2
        b = model(moved)["pred_boxes"][:, 0]

    assert not torch.allclose(a, b, atol=1e-6)


def test_synthetic_dataset_is_flagged():
    cfg = _tiny_cfg("EP+IP+AP")
    ds = SyntheticTitanDataset(cfg.data, "train", cfg.synthetic, load_video=True)
    assert ds.is_synthetic is True
