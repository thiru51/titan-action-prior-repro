"""TITAN forecasting network with toggleable ego / interaction / action priors.

Layout follows the paper: per-agent past-trajectory encoding, an ego-motion
encoding from vehicle odometry (EP), pair-wise agent interaction (IP), an
action prior from the 3D-conv action branch (AP), all fused into a GRU decoder
that rolls out future boxes. The Agent Importance Mechanism pools agents into
the ego-motion head and exposes per-agent importance weights.

Setting priors.ego / interaction / action reproduces the paper's ablation rows
(vanilla, AP, EP, IP, EP+AP, EP+IP, EP+IP+AP).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..data.schema import ACTION_GROUPS
from .action_branch import ActionBranch
from .aim import AgentImportance
from .interaction import InteractionEncoder


class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        # Velocity is fed alongside position: a GRU over raw pixel coordinates
        # has to spend capacity learning to difference them.
        self.input_proj = nn.Linear(8, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """boxes: (B, A, T, 4) normalised -> (B, A, hidden)."""
        b, a, t, _ = boxes.shape
        vel = torch.zeros_like(boxes)
        vel[:, :, 1:] = boxes[:, :, 1:] - boxes[:, :, :-1]
        x = torch.cat([boxes, vel], dim=-1).flatten(0, 1)
        _, h = self.gru(self.drop(torch.relu(self.input_proj(x))))
        return h[-1].view(b, a, -1)


class EgoEncoder(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, ego: torch.Tensor) -> torch.Tensor:
        """ego: (B, T, 2) = (longitudinal accel, yaw rate) -> (B, hidden)."""
        _, h = self.gru(self.drop(torch.relu(self.input_proj(ego))))
        return h[-1]


class TitanNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        m, p = cfg.model, cfg.priors
        self.priors = p
        self.pred_len = cfg.data.pred_len

        self.traj_encoder = TrajectoryEncoder(m.traj_hidden, m.dropout)

        fusion_dim = m.traj_hidden

        self.ego_encoder = EgoEncoder(m.ego_hidden, m.dropout) if p.ego else None
        if p.ego:
            fusion_dim += m.ego_hidden

        self.action_branch = (
            ActionBranch(
                out_dim=m.action_feat_dim,
                backbone=m.action_backbone,
                pretrained=m.pretrained_backbone,
                freeze_backbone=m.freeze_backbone,
                dropout=m.dropout,
            )
            if p.action
            else None
        )
        if p.action:
            fusion_dim += m.action_feat_dim

        # Interaction runs on the agent encoding *after* the action prior is
        # concatenated, so neighbours' actions are visible to the attention.
        self.interaction = None
        if p.interaction:
            interact_in = m.traj_hidden + (m.action_feat_dim if p.action else 0)
            self.interaction_proj = nn.Linear(interact_in, m.interaction_dim)
            self.interaction = InteractionEncoder(m.interaction_dim, dropout=m.dropout)
            fusion_dim += m.interaction_dim

        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, m.decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(m.dropout),
        )

        self.decoder = nn.GRUCell(m.decoder_hidden + 4, m.decoder_hidden)
        self.out = nn.Linear(m.decoder_hidden, 4)

        self.aim = AgentImportance(m.decoder_hidden)
        self.ego_head = nn.Sequential(
            nn.Linear(m.decoder_hidden, m.decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(m.decoder_hidden, self.pred_len * 2),
        )

        # Per-group log-variance for the paper's uncertainty-weighted action loss.
        if p.action:
            self.action_log_var = nn.Parameter(torch.zeros(len(ACTION_GROUPS)))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes = batch["obs_boxes"]
        agent_mask = batch.get("agent_mask")
        b, a = boxes.shape[:2]

        feats = self.traj_encoder(boxes)
        parts = [feats]
        action_logits: dict[str, torch.Tensor] = {}

        action_feats = None
        if self.action_branch is not None:
            tubes = batch.get("tubes")
            if tubes is None:
                raise KeyError("action prior is enabled but the batch has no 'tubes'")
            action_feats, action_logits = self.action_branch(tubes)
            parts.append(action_feats)

        if self.interaction is not None:
            interact_in = feats if action_feats is None else torch.cat([feats, action_feats], dim=-1)
            inter = self.interaction(
                self.interaction_proj(interact_in), boxes[:, :, -1, :], agent_mask
            )
            parts.append(inter)

        if self.ego_encoder is not None:
            ego = batch.get("obs_ego")
            if ego is None:
                raise KeyError("ego prior is enabled but the batch has no 'obs_ego'")
            ego_feat = self.ego_encoder(ego)
            parts.append(ego_feat.unsqueeze(1).expand(b, a, -1))

        fused = self.fuse(torch.cat(parts, dim=-1))

        scene, importance = self.aim(fused, agent_mask)
        ego_pred = self.ego_head(scene).view(b, self.pred_len, 2)

        # Decode residual box deltas and integrate. Predicting absolute pixel
        # coordinates from scratch makes the decoder relearn the identity of the
        # last observed box every step; deltas keep it on the actual dynamics.
        h = fused.flatten(0, 1)
        ctx = h
        cur = boxes[:, :, -1, :].flatten(0, 1)
        outputs = []
        for _ in range(self.pred_len):
            h = self.decoder(torch.cat([ctx, cur], dim=-1), h)
            cur = cur + self.out(h)
            outputs.append(cur)

        pred = torch.stack(outputs, dim=1).view(b, a, self.pred_len, 4)

        return {
            "pred_boxes": pred,
            "ego_pred": ego_pred,
            "importance": importance,
            "action_logits": action_logits,
        }
