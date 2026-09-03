"""The seven-prior ablation network, wired for CARLA recordings.

This is `titan_net.TitanNet` with one part swapped, and it does not touch that
file. It imports `TrajectoryEncoder`, `EgoEncoder`, `InteractionEncoder` and
`PriorConfig` from the TITAN path, so the ego prior, the interaction prior, the
fusion and the delta-integrating GRU decoder are literally the same code.

The one difference is the **action prior**. `TitanNet` gets it from
`ActionBranch`, a 3D convolutional network over a cropped video tube of the
pedestrian. The CARLA recordings kept only their annotations -- the RGB frames
were deleted after pose caching in the project that recorded them -- so there
is no tube to convolve. Here the action prior is instead an embedding of the
per-frame contextual action label read straight from the simulator, encoded by
a small GRU so the *timing* of a transition (stepping onto the road, stopping
at the kerb) is visible and not just the final state.

Two consequences, both of which belong in any write-up of a result from this
file:

1. The action prior is exact. It is a map query, not an annotator's judgement.
   That removes label noise as an explanation for whatever it does or does not
   contribute.
2. The action prior is restricted. It covers TITAN's contextual group only.
   CARLA has no gaze, no gestures and no carried objects, so TITAN's
   communicative and transportive groups are absent, and this cannot speak to
   the full action prior the paper describes.

The Agent Importance Mechanism is deliberately left out. AIM is trained through
an auxiliary future-ego-motion prediction, and adding that supervision to every
row would hand the ego-prior-off configurations ego information through the
back door. The ablation has to differ only in the prior set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PriorConfig
from ..data.carla import NUM_CONTEXTUAL_ACTIONS
from .interaction import InteractionEncoder
from .titan_net import EgoEncoder, TrajectoryEncoder


class ContextualActionEncoder(nn.Module):
    """Per-frame contextual action labels -> one feature vector per agent."""

    def __init__(self, out_dim: int, embed: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Embedding(NUM_CONTEXTUAL_ACTIONS, embed)
        self.gru = nn.GRU(embed, out_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (B, A, T) int64 -> (B, A, out_dim)."""
        b, a, _ = actions.shape
        _, h = self.gru(self.drop(self.embed(actions.flatten(0, 1))))
        return h[-1].view(b, a, -1)


class CarlaTitanNet(nn.Module):
    """Trajectory forecaster with switchable ego / interaction / action priors.

    `priors` selects one of the paper's seven ablation rows: vanilla, AP, EP,
    IP, EP+AP, EP+IP, EP+IP+AP. Everything else is held fixed across rows.
    """

    def __init__(
        self,
        priors: PriorConfig,
        pred_len: int = 20,
        traj_hidden: int = 128,
        ego_hidden: int = 64,
        action_dim: int = 128,
        interaction_dim: int = 128,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.priors = priors
        self.pred_len = pred_len

        self.traj_encoder = TrajectoryEncoder(traj_hidden, dropout)
        fusion_dim = traj_hidden

        self.ego_encoder = EgoEncoder(ego_hidden, dropout) if priors.ego else None
        if priors.ego:
            fusion_dim += ego_hidden

        self.action_encoder = (
            ContextualActionEncoder(action_dim, dropout=dropout) if priors.action else None
        )
        if priors.action:
            fusion_dim += action_dim

        # Interaction runs after the action prior is concatenated, exactly as in
        # TitanNet, so a neighbour's action is visible to the attention.
        self.interaction = None
        if priors.interaction:
            interact_in = traj_hidden + (action_dim if priors.action else 0)
            self.interaction_proj = nn.Linear(interact_in, interaction_dim)
            self.interaction = InteractionEncoder(interaction_dim, dropout=dropout)
            fusion_dim += interaction_dim

        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.decoder = nn.GRUCell(decoder_hidden + 4, decoder_hidden)
        self.out = nn.Linear(decoder_hidden, 4)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        boxes = batch["obs_boxes"]
        agent_mask = batch.get("agent_mask")
        b, a = boxes.shape[:2]

        feats = self.traj_encoder(boxes)
        parts = [feats]

        action_feats = None
        if self.action_encoder is not None:
            actions = batch.get("obs_actions")
            if actions is None:
                raise KeyError("action prior is enabled but the batch has no 'obs_actions'")
            action_feats = self.action_encoder(actions)
            parts.append(action_feats)

        if self.interaction is not None:
            interact_in = feats if action_feats is None else torch.cat([feats, action_feats], -1)
            parts.append(
                self.interaction(
                    self.interaction_proj(interact_in), boxes[:, :, -1, :], agent_mask
                )
            )

        if self.ego_encoder is not None:
            ego = batch.get("obs_ego")
            if ego is None:
                raise KeyError("ego prior is enabled but the batch has no 'obs_ego'")
            parts.append(self.ego_encoder(ego).unsqueeze(1).expand(b, a, -1))

        fused = self.fuse(torch.cat(parts, dim=-1))

        # Decode residual box deltas and integrate onto the last observed box,
        # the same recipe TitanNet and the ETH/UCY LSTM both use.
        h = fused.flatten(0, 1)
        ctx = h
        cur = boxes[:, :, -1, :].flatten(0, 1)
        outputs = []
        for _ in range(self.pred_len):
            h = self.decoder(torch.cat([ctx, cur], dim=-1), h)
            cur = cur + self.out(h)
            outputs.append(cur)
        return torch.stack(outputs, dim=1).view(b, a, self.pred_len, 4)
