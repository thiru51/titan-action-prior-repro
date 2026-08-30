"""Interaction prior.

TITAN's IP term models pair-wise interactions between agents in the scene. This
implements it as masked multi-head self-attention over the per-agent trajectory
encodings, with the relative spatial offset between agent boxes injected as an
additive bias so "who is near whom" is available to the attention rather than
having to be inferred from absolute image coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InteractionEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must divide by num_heads {num_heads}")
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_bias = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_heads),
        )
        self.num_heads = num_heads

    def forward(
        self,
        agent_feats: torch.Tensor,
        last_boxes: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """agent_feats (B, A, D); last_boxes (B, A, 4) in normalised coords."""
        b, a, _ = agent_feats.shape

        centres = last_boxes[..., :2]
        delta = centres.unsqueeze(2) - centres.unsqueeze(1)
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        # (B, A, A, H) -> (B, H, A, A) is the layout nn.MultiheadAttention wants.
        bias = self.spatial_bias(torch.cat([delta, dist], dim=-1)).permute(0, 3, 1, 2)

        if agent_mask is not None:
            pad = ~agent_mask.bool()
            # An all-padded row makes softmax produce NaN, so keep one slot open
            # and drop the result afterwards instead.
            all_pad = pad.all(dim=1)
            if all_pad.any():
                pad = pad.clone()
                pad[all_pad, 0] = False
            # Padding is folded into the additive bias rather than passed as a
            # separate key_padding_mask: mixing a bool mask with a float
            # attn_mask is deprecated, and one float mask says the same thing.
            bias = bias.masked_fill(pad.view(b, 1, 1, a), torch.finfo(bias.dtype).min)

        attended, _ = self.attn(
            agent_feats, agent_feats, agent_feats,
            attn_mask=bias.reshape(b * self.num_heads, a, a),
            need_weights=False,
        )
        x = self.norm(agent_feats + attended)
        x = self.norm2(x + self.ffn(x))

        if agent_mask is not None:
            x = x * agent_mask.unsqueeze(-1).to(x.dtype)
        return x
