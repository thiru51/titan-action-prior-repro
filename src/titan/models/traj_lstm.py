"""LSTM encoder-decoder for ETH/UCY, with optional social pooling.

This is the Social-LSTM family stripped to its two published control settings:

  social=False  a vanilla LSTM over one person's own past, no neighbours.
                This is the "LSTM" row in Social-GAN's Table 1.
  social=True   the same network plus a pooling step that lets each person see
                every other person in the same window before decoding. This is
                the idea behind the "S-LSTM" row.

It does not touch `titan_net.py`. It is a separate model on a separate dataset,
and mixing the two would make it impossible to say which code produced which
number. The structure is deliberately the same though -- encode the past,
decode a fixed horizon of *displacements*, integrate onto the last observed
position -- so a result here says something about that decoding recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SocialPool(nn.Module):
    """Max-pool over neighbours, conditioned on where they are relative to you.

    Max rather than mean: a person's path is dominated by the one or two
    neighbours about to get in the way, and averaging washes those out into a
    crowd that is mostly irrelevant.
    """

    def __init__(self, hidden: int, embed: int = 64) -> None:
        super().__init__()
        self.rel_embed = nn.Linear(2, embed)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + embed, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, seq_start_end: torch.Tensor) -> torch.Tensor:
        """h: (N, hidden) per-person state. pos: (N, 2) last observed position."""
        out = torch.zeros_like(h)
        for start, end in seq_start_end.tolist():
            n = end - start
            if n == 0:
                continue
            hs = h[start:end]
            ps = pos[start:end]
            rel = ps.unsqueeze(0) - ps.unsqueeze(1)  # (n, n, 2): j seen from i
            feat = torch.cat(
                [hs.unsqueeze(0).expand(n, n, -1), torch.relu(self.rel_embed(rel))], dim=-1
            )
            out[start:end] = self.mlp(feat).max(dim=1).values
        return out


class TrajLSTM(nn.Module):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        embed: int = 64,
        hidden: int = 64,
        dropout: float = 0.0,
        social: bool = False,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.social = social

        # The network only ever sees step-to-step displacements, never absolute
        # world coordinates. That makes it invariant to where in the scene a
        # person happens to be, which is the only reason a model trained on
        # four scenes can be evaluated on a fifth one it has never seen.
        self.in_embed = nn.Linear(2, embed)
        self.encoder = nn.LSTM(embed, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.pool = SocialPool(hidden) if social else None
        self.bridge = nn.Linear(hidden * 2 if social else hidden, hidden)

        self.decoder = nn.LSTMCell(embed, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, obs: torch.Tensor, seq_start_end: torch.Tensor | None = None) -> torch.Tensor:
        """obs: (N, obs_len, 2) absolute positions -> (N, pred_len, 2) absolute."""
        disp = obs[:, 1:] - obs[:, :-1]
        _, (h, c) = self.encoder(self.drop(torch.relu(self.in_embed(disp))))
        h, c = h[-1], c[-1]

        if self.pool is not None:
            if seq_start_end is None:
                raise ValueError("social pooling needs seq_start_end to know who shares a window")
            h = torch.cat([h, self.pool(h, obs[:, -1], seq_start_end)], dim=-1)
        h = torch.relu(self.bridge(h))

        cur_pos = obs[:, -1]
        step = disp[:, -1]
        outputs = []
        for _ in range(self.pred_len):
            h, c = self.decoder(torch.relu(self.in_embed(step)), (h, c))
            step = self.out(h)
            cur_pos = cur_pos + step
            outputs.append(cur_pos)
        return torch.stack(outputs, dim=1)
