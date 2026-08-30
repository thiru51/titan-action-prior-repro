"""Agent Importance Mechanism.

The paper describes AIM as a self-attention-like weighting over per-agent
encodings, H~_t^{e,i} = w_t^i * H_t^{e,i} with w_t^i = phi_t(H_t^{e,i}), learned
jointly with future ego-motion prediction so the weights end up expressing how
much each agent matters to where the ego vehicle is going.

The paper does not publish the exact shape of phi, so this is a faithful but
independent implementation: a small MLP scoring head, a mask-aware softmax over
the agents present in the frame, and a weighted pool. The weights are returned
rather than hidden because their interpretability -- attributing ego risk to
individual agents -- is the point of the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AgentImportance(nn.Module):
    def __init__(self, agent_dim: int, hidden: int = 64, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.score = nn.Sequential(
            nn.Linear(agent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        agent_feats: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """agent_feats: (B, A, D); agent_mask: (B, A) bool, True = real agent.

        Returns the pooled scene encoding (B, D) and the importance weights (B, A).
        """
        if agent_feats.dim() != 3:
            raise ValueError(f"expected (B, A, D), got {tuple(agent_feats.shape)}")

        logits = self.score(agent_feats).squeeze(-1) / self.temperature

        if agent_mask is not None:
            # Padded slots must not steal probability mass from real agents.
            logits = logits.masked_fill(~agent_mask.bool(), torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=-1)

        if agent_mask is not None:
            # A frame with zero agents softmaxes to uniform garbage; zero it so
            # the pooled context is simply empty instead of noise.
            empty = ~agent_mask.bool().any(dim=-1, keepdim=True)
            weights = weights.masked_fill(empty, 0.0)

        pooled = torch.einsum("ba,bad->bd", weights, agent_feats)
        return pooled, weights
