"""Action-prior branch.

The paper finetunes single-stream I3D and a 3D ResNet, both pretrained on
Kinetics-600. Kinetics-600 I3D weights are not distributed with torchvision,
so this uses torchvision's r3d_18 (Kinetics-400). That is a substitution of the
3D-ResNet arm the paper also reports, not of I3D -- see README for the honest
statement of what this changes.

The branch consumes a per-agent cropped tube (the pedestrian's box tracked over
the observation window) and emits both the multi-head action logits and a
feature vector that the trajectory decoder consumes as the action prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.schema import ACTION_GROUPS

_BACKBONES = {
    "r3d_18": ("r3d_18", "R3D_18_Weights"),
    "mc3_18": ("mc3_18", "MC3_18_Weights"),
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights"),
}


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(_BACKBONES)}")
    import torchvision.models.video as tvv

    ctor_name, weights_name = _BACKBONES[name]
    ctor = getattr(tvv, ctor_name)
    weights = None
    if pretrained:
        try:
            weights = getattr(tvv, weights_name).DEFAULT
        except AttributeError:
            weights = None
    net = ctor(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


class ActionBranch(nn.Module):
    def __init__(
        self,
        out_dim: int = 128,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone, pretrained)
        self.backbone_name = backbone
        # 3D convs hit the tensor-core kernels only with the channel dimension
        # innermost. Weights are laid out once here; activations are converted
        # per batch in forward.
        self.backbone = self.backbone.to(memory_format=torch.channels_last_3d)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.project = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        # One head per hierarchical action group; TITAN's labels are parallel
        # multi-class attributes, not a single flat softmax.
        self.heads = nn.ModuleDict(
            {g.key: nn.Linear(out_dim, g.num_classes) for g in ACTION_GROUPS}
        )
        self.out_dim = out_dim

    def forward(self, tubes: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """tubes: (B, A, C, T, H, W) -> features (B, A, out_dim) and per-group logits."""
        if tubes.dim() != 6:
            raise ValueError(f"expected (B, A, C, T, H, W), got {tuple(tubes.shape)}")
        b, a = tubes.shape[:2]
        flat = tubes.flatten(0, 1).contiguous(memory_format=torch.channels_last_3d)
        feats = self.project(self.backbone(flat))
        logits = {k: head(feats).view(b, a, -1) for k, head in self.heads.items()}
        return feats.view(b, a, -1), logits
