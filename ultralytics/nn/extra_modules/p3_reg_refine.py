# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import torch
import torch.nn as nn

__all__ = ('P3RegRefine',)

def _norm_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class P3RegRefine(nn.Module):
    """Shape-preserving local residual for the P3 regression input."""

    def __init__(self, channels, reduction=4, residual_scale=0.1):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not isinstance(reduction, int) or reduction < 1:
            raise ValueError("reduction must be a positive integer")
        if not 0 < float(residual_scale) < float("inf"):
            raise ValueError("residual_scale must be positive and finite")
        hidden = max(8, channels // reduction)
        groups = _norm_groups(hidden)
        self.residual_scale = float(residual_scale)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1,
                      groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, x):
        delta = self.project(self.local(self.reduce(x)))
        # Do not mutate the feature also used by classification.
        return x + self.residual_scale * delta.to(dtype=x.dtype)
