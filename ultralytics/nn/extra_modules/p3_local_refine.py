# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2

__all__ = ('P3LocalRefine', 'C3k2_P3LocalRefine', 'C3k2_P3LocalRefineNoGate')

def _norm_groups(channels):
    # Use >=2 channels/group when possible, including narrow feature maps.
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


def _local_difference(x):
    # Do not introduce zero-padding edges on a spatially constant feature.
    average = F.avg_pool2d(
        x, kernel_size=3, stride=1, padding=1, count_include_pad=False
    )
    return x - average


class P3LocalRefine(nn.Module):
    """Local-contrast residual; output shape matches input shape."""

    def __init__(self, channels, reduction=4, enable_gate=True,
                 residual_scale=0.1):
        super().__init__()
        if channels < 1 or reduction < 1 or residual_scale <= 0:
            raise ValueError("channels/reduction/residual_scale must be positive")
        hidden = max(8, channels // reduction)
        groups = _norm_groups(hidden)
        self.enable_gate = bool(enable_gate)
        self.residual_scale = float(residual_scale)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1,
                      groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        if self.enable_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(8, 1, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)
        # Only projection is zero. Do not also introduce a zero residual alpha.
        # Projection gets a gradient on step 1; upstream layers learn later.
        nn.init.zeros_(self.project.weight)

    def _gate(self, context, contrast):
        if self.spatial_gate is None:
            return 0.5
        cues = torch.cat((
            context.abs().mean(dim=1, keepdim=True),
            contrast.abs().mean(dim=1, keepdim=True),
        ), dim=1)
        return self.spatial_gate(cues)

    def forward(self, x):
        context = self.reduce(x)
        contrast = _local_difference(context)
        gate = self._gate(context, contrast)
        delta = self.project(self.detail(contrast) * gate)
        # Out-of-place: do not modify features retained by other branches.
        return x + self.residual_scale * delta.to(dtype=x.dtype)


class C3k2_P3LocalRefine(C3k2):
    """C3k2 followed by P3 refinement; old constructor/state keys preserved."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5,
                 g=1, shortcut=True):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e,
                         g=g, shortcut=shortcut)
        with torch.random.fork_rng(devices=[]):
            self.p3_refine = P3LocalRefine(c2, enable_gate=True)

    def forward(self, x):
        return self.p3_refine(super().forward(x))

    def forward_split(self, x):
        return self.p3_refine(super().forward_split(x))


class C3k2_P3LocalRefineNoGate(C3k2):
    """Optional control: same local path, constant 0.5 spatial weighting."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5,
                 g=1, shortcut=True):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e,
                         g=g, shortcut=shortcut)
        with torch.random.fork_rng(devices=[]):
            self.p3_refine = P3LocalRefine(c2, enable_gate=False)

    def forward(self, x):
        return self.p3_refine(super().forward(x))

    def forward_split(self, x):
        return self.p3_refine(super().forward_split(x))
