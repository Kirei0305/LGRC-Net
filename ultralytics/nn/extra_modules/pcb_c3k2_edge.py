# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv

__all__ = ('C3k2_MSEdge',)

class EdgeEnhancer(nn.Module):
    """Original x + Sigmoid(Conv-BN(x - AvgPool3x3(x))) implementation."""

    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = x - self.pool(x)
        return x + self.out_conv(edge)


class MutilScaleEdgeInformationEnhance(nn.Module):
    """The supplied single-input, same-channel, same-size enhancement module."""

    def __init__(self, inc, bins=(3, 6, 9, 12)):
        super().__init__()
        bins = tuple(bins)
        if not bins or any(not isinstance(b, int) or b < 1 for b in bins):
            raise ValueError("bins must contain positive integer pooling grids")
        if inc < len(bins) or inc % len(bins):
            raise ValueError("inc must be positive and divisible by len(bins)")
        self.bins = bins
        branch_channels = inc // len(bins)
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                Conv(inc, branch_channels, 1),
                Conv(branch_channels, branch_channels, 3, g=branch_channels),
            ) for b in bins
        ])
        self.ees = nn.ModuleList(EdgeEnhancer(branch_channels) for _ in bins)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, feature in enumerate(self.features):
            resized = F.interpolate(feature(x), x_size[2:], mode="bilinear", align_corners=True)
            out.append(self.ees[idx](resized))
        return self.final_conv(torch.cat(out, dim=1))


class _MSEdgeResidualBlock(nn.Module):
    """Same-channel replacement branch with the old Bottleneck shortcut flag."""

    def __init__(self, dim, shortcut=True, bins=(3, 6, 9, 12)):
        super().__init__()
        self.enhance = MutilScaleEdgeInformationEnhance(dim, bins)
        self.add = bool(shortcut)

    def forward(self, x):
        delta = self.enhance(x)
        return x + delta if self.add else delta


class C3k2_MSEdge(C3k2):
    """Layer4 adapter; matches C3k2 argument order c1,c2,n,c3k,e,g,shortcut.

This focused adapter supports the current layer4 c3k=False,g=1 setting.
Other model layers continue using their original module classes.
"""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 bins=(3, 6, 9, 12)):
        if c3k or g != 1:
            raise ValueError("This layer4 adapter requires c3k=False and g=1")
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            _MSEdgeResidualBlock(self.c, shortcut, bins) for _ in range(n)
        )
