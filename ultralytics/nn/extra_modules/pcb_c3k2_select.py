# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv
from .pcb_c3k2_edge import C3k2_MSEdge, MutilScaleEdgeInformationEnhance

__all__ = ('C3k2_MSSelect',)

class ChannelPool(nn.Module):
    """Explicit supplementary helper: [channel max, channel mean]."""

    def forward(self, x):
        maximum = torch.max(x, dim=1, keepdim=True)[0]
        average = torch.mean(x, dim=1, keepdim=True)
        return torch.cat((maximum, average), dim=1)


class DSM_SpatialGate(nn.Module):
    """Supplied spatial branch; Conv(act=False) stays linear after BN."""

    def __init__(self, channel):
        super().__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU()),
        )
        self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        return self.dw1(x) * out + self.dw2(x)


class DSM_LocalAttention(nn.Module):
    """Supplied centered-feature interaction; unused sig/num_patch retained."""

    def __init__(self, channel, p):
        super().__init__()
        self.channel = channel
        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return self.a * out * x + self.b * x


class DualDomainSelectionMechanism(nn.Module):
    """Supplied pyramid=1, channel-wise a=0/b=1 initial identity."""

    def __init__(self, channel):
        super().__init__()
        pyramid = 1
        self.spatial_gate = DSM_SpatialGate(channel)
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid - 1, -1, -1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a * out + self.b * x


class MutilScaleEdgeInformationSelect(MutilScaleEdgeInformationEnhance):
    """Same supplied edge branches, then DSM before the final projection."""

    def __init__(self, inc, bins=(3, 6, 9, 12)):
        super().__init__(inc, bins)
        self.dsm = DualDomainSelectionMechanism(inc * 2)

    def forward(self, x):
        spatial_size = x.shape[2:]
        out = [self.local_conv(x)]
        for feature, edge in zip(self.features, self.ees):
            resized = F.interpolate(feature(x), spatial_size, mode="bilinear", align_corners=True)
            out.append(edge(resized))
        return self.final_conv(self.dsm(torch.cat(out, dim=1)))


class _MSSelectResidualBlock(nn.Module):
    """Preserve the existing x + enhancement shortcut around Select."""

    def __init__(self, dim, shortcut=True, bins=(3, 6, 9, 12)):
        super().__init__()
        self.enhance = MutilScaleEdgeInformationSelect(dim, bins)
        self.add = bool(shortcut)

    def forward(self, x):
        delta = self.enhance(x)
        return x + delta if self.add else delta


class C3k2_MSSelect(C3k2):
    """C3k2 interface: c1,c2,n,c3k,e,g,shortcut; current c3k=False/g=1."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 bins=(3, 6, 9, 12)):
        if c3k or g != 1:
            raise ValueError("This adapter requires c3k=False and g=1")
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            _MSSelectResidualBlock(self.c, shortcut, bins) for _ in range(n)
        )
