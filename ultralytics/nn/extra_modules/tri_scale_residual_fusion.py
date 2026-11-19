# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = ('TriScaleResidualFusion', 'ScaleFeatureSelect')

def _resize(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
        if x.shape[-2] % size[0] == 0 and x.shape[-1] % size[1] == 0:
            kernel = (x.shape[-2] // size[0], x.shape[-1] // size[1])
            return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class TriScaleResidualFusion(nn.Module):
    """Three tensors in; three locally anchored, cross-scale residual tensors out."""

    def __init__(self, inc, hidden=16, residual_scale=0.1):
        super().__init__()
        if (not isinstance(inc, (list, tuple)) or len(inc) != 3
                or any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in inc)):
            raise ValueError("inc must contain three positive integer channel counts")
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not math.isfinite(float(residual_scale)) or not 0 < float(residual_scale) <= 1:
            raise ValueError("residual_scale must be finite and in (0,1]")
        self.channels = tuple(inc)
        self.hidden = hidden
        self.residual_scale = float(residual_scale)
        self.enabled = True
        joint = 3 * hidden
        with torch.random.fork_rng(devices=[]):
            self.projections = nn.ModuleList(Conv(c, hidden, 1) for c in inc)
            self.local = nn.ModuleList(Conv(joint, joint, 3, g=joint) for _ in inc)
            self.expand = nn.ModuleList(Conv(joint, c, 1, act=False) for c in inc)

    def _validate(self, x):
        if (not isinstance(x, (list, tuple)) or len(x) != 3
                or any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in x)):
            raise ValueError("Expected three [B,C,H,W] inputs [NeckP3,NeckP4,NeckP5]")
        if tuple(t.shape[1] for t in x) != self.channels:
            raise ValueError("Input channels disagree with the tasks.py registration")
        if any(t.shape[0] != x[0].shape[0] or t.device != x[0].device
               or t.dtype != x[0].dtype or not t.is_floating_point() for t in x):
            raise ValueError("Inputs must share batch size, floating dtype and device")

    def forward(self, x):
        self._validate(x)
        if not self.enabled:
            return list(x)
        embeddings = [project(t) for project, t in zip(self.projections, x)]
        outputs = []
        for original, local, expand in zip(x, self.local, self.expand):
            aligned = [_resize(z, original.shape[-2:]) for z in embeddings]
            delta = expand(local(torch.cat(aligned, dim=1)))
            outputs.append(original + self.residual_scale * delta.to(dtype=original.dtype))
        return outputs


class ScaleFeatureSelect(nn.Module):
    """Parameter-free selector for a tensor list saved by Ultralytics BaseModel."""

    def __init__(self, index):
        super().__init__()
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 3:
            raise ValueError("index must be 0(P3),1(P4),2(P5)")
        self.index = index

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("ScaleFeatureSelect requires the three-output fusion list")
        return x[self.index]
