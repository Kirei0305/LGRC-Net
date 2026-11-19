"""LGRC-Net shared detection head with the complete FRM removed.

Install at ultralytics/nn/extra_modules/lgrc_no_frm_head.py.
Uses the existing, working P3NativeReg head and its complete decoder.
LRM is a Neck module and is controlled separately by the YAML.
"""

import torch
import torch.nn as nn

from .p3_native_reg_head import Detect_LSDECD_P3NativeReg

__all__ = ("Detect_LSDECD_NoFRM",)


class _SharedFeatureOnly(nn.Module):
    """Remove the native-reference fusion without adding parameters."""

    def forward(self, shared_feature, unused_native):
        return shared_feature


class Detect_LSDECD_NoFRM(Detect_LSDECD_P3NativeReg):
    """Three input/output scales; shared box/class heads; no FRM parameters."""

    def __init__(self, nc=80, hidc=256, ch=()):
        if (
            len(ch) != 3
            or any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in ch)
        ):
            raise ValueError("NoFRM needs three input channels [P3, P4, P5]")
        # Reuse the working parent initialization and common state_dict names.
        # This fourth channel is constructor compatibility, not a graph input.
        super().__init__(nc=nc, hidc=hidc, ch=tuple(ch) + (ch[0],))
        self.p3_native_reg_fuse = _SharedFeatureOnly()
        self.p3_reg_refine = nn.Identity()
        self.use_p3_reg = False
        if self.nl != 3:
            raise ValueError("NoFRM must keep three detection scales")

    def forward(self, x):
        if (
            not isinstance(x, (list, tuple))
            or len(x) != 3
            or any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in x)
        ):
            raise ValueError("NoFRM expects three tensors [P3, P4, P5]")
        # Both parent FRM operations now return their first input unchanged.
        # None is ignored; no native P3 feature is read. Reuse decoding/export.
        return super().forward([x[0], x[1], x[2], None])
