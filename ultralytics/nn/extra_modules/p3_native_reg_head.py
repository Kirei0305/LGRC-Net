# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors
from .head import Detect_LSDECD

__all__ = ('P3NativeRegFuse', 'Detect_LSDECD_P3NativeReg')

class P3NativeRegFuse(nn.Module):
    """Linear concatenation residual, with no mutation or resolution change."""

    def __init__(self, feature_channels, native_channels, residual_scale=0.1):
        super().__init__()
        for channels in (feature_channels, native_channels):
            if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
                raise ValueError("Channel counts must be positive integers")
        if not 0 < float(residual_scale) <= 1:
            raise ValueError("residual_scale must be in (0,1]")
        self.feature_channels = feature_channels
        self.native_channels = native_channels
        self.residual_scale = float(residual_scale)
        with torch.random.fork_rng(devices=[]):
            self.mix = nn.Conv2d(feature_channels + native_channels,
                                 feature_channels, 1, bias=False)
            nn.init.zeros_(self.mix.weight)

    def forward(self, feature, native):
        if (not isinstance(feature, torch.Tensor) or not isinstance(native, torch.Tensor)
                or feature.ndim != 4 or native.ndim != 4):
            raise ValueError("Expected shared P3 and native P3 in [B,C,H,W]")
        if feature.shape[1] != self.feature_channels or native.shape[1] != self.native_channels:
            raise ValueError("P3 channels disagree with tasks.py registration")
        if feature.shape[0] != native.shape[0] or feature.shape[-2:] != native.shape[-2:]:
            raise ValueError("Native P3 must have the same batch/H/W as shared P3")
        if feature.device != native.device or not feature.is_floating_point() or not native.is_floating_point():
            raise ValueError("P3 inputs must be floating tensors on the same device")
        native = native.to(dtype=feature.dtype)  # AMP may give the two paths different dtypes
        delta = self.mix(torch.cat((feature, native), dim=1))
        return feature + self.residual_scale * delta.to(dtype=feature.dtype)


class Detect_LSDECD_P3NativeReg(Detect_LSDECD):
    """Four feature inputs, but still THREE detection outputs and strides."""

    def __init__(self, nc=80, hidc=256, ch=()):
        if (len(ch) != 4 or any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in ch)
                or ch[3] != ch[0]):
            raise ValueError("ch must be [P3,P4,P5,native_P3], with ch[3]==ch[0]")
        super().__init__(nc, hidc, ch[:3])  # Auxiliary native input does NOT create a scale.
        reg = getattr(self, "p3_reg_refine", None)
        if type(reg).__name__ != "P3RegRefine" or not getattr(self, "use_p3_reg", True):
            raise ValueError("Restore original P3RegRefine in head.py Detect_LSDECD and enable it")
        if getattr(self, "use_p3_cls", False):
            raise ValueError("This experiment expects exp78 with P3ClsContext disabled")
        self.p3_native_reg_fuse = P3NativeRegFuse(hidc, ch[3], residual_scale=0.1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 4:
            raise ValueError("Head needs [fused_P3,fused_P4,fused_P5,layer16_native_P3]")
        native = x[3]
        x = list(x[:3])  # Do not overwrite any saved Neck/selector input.
        for i in range(self.nl):
            feat = self.share_conv(self.conv[i](x[i]))
            reg_feat = feat
            if i == 0:
                reg_feat = self.p3_native_reg_fuse(feat, native)
                reg_feat = self.p3_reg_refine(reg_feat)  # Retain original3x3 P3Reg.
            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(feat)), dim=1)
        if self.training:
            return x

        # Matches the photographed LSDECD inference/decode/export body.
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)
        if self.export and self.format in ("tflite", "edgetpu"):
            img_h, img_w = shape[2:]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
