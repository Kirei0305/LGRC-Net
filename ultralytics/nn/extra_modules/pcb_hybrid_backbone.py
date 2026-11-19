# Ultralytics YOLO, AGPL-3.0 license
# Model definitions extracted unchanged from the user-uploaded 2026-10-18 snapshot.
# Historical experiment instructions and unrelated self-checks are archived separately.

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

__all__ = ('LiteP5Encoder',)


def _position_2d(height, width, dim, device, dtype):
    """Sine/cosine x/y positions, matching flatten(2) row-major token order."""
    if dim % 4:
        raise ValueError("Position embedding dimension must be divisible by four")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4)
    omega = torch.pow(10000.0, -omega)
    px = xx.reshape(-1, 1) * omega.reshape(1, -1)
    py = yy.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((px.sin(), px.cos(), py.sin(), py.cos()), dim=1).to(dtype).unsqueeze(0)


class ConvGLUFFN(nn.Module):
    """1x1 split -> local DW3x3/GELU value times raw gate -> 1x1."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.expand = nn.Conv2d(dim, 2 * hidden, 1, bias=True)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden, dim, 1, bias=True)

    def forward(self, x):
        value, gate = self.expand(x).chunk(2, dim=1)
        return self.project(self.act(self.local(value)) * gate)


class P5EncoderLayer(nn.Module):
    """Pre-LN self-attention residual and pre-LN convolutional FFN residual."""

    def __init__(self, dim, heads, ffn_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ConvGLUFFN(dim, max(1, round(dim * ffn_ratio)))

    def forward(self, x):
        batch, dim, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        position = _position_2d(height, width, dim, x.device, normalized.dtype)
        qk = normalized + position
        attended = self.attn(qk, qk, normalized, need_weights=False)[0]
        tokens = tokens + attended
        local_input = self.norm2(tokens).transpose(1, 2).reshape(batch, dim, height, width)
        local_output = self.ffn(local_input).flatten(2).transpose(1, 2)
        return (tokens + local_output).transpose(1, 2).reshape(batch, dim, height, width)


class LiteP5Encoder(nn.Module):
    """Compress P5 -> self-attention/ConvGLU -> expand -> residual P5 output."""

    def __init__(self, c1, c2, n=1, embed_ratio=0.25, num_heads=4,
                 ffn_ratio=2.0, residual_scale=0.1):
        super().__init__()
        if min(c1, c2, n, num_heads) < 1:
            raise ValueError("Channels, encoder depth and head count must be positive")
        if not math.isfinite(float(embed_ratio)) or not 0 < embed_ratio <= 1:
            raise ValueError("embed_ratio must be finite and in (0,1]")
        if not math.isfinite(float(ffn_ratio)) or ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive and finite")
        if not math.isfinite(float(residual_scale)) or not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be finite and in (0,1]")
        divisor = math.lcm(4, num_heads)
        self.embed_dim = math.ceil(max(16, c2 * embed_ratio) / divisor) * divisor
        self.num_heads = num_heads
        self.residual_scale = float(residual_scale)
        self.skip = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, act=False)
        self.reduce = Conv(c1, self.embed_dim, 1)
        self.layers = nn.ModuleList(P5EncoderLayer(self.embed_dim, num_heads, ffn_ratio)
                                    for _ in range(n))
        self.project = Conv(self.embed_dim, c2, 1, act=False)

    def forward(self, x):
        original = self.skip(x)
        hidden = self.reduce(x)
        for layer in self.layers:
            hidden = layer(hidden)
        delta = self.project(hidden)
        return original + self.residual_scale * delta.to(dtype=original.dtype)
