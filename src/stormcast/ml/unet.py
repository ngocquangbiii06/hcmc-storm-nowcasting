"""Compact U-Net (Ronneberger et al., 2015) for 256x256 inputs."""
from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = 1, base: int = 32, depth: int = 4, sigmoid_output: bool = True
    ):
        super().__init__()
        self.sigmoid_output = sigmoid_output
        chans = [base * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, chans[0])
        self.down = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(chans[i], chans[i + 1])) for i in range(depth)
        )
        self.up = nn.ModuleList(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth)))
        self.up_conv = nn.ModuleList(DoubleConv(chans[i] * 2, chans[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        h = skips.pop()
        for up, conv in zip(self.up, self.up_conv, strict=True):
            h = conv(torch.cat([up(h), skips.pop()], dim=1))
        out = self.head(h)
        return torch.sigmoid(out) if self.sigmoid_output else out
