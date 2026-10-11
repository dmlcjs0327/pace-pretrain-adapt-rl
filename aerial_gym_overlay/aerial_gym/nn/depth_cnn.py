"""
Small CNN to encode depth images into a fixed-size latent vector for RL observation.
Input: (N, 1, H, W) depth image, optionally normalized to [0, 1].
Output: (N, latent_dim).
"""
import torch
import torch.nn as nn
from typing import Tuple

class DepthCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        input_height: int = 135,
        input_width: int = 240,
        channels: Tuple[int, ...] = (32, 64, 64),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        # Conv blocks: reduce spatial size and increase channels
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            kernel = 5 if i == 0 else 3
            stride = 2 if i == 0 else 2
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm2d(c_out),
                nn.ELU(inplace=True),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        # Adaptive pool to fixed size so any input size works after enough downsampling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
