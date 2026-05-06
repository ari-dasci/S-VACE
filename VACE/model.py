import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import RevIN1d


class PatchChannelEncoder(nn.Module):
    """Channel-aware patch encoder with depthwise-separable first stage.

    Stage 1: per-channel depthwise convolution (no cross-channel mixing).
    Stage 2: pointwise 1x1 convolution to mix channels.
    Remaining stages: standard Conv1d blocks.
    """

    def __init__(self, in_channels=1, projection_dim=256,
                 layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin=True,
                 revin_affine=False,
                 revin_eps=1e-5,
                 revin_min_sigma=1e-5,
                 channel_expansion=8,
                 use_bn=True):
        super().__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim
        self.in_channels = in_channels

        norm = lambda c: nn.BatchNorm1d(c) if use_bn else nn.Identity()

        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        # Stage 1: depthwise conv — each channel independently (groups=in_channels)
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * channel_expansion,
                      kernel_size=kss[0], stride=1,
                      padding=kss[0] // 2, bias=False,
                      groups=in_channels),
            norm(in_channels * channel_expansion),
            nn.ReLU(inplace=True)
        )

        # Stage 2: pointwise conv — mix channels
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels * channel_expansion, layers[0],
                      kernel_size=1, bias=False),
            norm(layers[0]),
            nn.ReLU(inplace=True)
        )

        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(layers[i-1], layers[i],
                          kernel_size=kss[i], stride=1,
                          padding=kss[i] // 2, bias=False),
                norm(layers[i]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(layers))
        ])

        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x, return_embedding=False, return_projection=False):
        if self.revin is not None:
            x = self.revin.norm(x)

        x = self.depthwise(x)
        x = self.pointwise(x)

        for block in self.convblocks:
            x = block(x)

        h = self.fc_embedding(x).flatten(start_dim=1)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("forward requires return_embedding=True or return_projection=True")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)


class PatchEncoder(nn.Module):
    """Baseline patch encoder — standard Conv1d blocks with no channel separation."""

    def __init__(self, in_channels=1, projection_dim=256,
                 layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin=True,
                 revin_affine=False,
                 revin_eps=1e-5,
                 revin_min_sigma=1e-5):
        super().__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(layers[i - 1] if i > 0 else in_channels, layers[i],
                          kernel_size=kss[i], stride=1, padding=kss[i] // 2, bias=False),
                nn.BatchNorm1d(layers[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(layers))
        ])

        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(layers[-1], projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x, return_embedding=False, return_projection=False):
        if self.revin is not None:
            x = self.revin.norm(x)

        for block in self.convblocks:
            x = block(x)

        h = self.fc_embedding(x).flatten(start_dim=1)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("forward requires return_embedding=True or return_projection=True")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        return self.projection_head(h)
