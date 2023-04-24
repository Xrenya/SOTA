from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 128,
        scales: Tuple[int] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1)
            ) for scale in scales
        ])

        self.bottleneck = ConvModule(
            in_channels=in_channels + len(scales) * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        for stage in self.stages:
            outputs.append(
                F.interpolate(stage(features), size=features.shape[-2:], mode='bilinear', align_corners=True)
            )

        outputs = [features] + outputs[::-1]
        output = self.bottleneck(torch.cat(outputs, dim=1))
        return output


# model = PPM(512, 128)
# x = torch.rand(2, 512, 7, 7)
# print(model(x).shape)


class UPerHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int,
        num_classes: int = 19,
        scales: Tuple[int] = (1, 2, 3, 6),
    ):
        super().__init__()
        # PPM (Pyramid Pooling Module) module
        self.ppm = PPM(in_channels=in_channels[-1], hidden_dim=hidden_dim, scales=scales)

        # FPN (Feature Pyramid Network) module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for in_channel in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_channels=in_channel, out_channels=hidden_dim, kernel_size=1))  # channel reduction
            self.fpn_out.append(ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1))  # 'same'

        self.bottleneck = ConvModule(in_channels=len(in_channels) * hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)  # 'same'
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv_seg = nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1)

    def forward(self, features: Tuple[torch.Tensor]) -> torch.Tensor:
        ppm_features = self.ppm(features[-1])

        fpn_features = [ppm_features]

        for i in range(len(self.fpn_in) - 1, -1, -1):
            feature = self.fpn_in[i](features[i])
            ppm_features = feature + F.interpolate(ppm_features, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](ppm_features))

        fpn_features.reverse()
        for i in range(1, len(fpn_features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)

        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output
