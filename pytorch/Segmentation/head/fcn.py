from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1
    ):
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )


class FCNHead(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 128, num_classes: int = 19):
        super().__init__()
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
        )
        self.cls = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        hidden_features = self.conv(features[-1])
        hidden_features = self.cls(hidden_features)
        return hidden_features
      
 
if __name__ == '__main__':
  in_channels = [64, 128, 256, 512]
  out_channels = 128
  num_classes = 19
  features = [torch.rand(2, f, 224 // 2**i, 224 // 2**i) for i, f in enumerate(in_channels)]
  fcn_head = FCNHead(in_channels=in_channels[-1], hidden_size=out_channels, num_classes=num_classes)
  print(fcn_head(features).shape)
