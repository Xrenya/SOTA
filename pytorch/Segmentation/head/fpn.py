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


class FPNHead(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 128, num_classes: int = 19):
        super().__init__()
        self.lateral_convs = []
        self.output_convs = []

        for in_channel in in_channels[::-1]:
            self.lateral_convs.append(
                ConvModule(in_channels=in_channel, out_channels=out_channels, kernel_size=1)
            )

        for _ in range(len(in_channels) - 1):
            self.output_convs.append(
                ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )
        self.lateral_convs = nn.ModuleList(self.lateral_convs)
        self.output_convs = nn.ModuleList(self.output_convs)
        self.conv_seg = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]
        print(features[0].shape)
        hidden_features = self.lateral_convs[0](features[0])

        for i in range(1, len(features)):
            hidden_features = F.interpolate(hidden_features, scale_factor=2.0, mode='nearest')
            hidden_features = hidden_features + self.lateral_convs[i](features[i])
            hidden_features = self.output_convs[i - 1](hidden_features)
        hidden_features = self.conv_seg(self.dropout(hidden_features))
        return hidden_features

if __name__ == '__main__':
    in_channels = [64, 128, 256, 512]
    out_channels = 128
    num_classes = 19
    features = [torch.rand(2, f, 224 // 2**i, 224 // 2**i) for i, f in enumerate(in_channels)]
    fpn_head = FPNHead(in_channels=in_channels, out_channels=out_channels, num_classes=num_classes)
    print(fpn_head(features).shape)
