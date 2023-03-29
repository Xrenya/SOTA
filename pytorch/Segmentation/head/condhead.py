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
        
        
class CondHead(nn.Module):
    def __init__(self, in_channels: int = 512, hidden_size: int = 256, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        self.weight_num = hidden_size * num_classes
        self.bias_num = num_classes

        self.conv = ConvModule(in_channels=in_channels, out_channels=hidden_size, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.1)

        self.guidance_project = nn.Conv2d(in_channels=hidden_size, out_channels=num_classes, kernel_size=1)
        self.filter_project = nn.Conv2d(in_channels=hidden_size * num_classes, out_channels=self.weight_num + self.bias_num, kernel_size=1, groups=num_classes)

    def forward(self, features):
        x = self.dropout(self.conv(features[-1]))
        B, C, H, W = x.shape
        guidance_mask = self.guidance_project(x)
        cond_logit = guidance_mask

        key = x
        value = x
        guidance_mask = guidance_mask.softmax(dim=1).view(*guidance_mask.shape[:2], -1)
        key = key.view(B, C, -1).permute(0, 2, 1)


        cond_filter = torch.matmul(guidance_mask, key)
        cond_filter /= H * W
        cond_filter = cond_filter.view(B, -1, 1, 1)
        cond_filter = self.filter_project(cond_filter)
        cond_filter = cond_filter.view(B, -1)

        weight, bias = torch.split(cond_filter, [self.weight_num, self.bias_num], dim=1)
        weight = weight.reshape(B * self.num_classes, -1, 1, 1)
        bias = bias.reshape(B * self.num_classes)

        value = value.view(-1, H, W).unsqueeze(0)
        seg_logit = F.conv2d(value, weight=weight, bias=bias, stride=1, padding=0, groups=B).view(B, self.num_classes, H, W)

        if self.training:
            return cond_logit, seg_logit
        return seg_logit
