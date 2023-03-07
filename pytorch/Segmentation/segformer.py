from einops import rearrange
import torch
import torch.nn as nn
from torch import  Tensor
from torchvision.ops import StochasticDepth
from typing import List, Iterable


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm
    Rearranging channels to run LayerNorm on images
    """
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        attn_p: float = 0.,
        proj_p: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5

        self.qkv = nn.Linear(channels, 3 * channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_p)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(p=proj_p)

    def forward(self, query, key, value):
        attn = query @ key.transpose(-1, -2) * self.scale
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class EfficientMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8
    ):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=reduction_ratio,
                stride=reduction_ratio
            ),
            LayerNorm2d(channels)
        )
        self.att = MultiheadAttention(
            channels, num_heads=num_heads
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out, _ = self.att(x, reduced_x, reduced_x)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out


class MixMLP(nn.Sequential):
    def __init__(selfs, channels: int, expansion: int = 4):
        super().__init__(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=channels * expansion,
                out_channels=channels,
                kernel_size=1
            )
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x


class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = 0.0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(
                        channels, reduction_ratio, num_heads
                    )
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_prob: List[float],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            overlap_size=overlap_size
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    channels=out_channels,
                    reduction_ratio=reduction_ratio,
                    num_heads=num_heads,
                    mlp_expansion=mlp_expansion,
                    drop_path_prob=drop_prob[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


def chunks(data: Iterable, sizes: List[int]):
    cur = 0
    for size in sizes:
        chunk = data[cur:cur + size]
        cur += size
        yield chunk


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = 0.0
    ):
        super().__init__()
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions):
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegFormerDecoderBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2
    ):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        )


class SegFormerDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        widths: List[int],
        scale_factors: List[int]
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


class SegFormerSegmentationHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_features: int = 4
    ):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * num_features,
                out_channels=channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(channels)
        )
        self.predict = nn.Conv2d(
            in_channels=channels,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x


class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: List[int],
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob
        )
        self.decoder = SegFormerDecoder(
            decoder_channels,
            widths[::-1],
            scale_factors
        )
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation
    
segformer = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=100,
)

segmentation = segformer(torch.randn((1, 3, 224, 224)))
segmentation.shape # torch.Size([1, 100, 56, 56])
