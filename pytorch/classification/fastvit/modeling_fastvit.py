import copy
import heapq
from typing import Optional, Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import MobileOneBlock, reparameterize_model
from replknet import ReparamLargeKernelConv


def convolutional_stem(
    in_channels: int,
    out_channels: int,
    inference_mode: bool = False,
) -> nn.Sequential:
    """Convolutional stem with MobileOne blocks
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        inference_mode (bool): Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        blocks (nn.Sequential): Stem blocks
    """
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class RepMixer(nn.Module):
    """Re-parameterizable token mixer

    Paper: `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ) -> None:
        """RepMixer Module

        Args:
            dim (int): Dimension input. :math:`C_{int}` from an expected input of size :math: `(B, C_{in}, H, W)`.
            kernel_size (int): Kernel size for spatial mixing. Default: 3.
            use_layer_scale (bool): Learnable layer scale. Default: ``True``.
            layer_scale_init_value (float): Initial value for layer scale. Default: 1e-5.
            inference_mode (bool): Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True
            )
        else:
            self.norm = MobileOneBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)),
                    requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm)
            else:
                x = x + self.mixer(x) - self.norm(x)
        return x

    def reparameterize(self):
        """Re-parameterize mixer and norm into single Conv2d layer
        for efficient inference
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            weight = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            bias = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            weight = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            bias = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = weight
        self.reparam_conv.bias.data = bias

        for param in self.parameters():
            param.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")

        self.inference_mode = True


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """FFN Module

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (None | int): Number of channels after expansion. Default: None.
            out_channels (None | int): Number of output channels. Default: None.
            act_layer (nn.Module): Activation layer. Default: ``GELU``
            drop (float): Dropout rate. Default: 0.0
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add(
            "bn",
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ) -> None:
        """Mixer BLock

        Args:
            dim (int): Embedding dimension.
            kernel_size (int): Kernel size for RepMixer. Default: 3.
            mlp_ratio (float): MLP expansion ratio. Default: 4.0.
            act_layer (nn.Module): Activation layer. Default: ``nn.GELU``
            drop (float): Drop rate. Default: 0.0.
            drop_path (float): Drop path rate. Default: 0.0.
            use_layer_scale (bool): Whether to apply layer scale. Default: ``True``.
            layer_scale_init_value (float): Layer scale value at initialization. Default: 1e-5
            inference_mode (bool): Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()

        self.token_mixer = RepMixer(
            dim=dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode
        )

        assert mlp_ratio > 0, (
            f"MLP ratio should be greater than 0, but got {mlp_ratio}"
        )
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn)
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class RepCPE(nn.Module):
    """Conditional positional encoding

    Paper: `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`

    Reparameterize module to remove a skip connection.
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spartial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode: bool = False,
    ) -> None:
        """Conditional Positional Encoding

        Args:
            in_channels (int): Number of input channels.
            embed_dim (int): Number of embedding dimensions. Default: 768.
            spartial_shape (int | tuple of int): Spatial shape of kernel for positional encoding. Default: (7, 7).
            inference_mode (bool): Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        if isinstance(spartial_shape, int):
            spartial_shape = (spartial_shape, spartial_shape)
        self.spartial_shape = spartial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spartial_shape,
                stride=1,
                padding=int(self.spartial_shape[0] // 2),
                groups=self.groups,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=spartial_shape,
                stride=1,
                padding=int(self.spartial_shape[0] // 2),
                groups=embed_dim,
                bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
        else:
            x = x + self.pe(x)
        return x

    def reparameterize(self) -> None:
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spartial_shape[0],
                self.spartial_shape[1],
            ),
            dtype=self.pe.weight.type,
            device=self.pe.weight.device
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spartial_shape[0] // 2,
                self.spartial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        weight = id_tensor + self.pe.weight
        bias = self.pe.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spartial_shape,
            stride=1,
            padding=int(self.spartial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = weight
        self.reparam_conv.bias.data = bias

        for param in self.parameters():
            param.detach_()
        self.__delattr__("pe")


class MHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Multi-head self-attention Layer

        Args:
            dim (int): Number of embedding dimensions.
            head_dim (int): Number of hidden dimension per head. Default: 32.
            qkv_bias (bool): Use bias or not. Default: ``False``.
            attn_drop (float): Dropout rate for attention tensor.
            proj_drop (float): Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, (
            f"dim={dim} should divisible by head_dim={head_dim}"
        )
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(
            in_features=dim,
            out_features=dim * 3,
            bias=qkv_bias
        )
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(
            in_features=dim,
            out_features=dim,
        )
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, emb_dim, height, width = x.shape
        num_patches = height * width
        x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (batch_size, num_patches, emb_dim)
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # trick to make q @ k.T more stable
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, emb_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(batch_size, emb_dim, height, width)

        return x


class AttentionBlock(nn.Module):
    """MetaFormer block with MHSA as token mixer

    Paper: `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        """Attention Block

        Args:
            dim (int): Embedding dimension.
            mlp_ratio (float): MLP expansion ratio. Default: 4.0.
            act_layer (nn.Module): Activation layer. Default: ``nn.GELU``.
            norm_layer (nn.Module): Normalization layer. Default: ``nn.BatchNorm2d``.
            drop (float): Dropout rate. Default: 0.0.
            drop_path (float): Drop path rate. Default: 0.0.
            use_layer_scale (bool):  Whether to apply layer scale. Default: ``True``.
            layer_scale_init_value (float): Layer scale value at initialization. Default: 1e-5
        """
        super().__init__()

        self.norm = norm_layer
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, (
            f"MLP ratio should be greater than 0, but got {mlp_ratio}"
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.use_layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
                requires_grad=True
            )
            self.use_layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
                requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(self.use_layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.use_layer_scale_2 * self.convffn(self.norm(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(self.norm(x)))
        return x


def basic_block(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode: bool = False,
) -> nn.Sequential:
    """Build FastViT blocks withing a stage

    Args:
        dim (int): Embeddings dimension.
        block_index (int): Block index.
        num_blocks (List[int]): Number of blocks per stage.
        token_mixer_type (str): Token mixer type.
        kernel_size (int): Kernel size.
        mlp_ratio (float): MLP expasion ratio.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Drop path rate.
        use_layer_scale (bool): Flag to turn on layer scale regularization.
        layer_scale_init_value (float): Layer scale value at initialization.
        inference_mode (bool): Whether the model at the inference mode
    Returns:
        layers (nn.Sequential): Blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx * sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    user_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    user_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type {} not supported.".format(token_mixer_type)
            )
        blocks = nn.Sequential(*blocks)
        return blocks


class PatchEmbed(nn.Module):
    """Convolutional Patch Embedding Layer"""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
    ) -> None:
        """Patch Embedding Layer

        Args:
            patch_size (int): Embedding layer's patch size.
            stride (int): Embedding layer's stride.
            in_channels (int): Number of channels of input tensor.
            embed_dim (int): Number of embedding dimensions.
            inference_mode (bool): Whether the model at the inference mode.
        """
        super().__init__()
        block = []
        block.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode
            )
        )
        block.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class FastViT(nn.Module):
    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims: Optional[int] = None,
        mlp_ratios: Optional[float] = None,
        downsamples: Optional[List] = None,
        repmixer_kernle_size: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes: int = 1000,
        pos_embs: Optional[List] = None,
        down_patch_size: int = 7,
        down_stride: int = 2,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        fork_feat: bool = False,
        init_cfg: Optional[Dict] = None,
        pretrained: Optional = None,
        cls_ratio: float = 2.0,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        # stem
        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode)

        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims[i], embed_dims[i], inference_mode=inference_mode
                    )
                )
            stage = basic_block(
                dim=embed_dims[i],
                block_index=i,
                num_blocks=layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernle_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)

            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages
            if downsamples[i] or embed_dims[i] !=embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                    )
                )
        self.network = nn.ModuleList(network)

        # For segmentation and detection, extract intermidiate output
        if self.fork_feat:
            # Add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 or os.environ.get("FORK_LAST3", None):
                    # For RetinaNet, `start_level=1`. The first norm layer will not used.
                    # cmd: `FORK_LAST3=1 python -m torch.distributed.lauch ...`
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.conv_exp = MobileOneBlock(
                in_channels=embed_dims[-1],
                out_channels=int(embed_dims[-1] * cls_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                groups=embed_dims[-1],
                inference_mode=inference_mode,
                use_se=True,
                num_conv_branches=1
            )
            self.gap = nn.AdaptiveAvgPool3d(output_size=1)
            self.head = (
                nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes)
                if num_classes > 0 else nn.Identity()
            )

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _scrub_checkpoint(checkpoint, model):
        sterile_dict = {}
        for k1, v1 in checkpoint.items():
            if k1 not in model.state_dict():
                continue
            if v1.shape == model.state_dict()[k1].shape:
                sterile_dict[k1] = v1
        return sterile_dict

    def init_weights(self, pretrained: str = None) -> None:
        if self.init_cfg is None and pretrained is None:
            pass
        else:
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg["checkpoint"]
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt

            sterile_dict = FastViT._scrub_checkpoint(_state_dict)
            state_dict = sterile_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outputs.append(x_out)
        if self.fork_feat:
            return outputs
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.forward_embeddings(x)
        # Backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        cls_out = self.head(x)
        return cls_out
