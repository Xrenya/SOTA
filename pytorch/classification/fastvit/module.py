import copy
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileOneBlock", "reparameterize_model"]


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """A squeeze and excitation block
        Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)

        Args:
            in_channels (int): Number of input channels.
            rd_ratio (float): Reduction ratio.
        """
        super().__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = inputs.shape
        x = F.avg_pool2d(inputs, kernel_size=(height, width))
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, channels, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU()
    ) -> None:
        """MobileOne Block

        Args:
            in_channels (int): Number channels in the input.
            out_channels (int): Number channels produced by the block.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride size.
            padding (int): Zero-padding size .
            dilation (int): Kernel dilation factor.
            groups (int): Group number.
            inference_mode (bool): If True, instantiates model in inference mode.
            use_se (bool): Whether to use SE-ReLU activations.
            use_act (bool): Whether to use activations. Default: ``True``.
            use_scale_branch (bool): Whether to use scale branch. Default: ``True``.
            num_conv_branches (int): Number of linear conv branches.
            activation (nn.Module): Activation function. Default: ``nn.GELU()``.
        """
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # use SE-ReLU block
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        # use activation function
        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        # inference mode
        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # re-parameterizable skip connection
            self.rbr_skip = (
                nn.BatchNorm(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # re-parameterizable conv branches
            if num_conv_branches > 0:
                rbr_conv = []
                for _ in range(num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            # re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reparameterize(self):
        """Re-parameterize multi-branched achitecture used at the
        training time to obtain a plane CNN-like structure for inference
        Paper: [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)

        Returns:
            None
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # delete unused branches
        for param in self.parameters():
            param.detach()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")
        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-parameterized kernel and bias

        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L90

        Returns:
            Tuple of (kernel, bias) after fusion branches
        """
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # pad scale branch kernel to match conv branch kernel size
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for i in range(self.num_conv_branches):
                kernel, bias = self._fuse_bn_tensor(self.rbr_conv[i])
                kernel_conv += kernel
                bias_conv += bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self,
        branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse BatchNorm2d layer with preceeding Conv2d layer.

        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L102

        Args:
            branch (Union[nn.Sequential, nn.BatchNorm2d]): Sequences of ops to be fuse

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.type,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Construct Conv2d-BatchNorm2D layer
        Args:
            kernel_size (int): Size of the convolution kernel
            padding (int): Zero-padding size

        Returns:
            Conv2d-BN module
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False
            )
        )
        mod_list.add_module(
            "bn",
            nn.BatchNorm2d(num_features=self.out_channels)
        )
        return mod_list


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Re-parameterize model into a single branch for inference
    Args:
        model (nn.Module): MobileOne model in train mode

    Returns:
        model (nn.Module): MobileOne in inference mode
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model
