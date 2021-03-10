import torch
import torch.nn as nn
from utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
​
__all__ = [
    "vgg",
    "vgg11", "vgg11_bn", 
    "vgg13", "vgg13_bn",  
    "vgg16_C", "vgg16_bn_C",
    "vgg16", "vgg16_bn", 
    "vgg19", "vgg19_bn" 
]
model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}
class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        head: nn.Module = None
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.head = head
        if self.head is None:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(in_features=4096, out_features=num_classes),
            )
        elif (init_weights is False) and self.head:
            _initialize_weights(self.head)
        if init_weights:
            _initialize_weights(self)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        if self.head  is not None:
            x = self.head (x)
        else:
            x = self.classifier(x)
        return x
​
​
def _initialize_weights(module) -> None:
    for m in module.modules():
        print(m)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.normal_(tensor=m.weight, mean=0., std=0.01)
                nn.init.constant_(tensor=m.bias, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(tensor=m.weight, val=1.)
            nn.init.constant_(tensor=m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(tensor=m.weight)
            if m.bias is not None:
                nn.init.constant_(tensor=m.bias, val=0.)
                
                    
def make_layers(cfg: List[Union[str, int, List[int]]], batch_norm: bool=False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    layer = 0
    if conv1:
        for v in cfg:
            if v ==  "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif layer in [8, 12, 16]:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(num_features=v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(num_features=v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            layer += 1
    else:
        for v in cfg:
            if v ==  "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(num_features=v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)
​
​
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}
​
​
def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, conv1: bool, head: nn.Module=None, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, conv1=conv1), head=head, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        print(model.load_state_dict(state_dict, strict=False))
    return model
​
​
def vgg11(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 11-layer model (configuration "A")')
    return _vgg("vgg11", "A", False, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg11_bn(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 11-layer model (configuration "A")')
    return _vgg("vgg11", "A", True, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg13(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 13-layer model (configuration "B")')
    return _vgg("vgg13", "B", False, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg13_bn(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 13-layer model (configuration "B")')
    return _vgg("vgg13", "B", True, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg16_C(pretrained: bool=False, progress: bool=False, conv1: bool=True, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "C") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): Pre-trained model is not supported
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): If true conv2d 1x1 applied (no pre-trained model)
        head (nn.Module): Custom classifier head
    """
    if pretrained:
        raise ValueError('Pre-trained mode is not available for VGG 16-layer model (configuration "C"), got pre-trained is {}'.format(pretrained))
    return _vgg("vgg16", "C", False, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg16_bn_C(pretrained: bool=False, progress: bool=False, conv1: bool=True, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "C") with batch normalization from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): Pre-trained model is not supported
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): If true conv2d 1x1 applied (no pre-trained model)
        head (nn.Module): Custom classifier head
    """
    if pretrained:
        raise ValueError('Pre-trained mode is not available for VGG 16-layer model (configuration "C"), got pre-trained is {}'.format(pretrained))
    return _vgg("vgg16", "C", True, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg16(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        conv1 (bool): If true conv2d 1x1 applied (no pre-trained model)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 16-layer model (configuration "D")')
    return _vgg("vgg16", "D", False, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg16_bn(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        conv1 (bool): If true conv2d stride=(1x1) applied (no pre-trained model)
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 16-layer model (configuration "D")')
    return _vgg("vgg16_bn", "E", True, pretrained, progress, conv1, head, **kwargs)
​
​
def vgg19(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E") with batch normalization from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 19-layer model (configuration "E")')
    return _vgg("vgg19", "E", True, pretrained, progress, conv1, head, **kwargs)
True
​
def vgg19_bn(pretrained: bool=False, progress: bool=False, conv1: bool=False, head: nn.Module=None, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E") with batch normalization from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        conv1 (bool): Conv2d with stride=(1x1) does not supported in this model
        head (nn.Module): Custom classifier head
    """
    if conv1:
        raise ValueError('Conv2d with stride=(1x1) is not implemented in VGG 19-layer model (configuration "E")')
    return _vgg("vgg19\bn", "E", True, pretrained, progress, conv1, head, **kwargs)
