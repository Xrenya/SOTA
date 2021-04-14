# SOTA
SOTA's Implementations

This repository contains model definitions, training scripts, and other examples for Keras (Tensorflow backend) & PyTorch implementations for classification, detection, and segmentation (computer vision).

## Models

### Classification

- [ ] LeNet [Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) [Model](models/classification/lenet.py)
- [ ] AlexNet [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) [Model](models/classification/alexnet.py)
- [x] VGG16 and VGG19 [Paper](https://arxiv.org/pdf/1409.1556.pdf) [PyTorch](https://github.com/Xrenya/SOTA/blob/main/pytorch/VGG/vgg.py)
- [ ] ResNet [Paper](https://arxiv.org/pdf/1512.03385v1.pdf)
- [ ] YOLO9000 [Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [ ] DenseNet [Paper](https://arxiv.org/pdf/1608.06993.pdf)

### Detection
- [ ] Faster RCNN [Paper](https://arxiv.org/pdf/1506.01497.pdf)
- [ ] SSD [Paper](https://arxiv.org/pdf/1512.02325)
- [ ] YOLOv2 [Paper](https://arxiv.org/pdf/1612.08242.pdf)
- [ ] R-FCN [Paper](https://arxiv.org/pdf/1605.06409.pdf)

### Segmentation
|  status  |      Model     |   Paper       | Keras (tf)  | PyTorch                   
|----------|----------------|---------------|-------------|-------------
| [ ]| FCN8| [Paper](https://arxiv.org/pdf/1411.4038.pdf) | | |
- [x] SegNet [Paper](https://arxiv.org/pdf/1511.00561) [PyTorch](https://github.com/Xrenya/SOTA/blob/main/pytorch/Segmentation/SegNet.py)
- [x] U-Net [Paper](https://arxiv.org/pdf/1505.04597)[PyTorch](https://github.com/Xrenya/SOTA/blob/main/pytorch/Segmentation/UNet.py)
- [ ] E-Net [Paper](https://arxiv.org/pdf/1606.02147.pdf)
- [ ] ResNetFCN [Paper](https://arxiv.org/pdf/1611.10080.pdf)
- [ ] PSPNet [Paper](https://arxiv.org/pdf/1612.01105.pdf)
- [ ] Mask RCNN [Paper](https://arxiv.org/pdf/1703.06870.pdf)

## Datasets

### Classification

- [x] MNIST
- [ ] CIFAR-10
- [ ] MNIST-Fashion
- [ ] ImageNet
- [ ] Pascal VOC

### Detection
- [ ] Pascal VOC
- [ ] LISA Traffic Sign
- [ ] KITTI
- [ ] MSCOCO

### Segmentation
- [ ] CamVid
- [ ] Cityscapes
- [ ] Pascal VOC
- [ ] KITTI
- [ ] SYNTHIA
- [ ] GTA-V Segmentation
- [ ] MSCOCO
