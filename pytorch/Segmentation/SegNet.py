class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    TO DO:
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )
        self.pool0 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)  # 256 -> 128
        
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True) # 128 -> 64
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True) # 64 -> 32
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True) # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)

        # decoder (upsampling)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512)
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128)
        )
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64)
        )
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=1)
        )

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        x0 = e0.size()
        xp0, ind0 = self.pool0(e0)
        e1 = F.relu(self.enc_conv1(xp0))
        x1 = e1.size()
        xp1, ind1 = self.pool1(e1)

        e2 = F.relu(self.enc_conv2(xp1))
        x2 = e2.size()
        xp2, ind2 = self.pool2(e2)

        e3 = F.relu(self.enc_conv3(xp2))
        x3 = e3.size()
        xp3, ind3 = self.pool3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(xp3))
        x4 = b.size()
        xp4, ind4 = self.pool4(b)

        # decoder
        d0 = F.relu(self.dec_conv0(F.max_unpool2d(input=xp4, indices=ind4, kernel_size=(2, 2), stride=2, padding=0, output_size=x4)))
        d1 = F.relu(self.dec_conv1(F.max_unpool2d(input=d0, indices=ind3, kernel_size=(2, 2), stride=2, padding=0, output_size=x3)))
        d2 = F.relu(self.dec_conv2(F.max_unpool2d(input=d1, indices=ind2, kernel_size=(2, 2), stride=2, padding=0, output_size=x2)))
        d3 = F.relu(self.dec_conv3(F.max_unpool2d(input=d2, indices=ind1, kernel_size=(2, 2), stride=2, padding=0, output_size=x1)))
        d4 = F.relu(self.dec_conv4(F.max_unpool2d(input=d3, indices=ind0, kernel_size=(2, 2), stride=2, padding=0, output_size=x0)))

        return d4
