class LeNet(nn.Module):
    def __init__(self, num_classes: int=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=2,
                                     stride=2)
        self.num_flat_features = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*4*4,
                             out_features=120)
        self.fc2 = nn.Linear(in_features=120,
                             out_features=84)
        self.fc3 = nn.Linear(in_features=84,
                             out_features=num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flatten(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
