import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class marknet1(nn.Module):
    # initializers
    def __init__(self):
        super(marknet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (4,4), (2,2), (1,1))
        self.conv2 = nn.Conv2d(64, 64*2, (4,4), (2,2), (1,1))
        self.conv2_bn = nn.BatchNorm2d(64*2)
        self.conv3 = nn.Conv2d(64*2, 64*4, (4,4), (2,2), (1,1))
        self.conv3_bn = nn.BatchNorm2d(64*4)
        self.conv4 = nn.Conv2d(64*4, 64*8, (4,4), (2,2), (1,1))
        self.conv4_bn = nn.BatchNorm2d(64*8)
        self.conv5 = nn.Conv2d(64*8, 64*16, (4,4), (2,2), (1,1))

        self.linear = nn.Linear(64*64, 2*68)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = x.reshape(x.size(0), 64*64)
        x = self.linear(x)

        return x


class marknet4(nn.Module):
    # initializers
    def __init__(self):
        super(marknet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (4,4), (2,2), (1,1))
        self.conv2 = nn.Conv2d(64, 64*2, (4,4), (2,2), (1,1))

        self.linear = nn.Linear(64*512, 2*68)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.reshape(x.size(0), 64*512)
        x = self.linear(x)

        return x
