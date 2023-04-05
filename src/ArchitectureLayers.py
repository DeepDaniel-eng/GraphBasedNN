import torch
from torch import nn
from constants.ArchitectureConstants import *

class ConnectionBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linearshape = nn.Linear(8192, 3072)
        self.act4 = nn.ReLU()
        self.batchnorm =  nn.BatchNorm2d(32)

    def forward(self, x, batch_size=64):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.batchnorm(self.pool2(x))
        return x

class ConvBlock(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Input 32 x 16 x 16 => output 3 x 32 x 32
        self.convs = nn.ModuleList([nn.Conv2d(32, 3, 5, padding=2) for i in range(4)])
        self.linearshape = nn.Linear(8192, 3072)
        self.act4 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.batchnorm =  nn.BatchNorm2d(3)


    def forward(self, x, batch_size=batch_size):
        out = torch.zeros(batch_size, 3, 32, 32).cuda()
        idxs = [[(0,0), (16,16)],
                [(0,16), (16,32)],
                [(16, 0), (32,16)],
                [(16,16), (32,32)]
                ]
        for i in range(len(self.convs)):
            conv = self.convs[i]
            sub_x = conv(x)
            start, end = idxs[i]
            out[:, :, start[0]:end[0], start[1]:end[1]] = sub_x
        return self.act4(self.batchnorm(self.drop1(out)))

