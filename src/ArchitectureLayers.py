import torch
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
        return x.reshape(batch_size, -1)

class ConvBlock(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linearshape = nn.Linear(8192, 3072)
        self.act4 = nn.ReLU()

    def forward(self, x, batch_size=64):
        x = self.linearshape(x.reshape(batch_size, -1))
        return self.act4(x).reshape(batch_size, 3, 32, 32)

