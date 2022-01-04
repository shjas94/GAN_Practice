import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride, self.padding)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.leaky_relu(out)
        return out


class Discriminator(nn.Module):
    f'''
    |input| = (B x 3 x 64 x 64)
    '''

    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = BasicBlock(3, 128, 3, 2)
        self.block2 = BasicBlock(128, 256, 3, 2)
        self.block3 = BasicBlock(256, 512, 3, 2)
        self.block4 = BasicBlock(512, 1024, 3, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.gap(out).squeeze()
        out = self.linear(out)
        return self.sigmoid(out)
