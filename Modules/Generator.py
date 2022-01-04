import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, self.padding, stride)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.convtranspose = nn.ConvTranspose2d(
            out_channel, out_channel, 4, stride=2, padding=1)

        # projection & upsampling for skip connection
        self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.convtranspose(out)
        return out + self.upsample(self.projection(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, self.padding, stride)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out + self.projection(x)


class Generator(nn.Module):
    def __init__(self, input_dim, expanded_dim, img_size):
        f'''
        input noise : 100-d vector
        img_shape : |H| or |W| of output image
        |output| = (64 x 64 x 3)
        '''
        super(Generator, self).__init__()
        self.expanded_dim = expanded_dim
        self.img_size = img_size
        self.linear = nn.Linear(
            input_dim, self.expanded_dim * (self.img_size // 16) ** 2)
        self.model = nn.Sequential(
            BasicBlock(self.expanded_dim, 1024, 3, 1),
            BasicBlock(1024, 512, 3, 1),
            BasicBlock(512, 256, 3, 1),
            BasicBlock(256, 128, 3, 1),
            ResidualBlock(128, 3, 3, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, z):
        projected_z = self.linear(z)
        projected_z = projected_z.view(
            projected_z.shape[0], self.expanded_dim, self.img_size//16, self.img_size//16)
        out = self.model(projected_z)
        return self.tanh(out)
