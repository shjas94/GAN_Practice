import torch.nn as nn


class BasicBlock(nn.Module):
    f'''
    Nearly Same as the block in Discriminator
    except for skip connection, ReLU activation
    '''

    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, self.padding, stride)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        return self.relu(out)


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
            nn.Upsample(scale_factor=2),
            BasicBlock(1024, 512, 3, 1),
            nn.Upsample(scale_factor=2),
            BasicBlock(512, 256, 3, 1),
            nn.Upsample(scale_factor=2),
            BasicBlock(256, 128, 3, 1),
            nn.Upsample(scale_factor=2),
            BasicBlock(128, 3, 3, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, z):
        projected_z = self.linear(z)
        projected_z = projected_z.view(
            projected_z.shape[0], self.expanded_dim, self.img_size//16, self.img_size//16)
        out = self.model(projected_z)
        return self.tanh(out)
