"""
# Features in DC-GAN:

- use stride > 1 conv to replace pooling layer
- remove full connect network
- all activate function expect the last layer is Leaky relu
- lr = 2e-4
"""

import torch.nn as nn


class BasicUpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, bn=True, activation=True):
        super(BasicUpsampleBlock, self).__init__()

        layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                                     bias=False)]

        if bn:
            layers.append(nn.BatchNorm2d(out_channel))

        if activation:
            layers.append(nn.LeakyReLU(0.2, True))

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsample(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super(Generator, self).__init__()

        # _in_channel = 512
        # layers = [BasicUpsampleBlock(in_channel, _in_channel, kernel_size=4, stride=1, padding=0)]
        #
        # # upsampling
        # cfg = [512, 256, 128, 64]
        # for _out_channel in cfg:
        #     layers.append(BasicUpsampleBlock(_in_channel, _out_channel))
        #     _in_channel = _out_channel
        #
        # # last conv
        # layers += [
        #     BasicUpsampleBlock(64, out_channel, bn=False, activation=False),
        #     nn.Tanh()
        # ]
        #
        # self.conv = nn.Sequential(*layers)
        self.conv = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(in_channel, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )

    def forward(self, x):
        return self.conv(x)




if __name__ == '__main__':
    import torch
    tensor = torch.rand((5, 100, 1, 1))
    print(tensor.shape)

    G = Generator(100)
    out = G(tensor)
    print(out.shape)

