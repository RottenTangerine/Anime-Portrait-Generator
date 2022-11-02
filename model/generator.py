"""
# Features in DC-GAN:

- use stride > 1 conv to replace pooling layer
- remove full connect network
- all activate function expect the last layer is Leaky relu
- lr = 2e-4
"""

import torch.nn as nn


class BasicUpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=True):
        super(BasicUpsampleBlock, self).__init__()

        layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_channel)]

        if activation:
            layers.append(nn.LeakyReLU(0.2, True))

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.upsample(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super(Generator, self).__init__()

        _in_channel = 512
        layers = [BasicUpsampleBlock(in_channel, _in_channel, kernel_size=4, stride=1, padding=0)]

        # upsampling
        cfg = [512, 256, 128, 64]
        for _out_channel in cfg:
            layers.append(BasicUpsampleBlock(_in_channel, _out_channel))
            _in_channel = _out_channel

        # last conv
        layers += [
            BasicUpsampleBlock(64, out_channel, activation=False),
            nn.Tanh()
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    import torch
    tensor = torch.rand((5, 100, 1, 1))
    print(tensor.shape)

    G = Generator(100)
    out = G(tensor)
    print(out.shape)

