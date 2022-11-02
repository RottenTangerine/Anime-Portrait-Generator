from model.basic_block import BasicConv
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()

        layers = []
        cfd = [64, 128, 256, 512]
        for out_channel in cfd:
            layers.append(BasicConv(in_channel, out_channel, 4, 2, 1))
            in_channel = out_channel

        self.conv = nn.Sequential(*layers)
        self.out = nn.Sequential(
            BasicConv(in_channel, in_channel, 3, 1, 1),
            nn.Conv2d(in_channel, 1, kernel_size=4, stride=2, padding=1),
        )
        self.c = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        x = self.c(x)
        x = x.view(x.shape[0], -1)
        return x

