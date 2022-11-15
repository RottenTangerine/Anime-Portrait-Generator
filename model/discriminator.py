from model.basic_block import BasicConv
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()

        layers = []
        # init conv
        layers.append(BasicConv(input_channel, 64, 3, 2, 1, bn=False))
        input_channel = 64
        cfd = [128, 256, 512]
        for out_channel in cfd:
            layers.append(BasicConv(input_channel, out_channel, 4, 2, 1))
            input_channel = out_channel
        layers.append(BasicConv(input_channel, input_channel, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Conv2d(input_channel, 1, kernel_size=4, stride=1, padding=0))
        layers.append(nn.Sigmoid())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return x


if __name__ == '__main__':
    import torch
    img = torch.randn((5, 3, 128, 128))
    print(img.shape)
    D = Discriminator(3)
    out = D(img)
    print(out.shape)

