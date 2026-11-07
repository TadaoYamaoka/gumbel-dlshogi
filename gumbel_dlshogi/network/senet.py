import functools

import torch.nn as nn

from gumbel_dlshogi.network import resnet


class SELayer(nn.Module):
    def __init__(self, channels, activation, reduction):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.act = activation
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.act(self.fc1(scale))
        scale = self.fc2(scale).sigmoid()
        return x * scale


class SEResNetBlock(nn.Module):
    def __init__(self, channels, activation, reduction):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation
        self.se = SELayer(channels, activation, reduction)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return self.act(self.se(out) + x)


class PolicyValueNetwork(resnet.PolicyValueNetwork):
    """The ResNet with Squeeze-and-Excitation blocks"""

    def __init__(
        self,
        blocks=10,
        channels=192,
        activation=None,
        fcl=256,
        reduction=8,
        *,
        input2_channels,
    ):
        super().__init__(
            blocks,
            channels,
            activation,
            fcl,
            input2_channels=input2_channels,
            block=functools.partial(SEResNetBlock, reduction=reduction),
        )
