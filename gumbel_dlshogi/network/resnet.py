import torch
import torch.nn as nn

from gumbel_dlshogi.features import FEATURES1_NUM, MOVE_LABELS_NUM, MOVE_PLANES_NUM


class Bias(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class ResNetBlock(nn.Module):
    def __init__(self, channels, activation):
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return self.act(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        blocks=10,
        channels=192,
        activation=None,
        fcl=256,
        *,
        input2_channels,
        block=ResNetBlock,
    ):
        """
        Args:
            activation: Activation module shared by the layers (nn.ReLU() if None)
            input2_channels: Number of planes of the input x2 (features2_num())
            block: Called as block(channels, activation) to create each block
        """
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        # input layer (the parameter names are the same as dlshogi)
        # x1: 駒の配置・利き・利き数、x2: 持ち駒・王手・入玉
        self.l1_1_1 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.l1_1_2 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.l1_2 = nn.Conv2d(
            in_channels=input2_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        # resnet blocks
        self.blocks = nn.Sequential(
            *[block(channels, activation) for _ in range(blocks)]
        )

        # policy head
        self.policy = nn.Conv2d(
            in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False
        )
        # ラベルごとに独立したバイアス(Conv2d の bias はチャネルごとなので使わない)
        self.policy_bias = Bias(MOVE_LABELS_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False
        )
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        return self.policy_value(self.features(x1, x2))

    def features(self, x1, x2):
        """Returns the output of the shared layers, used by the heads."""
        x = self.l1_1_1(x1) + self.l1_1_2(x1) + self.l1_2(x2)
        x = self.act(self.norm1(x))

        # resnet blocks
        return self.blocks(x)

    def policy_value(self, x):
        """Returns the policy logits and the value logit from the features."""
        # policy head
        policy = self.policy_bias(self.policy(x).flatten(1))

        # value head
        value = self.act(self.value_norm1(self.value_conv1(x)))
        value = self.act(self.value_fc1(value.flatten(1)))
        value = self.value_fc2(value)

        return policy, value
