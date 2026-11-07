import torch
import torch.nn as nn
import torch.nn.functional as F

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.features import FEATURES1_NUM, MOVE_LABELS_NUM, MOVE_PLANES_NUM


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, channels=192, fcl=256, *, input2_channels):
        """
        Args:
            input2_channels: Number of planes of the input x2 (features2_num())
        """
        super().__init__()
        # input layer
        # x1: 駒の配置・利き・利き数、x2: 持ち駒・王手・入玉
        self.l1_1 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.l1_2 = nn.Conv2d(
            in_channels=input2_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(
            in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False
        )
        # ラベルごとに独立したバイアス(Conv2d の bias はチャネルごとなので使わない)
        self.policy_bias = nn.Parameter(torch.zeros(MOVE_LABELS_NUM))

        # value head
        self.value_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=MOVE_PLANES_NUM, kernel_size=1, bias=False
        )
        self.value_norm1 = nn.BatchNorm2d(MOVE_PLANES_NUM)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        x = self.l1_1(x1) + self.l1_2(x2)
        x = F.relu(self.norm1(x))

        # resnet blocks
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x).flatten(1) + self.policy_bias

        # value head
        value = F.relu(self.value_norm1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(value.flatten(1)))
        value = self.value_fc2(value)

        return policy, value


class PolicyValueNetworkAddSigmoid(nn.Module):
    """推論用に value を sigmoid で勝率に変換するラッパー(学習時は logits のまま使う)"""

    def __init__(self, model):
        super().__init__()
        self.base_model = model

    def forward(self, x1, x2):
        y1, y2 = self.base_model(x1, x2)
        return y1, torch.sigmoid(y2)


def save_torchscript_model(model, filepath):
    """Save model as TorchScript"""
    model.eval()
    scripted_model = torch.jit.script(PolicyValueNetworkAddSigmoid(model))
    save_atomically(filepath, lambda path: scripted_model.save(str(path)))
    print(f"TorchScript model saved to {filepath}")
