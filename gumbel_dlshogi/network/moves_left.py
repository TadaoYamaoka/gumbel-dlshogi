"""The moves-left head: the number of plies until the end of the self-play game.

A network with the head is a subclass of the network without it, whose forward
returns `(policy, value, moves_left)` instead of `(policy, value)`. The number of the
outputs is fixed by the class, so that TorchScript can compile it.

The parameters of the head have names starting with MOVES_LEFT_PREFIX, so that a model
file without the head (e.g. of dlshogi) can be loaded with the head initialized.
"""

import functools

import torch.nn as nn
import torch.nn.functional as F

# The head predicts the number of plies divided by this scale
MOVES_LEFT_SCALE = 100.0
# Prefix of the names of the parameters of the head
MOVES_LEFT_PREFIX = "moves_left_"
# Planes and hidden units of the head
_PLANES = 8
_HIDDEN = 64


class MovesLeftHead(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.conv = nn.Conv2d(channels, _PLANES, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(_PLANES)
        self.act = activation
        self.fc1 = nn.Linear(_PLANES * 9 * 9, _HIDDEN)
        self.fc2 = nn.Linear(_HIDDEN, 1)

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        x = self.act(self.fc1(x.flatten(1)))
        # The number of plies is not negative
        return F.softplus(self.fc2(x))


class MovesLeftMixin:
    """Adds the moves-left head to a network with features() and policy_value()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moves_left_head = MovesLeftHead(self.norm1.num_features, self.act)

    def forward(self, x1, x2):
        x = self.features(x1, x2)
        policy, value = self.policy_value(x)
        return policy, value, self.moves_left_head(x)


@functools.cache
def with_moves_left(network_class):
    """Returns the subclass of the network class with the moves-left head."""
    if not (
        hasattr(network_class, "features") and hasattr(network_class, "policy_value")
    ):
        raise ValueError(
            f"{network_class.__name__} does not support the moves-left head "
            "(it needs features() and policy_value())"
        )
    # Cached, so that the same class gives the same subclass
    return type(
        network_class.__name__ + "MovesLeft", (MovesLeftMixin, network_class), {}
    )


def has_moves_left(model):
    """Whether the network (not compiled nor wrapped) has the moves-left head."""
    return isinstance(model, MovesLeftMixin)
