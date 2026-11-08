"""Policy value networks selected by a name, in the same way as dlshogi.

The name has the form <type><blocks>[x<channels>][_fcl<fcl>][_reduction<reduction>]
[_<activation>], for example "resnet10", "resnet20x256_fcl384_swish" or
"senet10_reduction16". <type> is the module of this package that defines
PolicyValueNetwork. The channels can be omitted for 10, 15, 20 and 30 blocks, and
the activation is relu or swish (relu by default). A user defined network is given
as "<module>.<class>", which is called with input2_channels only.
"""

import argparse
import inspect
import re
from importlib import import_module
from importlib.util import find_spec

import torch
import torch.nn as nn

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.network.moves_left import (
    MOVES_LEFT_SCALE,
    has_moves_left,
    with_moves_left,
)

_NETWORK_PATTERN = re.compile(
    r"(?P<type>[a-z]\w*?)(?P<blocks>\d+)(?:x(?P<channels>\d+))?(?:_fcl(?P<fcl>\d+))?"
    r"(?:_reduction(?P<reduction>\d+))?(?:_(?P<activation>[a-z]+))?"
)
_DEFAULT_CHANNELS = {10: 192, 15: 224, 20: 256, 30: 384}
_ACTIVATIONS = {"relu": nn.ReLU, "swish": nn.SiLU}


def _resolve_network(network):
    """Returns the class of the network and the arguments given by its name.

    Raises:
        ValueError: If the name does not give a network.
    """
    if "." in network:
        module_name, _, class_name = network.rpartition(".")
        try:
            return getattr(import_module(module_name), class_name), {}
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load the network {network!r}: {e}") from e

    m = _NETWORK_PATTERN.fullmatch(network)
    if m is None:
        raise ValueError(
            f"Invalid network {network!r}. Give it like resnet10x192_fcl256_relu, "
            "or as <module>.<class> for a user defined network."
        )
    module_name = f"{__name__}.{m['type']}"
    try:
        network_class = import_module(module_name).PolicyValueNetwork
    except ModuleNotFoundError as e:
        if e.name != module_name:
            raise
        raise ValueError(f"Unknown network type {m['type']!r} in {network!r}") from e

    blocks = int(m["blocks"])
    if m["channels"]:
        channels = int(m["channels"])
    elif blocks in _DEFAULT_CHANNELS:
        channels = _DEFAULT_CHANNELS[blocks]
    else:
        raise ValueError(
            f"Give the channels of {blocks} blocks like {m['type']}{blocks}x256"
        )
    kwargs = {"blocks": blocks, "channels": channels}
    if m["fcl"]:
        kwargs["fcl"] = int(m["fcl"])
    if m["reduction"]:
        if "reduction" not in inspect.signature(network_class).parameters:
            raise ValueError(f"The network type {m['type']!r} has no reduction")
        kwargs["reduction"] = int(m["reduction"])
    if m["activation"]:
        if m["activation"] not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {m['activation']!r} in {network!r} "
                f"(choose from {', '.join(_ACTIVATIONS)})"
            )
        kwargs["activation"] = _ACTIVATIONS[m["activation"]]
    return network_class, kwargs


def policy_value_network(network, *, input2_channels, moves_left=False):
    """Creates the network given by its name.

    Args:
        network: Name of the network (see the module docstring)
        input2_channels: Number of planes of the input x2 (features2_num())
        moves_left: Whether the network has the moves-left head, which makes it
            return `(policy, value, moves_left)` (see network.moves_left)

    Raises:
        ValueError: If the name does not give a network, or the network does not
            support the moves-left head.
    """
    network_class, kwargs = _resolve_network(network)
    if moves_left:
        network_class = with_moves_left(network_class)
    if "activation" in kwargs:
        kwargs["activation"] = kwargs["activation"]()
    return network_class(**kwargs, input2_channels=input2_channels)


def add_moves_left_argument(parser):
    """Adds the --moves_left argument (whether the network has the head)."""
    parser.add_argument(
        "--moves_left",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the moves-left head to the network, which predicts the number of "
        "plies until the end of the self-play game",
    )


def _network_type(network):
    """argparse type that checks the name of the network."""
    try:
        _resolve_network(network)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    return network


def add_network_argument(parser, default="resnet10_relu", help=None):
    """Adds the --network argument, which is checked when it is parsed."""
    parser.add_argument(
        "--network",
        type=_network_type,
        default=default,
        help=help
        or "Network type, e.g. resnet10_relu, resnet20x256_fcl384_swish, senet10",
    )


class PolicyValueNetworkAddSigmoid(nn.Module):
    """推論用に value を sigmoid で勝率に変換するラッパー(学習時は logits のまま使う)"""

    def __init__(self, model):
        super().__init__()
        self.base_model = model

    def forward(self, x1, x2):
        y1, y2 = self.base_model(x1, x2)
        return y1, torch.sigmoid(y2)


class PolicyValueMovesLeftNetworkAddSigmoid(nn.Module):
    """The wrapper for inference of a network with the moves-left head.

    It returns the policy logits, the value as a win rate and the moves left as a
    number of plies.
    """

    def __init__(self, model):
        super().__init__()
        self.base_model = model
        # TorchScript does not use a global constant, so keep it as an attribute
        self.moves_left_scale = MOVES_LEFT_SCALE

    def forward(self, x1, x2):
        y1, y2, y3 = self.base_model(x1, x2)
        return y1, torch.sigmoid(y2), y3 * self.moves_left_scale


def inference_wrapper(model):
    """Wraps the network for inference (2 or 3 outputs by the moves-left head)."""
    if has_moves_left(model):
        return PolicyValueMovesLeftNetworkAddSigmoid(model)
    return PolicyValueNetworkAddSigmoid(model)


def save_torchscript_model(model, filepath):
    """Save model as TorchScript"""
    model.eval()
    scripted_model = torch.jit.script(inference_wrapper(model))
    save_atomically(filepath, lambda path: scripted_model.save(str(path)))
    print(f"TorchScript model saved to {filepath}")


def load_torchscript_state_dict(filepath):
    """Returns the state dict of the network saved by save_torchscript_model."""
    state_dict = torch.jit.load(str(filepath), map_location="cpu").state_dict()
    # The network is saved in PolicyValueNetworkAddSigmoid
    prefix = "base_model."
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def add_compile_arguments(parser):
    """Adds the arguments of torch.compile (backend and mode)."""
    parser.add_argument(
        "--compile_backend",
        help="Backend of torch.compile. Defaults to inductor, or cudagraphs on CUDA "
        "if triton is not installed (e.g. on Windows).",
    )
    parser.add_argument(
        "--compile_mode",
        help="Mode of torch.compile for inductor, e.g. reduce-overhead or max-autotune",
    )


def resolve_compile_options(device, *, backend=None, mode=None):
    """Returns the arguments of torch.compile.

    inductor needs triton on CUDA, which is often not available on Windows. Then
    cudagraphs is used by default, which replays the recorded CUDA kernels and reduces
    the launch overhead. The shapes are static, so a new batch size compiles again.

    Raises:
        ValueError: If mode is given with a backend other than inductor. torch.compile
            raises it only when the model is run for the first time.
    """
    if backend is None:
        has_triton = find_spec("triton") is not None
        use_cudagraphs = device.type == "cuda" and not has_triton
        backend = "cudagraphs" if use_cudagraphs else "inductor"
    if mode and backend != "inductor":
        raise ValueError(
            f"The compile mode {mode!r} is only for the inductor backend, not "
            f"{backend!r} (inductor needs triton on CUDA)"
        )
    kwargs = {"backend": backend, "dynamic": False}
    if mode:
        kwargs["mode"] = mode
    return kwargs


def compile_model(model, device, *, backend=None, mode=None):
    """Compiles the model with torch.compile (see resolve_compile_options)."""
    kwargs = resolve_compile_options(device, backend=backend, mode=mode)
    print(
        "Using torch.compile("
        + ", ".join(f"{key}={value}" for key, value in kwargs.items())
        + ")"
    )
    return torch.compile(model, **kwargs)
