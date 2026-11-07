"""Saves and loads the parameters of the networks as npz files.

The format is the same as dlshogi (DeepLearningShogi) serializers: the state dict
of the network saved by numpy. The networks resnet<N>... and senet<N>... have the
same parameters as those of dlshogi, so the npz files can be used in both. The npz
file does not have the network type, so it is given by the user.

Usage (convert a TorchScript model of this package to an npz file):
    python -m gumbel_dlshogi.serializers model.pt model.npz
"""

import argparse

import numpy as np
import torch

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.network import load_torchscript_state_dict

# Parameter names of the npz files saved by Chainer (older versions of dlshogi)
_CHAINER_NAMES = {
    "W": "weight",
    "b": "bias",
    "avg_mean": "running_mean",
    "avg_var": "running_var",
    "gamma": "weight",
    "beta": "bias",
    "N": "num_batches_tracked",
}


def load_npz(path):
    """Returns the state dict saved in the npz file.

    The parameter names saved by Chainer (e.g. "l1/W") are converted to those of
    PyTorch (e.g. "l1.weight").
    """
    state_dict = {}
    with np.load(path) as f:
        for key in f.keys():
            names = key.split("/")
            if len(names) > 1:
                names[-1] = _CHAINER_NAMES.get(names[-1], names[-1])
            state_dict[".".join(names)] = torch.from_numpy(f[key])
    return state_dict


def save_npz(path, state_dict):
    """Saves the state dict to the npz file."""
    arrays = {key: value.detach().cpu().numpy() for key, value in state_dict.items()}

    def save(tmp):
        # Write to a file object, since numpy adds ".npz" to a path without it
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)

    save_atomically(path, save)
    print(f"npz model saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TorchScript model (.pt) to an npz file compatible with "
        "dlshogi"
    )
    parser.add_argument("model", help="TorchScript model (.pt) saved by this package")
    parser.add_argument("npz", help="Output npz file")
    args = parser.parse_args()

    save_npz(args.npz, load_torchscript_state_dict(args.model))


if __name__ == "__main__":
    main()
