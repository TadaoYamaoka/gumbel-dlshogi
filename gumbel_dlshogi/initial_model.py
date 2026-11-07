import argparse
import json
from pathlib import Path

import torch

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.features import features2_num, set_nyugyoku_features
from gumbel_dlshogi.network import (
    add_network_argument,
    load_torchscript_state_dict,
    policy_value_network,
    save_torchscript_model,
)
from gumbel_dlshogi.serializers import load_npz


def save_state_dict(model, output_path):
    """Save model state dict"""
    save_atomically(output_path, lambda path: torch.save(model.state_dict(), path))
    print(f"Model state dict saved to {output_path}")


def settings_path(model_path):
    """Path of the model settings saved with the initial model files"""
    return Path(model_path).with_suffix(".json")


def save_model_settings(path, model_settings, files):
    """Saves the model settings with the names of the files saved with them.

    It is written after the files, so the listed files are complete.
    """
    record = {"model_settings": model_settings, "files": sorted(files)}
    text = json.dumps(record, indent=2)
    save_atomically(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _read_settings_record(model_path):
    """Returns the saved settings record for the model file, or None if not found.

    The record has "model_settings" and "files". It is written by
    save_model_settings() or as the completion record of a cycle in run.py. A JSON
    file without them (e.g. written by an older version) is not a record.
    """
    path = settings_path(model_path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not {"model_settings", "files"} <= record.keys():
        return None
    return record


def read_model_settings(model_path):
    """Returns the settings that the model file was saved with.

    Returns None if they are unknown: the settings are not found, or they were saved
    with other files (e.g. the file was left from an earlier save).
    """
    record = _read_settings_record(model_path)
    if record is None or Path(model_path).name not in record["files"]:
        return None
    return record["model_settings"]


def check_model_state(state_dict_path, *, network, use_nyugyoku_features):
    """Raises ValueError if the saved state dict does not match the model settings."""
    _check_model_file(
        state_dict_path,
        torch.load(state_dict_path, map_location="cpu"),
        network=network,
        use_nyugyoku_features=use_nyugyoku_features,
    )


def check_torchscript_model(torchscript_path, *, network, use_nyugyoku_features):
    """Raises ValueError if the TorchScript model does not match the model settings."""
    _check_model_file(
        torchscript_path,
        load_torchscript_state_dict(torchscript_path),
        network=network,
        use_nyugyoku_features=use_nyugyoku_features,
    )


def _check_model_file(model_path, actual, *, network, use_nyugyoku_features):
    """Raises ValueError if the model file does not match the model settings.

    The settings saved with the file must be the same. The parameters do not
    determine some settings (e.g. relu and swish have the same parameters). If the
    settings were saved with other files only, the file was left from an earlier save
    and its settings are unknown, so it is an error.

    The names and shapes of the parameters determine the rest of the network and the
    input planes, so compare them with a model built from the settings. This also
    checks a file saved without the settings (e.g. not by create_initial_model). The
    model is built on the meta device, which only has shapes (no memory or random
    numbers).
    """
    model_settings = {
        "network": network,
        "use_nyugyoku_features": use_nyugyoku_features,
    }
    record = _read_settings_record(model_path)
    if record is None:
        print(
            f"Warning: The settings of {model_path} are not found in "
            f"{settings_path(model_path)}, so only its parameters are checked "
            "(not e.g. the activation)."
        )
    elif Path(model_path).name not in record["files"]:
        raise ValueError(
            f"{model_path} is not one of the files saved with the settings "
            f"{settings_path(model_path)}, so it may be left from an earlier save"
        )
    elif record["model_settings"] != model_settings:
        raise ValueError(f"{model_path} was created with {record['model_settings']}")

    set_nyugyoku_features(use_nyugyoku_features)
    with torch.device("meta"):
        expected = policy_value_network(
            network, input2_channels=features2_num()
        ).state_dict()
    _check_parameters(
        model_path,
        expected,
        actual,
        network=network,
        use_nyugyoku_features=use_nyugyoku_features,
    )


def _check_parameters(
    model_path, expected, actual, *, network, use_nyugyoku_features
):
    """Raises ValueError if the names or shapes of the parameters differ."""
    mismatches = sorted(
        key
        for key in expected.keys() | actual.keys()
        if key not in expected
        or key not in actual
        or expected[key].shape != actual[key].shape
    )
    if mismatches:
        details = ", ".join(mismatches[:5])
        if len(mismatches) > 5:
            details += ", ..."
        raise ValueError(
            f"{model_path} does not match network={network}, "
            f"use_nyugyoku_features={use_nyugyoku_features} "
            f"(mismatched parameters: {details})"
        )


def create_initial_model(
    output_dir,
    model_name,
    *,
    network,
    use_nyugyoku_features,
    save_torchscript=True,
    save_state_dict_file=True,
    npz=None,
):
    """Create an initial model and save it.

    The model settings are saved to <model_name>.json after the model files, with
    the names of the saved files, so that the checks can check what the parameters
    do not determine.

    Args:
        npz: Model file (.npz, e.g. of dlshogi) to initialize the model with. It
            must be the network given by `network` (the activation is not checked).

    Returns:
        `(torchscript_path, state_dict_path)`. A path is None if it was not saved.

    Raises:
        ValueError: If npz does not match the network.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_nyugyoku_features(use_nyugyoku_features)
    print(f"Creating initial model {network}")
    model = policy_value_network(network, input2_channels=features2_num())
    if npz:
        state_dict = load_npz(npz)
        _check_parameters(
            npz,
            model.state_dict(),
            state_dict,
            network=network,
            use_nyugyoku_features=use_nyugyoku_features,
        )
        model.load_state_dict(state_dict)
        print(f"Initialized with the npz model {npz}")

    # Remove the old settings before overwriting the files, so that they do not
    # describe a file left from the earlier save or if the save stops on the way
    model_settings_path = settings_path(output_dir / model_name)
    model_settings_path.unlink(missing_ok=True)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Save TorchScript model
    torchscript_path = None
    if save_torchscript:
        torchscript_path = output_dir / f"{model_name}.pt"
        save_torchscript_model(model, torchscript_path)

    # Save state dict
    state_dict_path = None
    if save_state_dict_file:
        state_dict_path = output_dir / f"{model_name}.pth"
        save_state_dict(model, state_dict_path)

    # Save the model settings for the files saved above
    saved_files = [path.name for path in (torchscript_path, state_dict_path) if path]
    if saved_files:
        save_model_settings(
            model_settings_path,
            {"network": network, "use_nyugyoku_features": use_nyugyoku_features},
            saved_files,
        )

    return torchscript_path, state_dict_path


def main():
    parser = argparse.ArgumentParser(
        description="Create and save an initial policy value network"
    )

    # Model arguments
    add_network_argument(parser)
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features in the input",
    )
    parser.add_argument(
        "--npz",
        type=str,
        help="Initialize with a model file (.npz, e.g. of dlshogi) of the network "
        "given by --network",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="initial_model",
        help="Base name for the model files",
    )
    parser.add_argument(
        "--save_torchscript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model as TorchScript",
    )
    parser.add_argument(
        "--save_state_dict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model state dict",
    )

    args = parser.parse_args()

    try:
        create_initial_model(
            args.output_dir,
            args.model_name,
            network=args.network,
            use_nyugyoku_features=args.use_nyugyoku_features,
            save_torchscript=args.save_torchscript,
            save_state_dict_file=args.save_state_dict,
            npz=args.npz,
        )
    except ValueError as e:
        parser.error(str(e))
    print("Model creation and saving completed!")


if __name__ == "__main__":
    main()
