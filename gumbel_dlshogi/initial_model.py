import argparse
from pathlib import Path

import torch

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.features import features2_num, set_nyugyoku_features
from gumbel_dlshogi.network import PolicyValueNetwork, save_torchscript_model


def save_state_dict(model, output_path):
    """Save model state dict"""
    save_atomically(output_path, lambda path: torch.save(model.state_dict(), path))
    print(f"Model state dict saved to {output_path}")


def check_model_state(
    state_dict_path, *, blocks, channels, fcl, use_nyugyoku_features
):
    """Raises ValueError if the saved state dict does not match the model settings.

    The shapes of the parameters determine the number of blocks, channels, fcl and
    input planes, so compare them with a model built from the settings. The model is
    built on the meta device, which only has shapes (no memory or random numbers).
    """
    set_nyugyoku_features(use_nyugyoku_features)
    with torch.device("meta"):
        expected = PolicyValueNetwork(
            blocks=blocks, channels=channels, fcl=fcl, input2_channels=features2_num()
        ).state_dict()
    actual = torch.load(state_dict_path, map_location="cpu")

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
            f"{state_dict_path} does not match blocks={blocks}, channels={channels}, "
            f"fcl={fcl}, use_nyugyoku_features={use_nyugyoku_features} "
            f"(mismatched parameters: {details})"
        )


def create_initial_model(
    output_dir,
    model_name,
    *,
    blocks,
    channels,
    fcl,
    use_nyugyoku_features,
    save_torchscript=True,
    save_state_dict_file=True,
):
    """Create an initial model and save it.

    Returns:
        `(torchscript_path, state_dict_path)`. A path is None if it was not saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_nyugyoku_features(use_nyugyoku_features)
    print(
        f"Creating initial model with {blocks} blocks, {channels} channels, {fcl} fcl"
    )
    model = PolicyValueNetwork(
        blocks=blocks, channels=channels, fcl=fcl, input2_channels=features2_num()
    )

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

    return torchscript_path, state_dict_path


def main():
    parser = argparse.ArgumentParser(
        description="Create and save initial PolicyValueNetwork model"
    )

    # Model arguments
    parser.add_argument(
        "--blocks", type=int, default=10, help="Number of residual blocks"
    )
    parser.add_argument("--channels", type=int, default=192, help="Number of channels")
    parser.add_argument(
        "--fcl", type=int, default=256, help="Fully connected layer size"
    )
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features in the input",
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

    create_initial_model(
        args.output_dir,
        args.model_name,
        blocks=args.blocks,
        channels=args.channels,
        fcl=args.fcl,
        use_nyugyoku_features=args.use_nyugyoku_features,
        save_torchscript=args.save_torchscript,
        save_state_dict_file=args.save_state_dict,
    )
    print("Model creation and saving completed!")


if __name__ == "__main__":
    main()
