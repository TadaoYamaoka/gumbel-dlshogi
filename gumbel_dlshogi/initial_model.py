import argparse
from pathlib import Path

import torch

from gumbel_dlshogi.features import features2_num, set_nyugyoku_features
from gumbel_dlshogi.network import PolicyValueNetwork, save_torchscript_model


def save_state_dict(model, output_path):
    """Save model state dict"""
    torch.save(model.state_dict(), output_path)
    print(f"Model state dict saved to {output_path}")


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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create initial model
    set_nyugyoku_features(args.use_nyugyoku_features)
    print(
        f"Creating initial model with {args.blocks} blocks, {args.channels} channels, {args.fcl} fcl"
    )
    model = PolicyValueNetwork(
        blocks=args.blocks,
        channels=args.channels,
        fcl=args.fcl,
        input2_channels=features2_num(),
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Save TorchScript model
    if args.save_torchscript:
        torchscript_path = output_dir / f"{args.model_name}.pt"
        save_torchscript_model(model, torchscript_path)

    # Save state dict
    if args.save_state_dict:
        state_dict_path = output_dir / f"{args.model_name}.pth"
        save_state_dict(model, state_dict_path)

    print("Model creation and saving completed!")


if __name__ == "__main__":
    main()
