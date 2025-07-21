import argparse
from pathlib import Path

import torch
from pydlshogi2.network.policy_value_resnet import PolicyValueNetwork


def create_initial_model(blocks=10, channels=192, fcl=256):
    """Create and return an initial PolicyValueNetwork model"""
    model = PolicyValueNetwork(blocks=blocks, channels=channels, fcl=fcl)
    return model


def save_torchscript_model(model, output_path):
    """Save model as TorchScript"""
    model.eval()

    class PolicyValueNetworkAddSigmoid(torch.nn.Module):
        def __init__(self, model):
            super(PolicyValueNetworkAddSigmoid, self).__init__()
            self.base_model = model

        def forward(self, x):
            y1, y2 = self.base_model(x)
            return y1, torch.sigmoid(y2)

    # Script the model
    scripted_model = torch.jit.script(PolicyValueNetworkAddSigmoid(model))

    # Save the scripted model
    scripted_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")


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
        action="store_true",
        default=True,
        help="Save model as TorchScript",
    )
    parser.add_argument(
        "--save_state_dict",
        action="store_true",
        default=True,
        help="Save model state dict",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create initial model
    print(
        f"Creating initial model with {args.blocks} blocks, {args.channels} channels, {args.fcl} fcl"
    )
    model = create_initial_model(
        blocks=args.blocks, channels=args.channels, fcl=args.fcl
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
