import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.data_loader import create_dataloader, create_test_dataloader
from gumbel_dlshogi.features import features2_num, set_nyugyoku_features
from gumbel_dlshogi.network import (
    add_compile_arguments,
    add_network_argument,
    compile_model,
    policy_value_network,
    resolve_compile_options,
    save_torchscript_model,
)


def train_epoch(model, dataloader, optimizer, scaler, device, amp_enabled=True):
    """Train for one epoch"""
    model.train()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    num_batches = 0

    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for features1, features2, policy_targets, value_targets in pbar:
            features1 = features1.to(device)
            features2 = features2.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()

            with autocast(device.type, enabled=amp_enabled):
                policy_output, value_output = model(features1, features2)

                # Policy loss - targets are probability distributions
                policy_loss = policy_criterion(policy_output, policy_targets)

                # Value loss - targets are game results
                value_loss = value_criterion(value_output.squeeze(-1), value_targets)

                total_batch_loss = policy_loss + value_loss

            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{total_batch_loss.item():.4f}",
                    "Policy": f"{policy_loss.item():.4f}",
                    "Value": f"{value_loss.item():.4f}",
                }
            )

    return {
        "total_loss": total_loss / num_batches,
        "policy_loss": policy_loss_sum / num_batches,
        "value_loss": value_loss_sum / num_batches,
    }


def evaluate(model, dataloader, device, amp_enabled=True):
    """Evaluate the model"""
    model.eval()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    policy_correct = 0
    value_correct = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating", unit="batch") as pbar:
            for features1, features2, policy_targets, value_targets in pbar:
                features1 = features1.to(device)
                features2 = features2.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)

                with autocast(device.type, enabled=amp_enabled):
                    policy_output, value_output = model(features1, features2)

                    # For test data, policy_targets are labels (long), not distributions
                    policy_loss = policy_criterion(policy_output, policy_targets)
                    # Calculate accuracy for classification
                    policy_pred = torch.argmax(policy_output, dim=1)
                    policy_correct += (policy_pred == policy_targets).sum().item()

                    value_loss = value_criterion(
                        value_output.squeeze(-1), value_targets
                    )

                    # Value accuracy (threshold at 0.5)
                    value_pred = torch.sigmoid(value_output.squeeze(-1))
                    value_binary_pred = (value_pred > 0.5).float()
                    value_binary_target = (value_targets > 0.5).float()
                    value_correct += (
                        (value_binary_pred == value_binary_target).sum().item()
                    )

                    total_batch_loss = policy_loss + value_loss

                total_loss += total_batch_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                total_samples += features1.size(0)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": f"{total_batch_loss.item():.4f}",
                        "Policy Acc": f"{policy_correct/total_samples:.4f}",
                        "Value Acc": f"{value_correct/total_samples:.4f}",
                    }
                )

    num_batches = len(dataloader)

    return {
        "total_loss": total_loss / num_batches,
        "policy_loss": policy_loss_sum / num_batches,
        "value_loss": value_loss_sum / num_batches,
        "policy_accuracy": policy_correct / total_samples,
        "value_accuracy": value_correct / total_samples,
    }


def save_checkpoint(model, optimizer, scaler, epoch, total_steps, loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "total_steps": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    save_atomically(filepath, lambda path: torch.save(checkpoint, path))
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer, scaler, device):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    epoch = checkpoint["epoch"]
    total_steps = checkpoint["total_steps"]
    loss = checkpoint["loss"]
    print(
        f"Checkpoint loaded from {filepath}, epoch {epoch}, steps {total_steps}, loss {loss:.4f}"
    )
    return epoch, total_steps


def load_initial_model(filepath, model, device):
    """Load initial model state dict"""
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Initial model loaded from {filepath}")


def train(
    train_dir,
    test_file,
    num_files,
    network,
    initial_model,
    epochs,
    batch_size,
    eval_batch_size,
    lr,
    weight_decay,
    num_workers,
    amp,
    checkpoint_dir,
    log_dir,
    resume,
    save_interval,
    eval_interval,
    save_torchscript,
    use_nyugyoku_features=False,
    compile_options=None,
):
    """Main training function

    Args:
        compile_options: If not None, the model is compiled with torch.compile by
            compile_model(**compile_options), e.g. {"backend": None, "mode": None}

    Returns:
        The path of the checkpoint saved at the last epoch (None if no epoch ran).
    """
    set_nyugyoku_features(use_nyugyoku_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    # Create SummaryWriter if log_dir is specified
    writer = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir_path)

    # Create model
    model = policy_value_network(network, input2_channels=features2_num()).to(device)

    # Load initial model if specified
    if initial_model:
        load_initial_model(initial_model, model, device)

    # Train and evaluate with the compiled model. It shares the parameters with model,
    # which is saved (the state dict of the compiled model has other names).
    run_model = model
    if compile_options is not None:
        run_model = compile_model(model, device, **compile_options)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create gradient scaler for mixed precision
    scaler = GradScaler(device.type, enabled=amp)

    # Create data loaders
    train_dataloader = create_dataloader(
        train_dir,
        batch_size=batch_size,
        num_files=num_files,
        num_workers=num_workers,
        shuffle=True,
        use_nyugyoku_features=use_nyugyoku_features,
    )

    test_dataloader = None
    if test_file:
        test_dataloader = create_test_dataloader(
            test_file,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
            use_nyugyoku_features=use_nyugyoku_features,
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    total_steps = 0
    if resume:
        start_epoch, total_steps = load_checkpoint(
            resume, model, optimizer, scaler, device
        )
        start_epoch += 1

    max_epochs = start_epoch + epochs

    # Training loop
    checkpoint_path = None
    for epoch in range(start_epoch, max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")

        # Training
        start_time = time.time()
        train_metrics = train_epoch(
            run_model, train_dataloader, optimizer, scaler, device, amp
        )
        train_time = time.time() - start_time

        # Update total steps
        total_steps += len(train_dataloader)

        print(
            f"Train - Loss: {train_metrics['total_loss']:.4f}, "
            f"Policy Loss: {train_metrics['policy_loss']:.4f}, "
            f"Value Loss: {train_metrics['value_loss']:.4f}, "
            f"Time: {train_time:.2f}s, "
            f"Total Steps: {total_steps}"
        )

        # Log training metrics to TensorBoard
        if writer:
            writer.add_scalar("epoch", epoch, total_steps)

            writer.add_scalar("train/loss", train_metrics["total_loss"], total_steps)
            writer.add_scalar(
                "train/policy_loss", train_metrics["policy_loss"], total_steps
            )
            writer.add_scalar(
                "train/value_loss", train_metrics["value_loss"], total_steps
            )

        # Evaluation
        if test_dataloader and (epoch + 1) % eval_interval == 0:
            start_time = time.time()
            eval_metrics = evaluate(run_model, test_dataloader, device, amp)
            eval_time = time.time() - start_time

            print(
                f"Eval - Loss: {eval_metrics['total_loss']:.4f}, "
                f"Policy Loss: {eval_metrics['policy_loss']:.4f}, "
                f"Value Loss: {eval_metrics['value_loss']:.4f}, "
                f"Policy Acc: {eval_metrics['policy_accuracy']:.4f}, "
                f"Value Acc: {eval_metrics['value_accuracy']:.4f}, "
                f"Time: {eval_time:.2f}s"
            )

            # Log evaluation metrics to TensorBoard
            if writer:
                writer.add_scalar("val/loss", eval_metrics["total_loss"], total_steps)
                writer.add_scalar(
                    "val/policy_loss", eval_metrics["policy_loss"], total_steps
                )
                writer.add_scalar(
                    "val/value_loss", eval_metrics["value_loss"], total_steps
                )
                writer.add_scalar(
                    "val/policy_accuracy", eval_metrics["policy_accuracy"], total_steps
                )
                writer.add_scalar(
                    "val/value_accuracy", eval_metrics["value_accuracy"], total_steps
                )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == max_epochs - 1:
            checkpoint_path = (
                checkpoint_dir_path / f"checkpoint_epoch_{epoch + 1:03d}.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                total_steps,
                train_metrics["total_loss"],
                checkpoint_path,
            )

    # Save TorchScript model if requested
    if save_torchscript:
        save_torchscript_model(model, save_torchscript)

    if writer:
        writer.close()

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train a policy value network")

    # Data arguments
    parser.add_argument(
        "train_dir",
        type=str,
        help="Directory containing training data files",
    )
    parser.add_argument("--test_file", type=str, help="Test data file path")
    parser.add_argument(
        "--num_files", type=int, help="Number of newest training files to use"
    )

    # Model arguments
    add_network_argument(parser)
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features in the input",
    )
    parser.add_argument(
        "--initial_model", type=str, help="Path to initial model state dict (.pth file)"
    )
    # Training arguments
    parser.add_argument(
        "--epochs", "-e", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--eval_batch_size", type=int, default=1024, help="Evaluation batch size"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loader workers"
    )

    # Mixed precision
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision"
    )

    # torch.compile
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model with torch.compile"
    )
    add_compile_arguments(parser)

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Directory to save TensorBoard logs",
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate every N epochs"
    )

    # TorchScript output
    parser.add_argument(
        "--save_torchscript",
        help="Save final model as TorchScript (.pt file)",
    )

    args = parser.parse_args()
    compile_options = None
    if args.compile:
        compile_options = {"backend": args.compile_backend, "mode": args.compile_mode}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolve_compile_options(device, **compile_options)
        except ValueError as e:
            parser.error(str(e))
    train(
        train_dir=args.train_dir,
        test_file=args.test_file,
        num_files=args.num_files,
        network=args.network,
        initial_model=args.initial_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=args.amp,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        save_torchscript=args.save_torchscript,
        use_nyugyoku_features=args.use_nyugyoku_features,
        compile_options=compile_options,
    )


if __name__ == "__main__":
    main()
