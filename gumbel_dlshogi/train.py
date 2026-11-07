import argparse
import os
import random
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
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

_NCCL_REQUIRED = "Training on 2 or more GPUs needs NCCL, which is available on Linux"


def train_epoch(
    model, dataloader, optimizer, scaler, device, amp_enabled=True, progress=True
):
    """Train for one epoch"""
    model.train()

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", unit="batch", disable=not progress)
    with pbar:
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


def _average_metrics(metrics, device):
    """Averages the metrics of the processes in data parallel training."""
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device)
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(zip(keys, values.tolist()))


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
    gpus=None,
):
    """Main training function

    Args:
        batch_size: Batch size of a step. In data parallel training, each GPU trains
            batch_size / len(gpus) samples of it.
        compile_options: If not None, the model is compiled with torch.compile by
            compile_model(**compile_options), e.g. {"backend": None, "mode": None}
        gpus: CUDA device ids to train on. With 2 or more, the model is trained in
            data parallel (DistributedDataParallel) by a process per GPU, which needs
            NCCL (Linux). If None, the default device is used.

    Returns:
        The path of the checkpoint saved at the last epoch (None if no epoch ran).
    """
    # The arguments of _train() (all but gpus)
    params = {key: value for key, value in locals().items() if key != "gpus"}

    if not gpus or len(gpus) == 1:
        if gpus:
            device = torch.device("cuda", gpus[0])
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _train(0, 1, device, None, **params)

    world_size = len(gpus)
    if not dist.is_nccl_available():
        raise ValueError(_NCCL_REQUIRED)
    if batch_size % world_size:
        raise ValueError(
            f"The batch size {batch_size} is not divisible by the number of GPUs "
            f"{world_size}"
        )
    # All processes shuffle the data with the same seed, and each takes its part
    seed = random.randrange(2**31)
    dist_init = (_find_free_port(), seed)
    result = torch.multiprocessing.get_context("spawn").SimpleQueue()
    torch.multiprocessing.spawn(
        _train_worker,
        args=(gpus, dist_init, result, params),
        nprocs=world_size,
        join=True,
    )
    return result.get()


def _find_free_port():
    """Returns a free TCP port of localhost to initialize the process group."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _train_worker(rank, gpus, dist_init, result, params):
    """A process of data parallel training on gpus[rank]"""
    port, seed = dist_init
    world_size = len(gpus)
    device = torch.device("cuda", gpus[rank])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    if rank != 0:
        # Only the first process prints the progress
        sys.stdout = open(os.devnull, "w")
    try:
        checkpoint_path = _train(rank, world_size, device, seed, **params)
        if rank == 0:
            result.put(checkpoint_path)
    finally:
        dist.destroy_process_group()


def _train(
    rank,
    world_size,
    device,
    seed,
    *,
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
    use_nyugyoku_features,
    compile_options,
):
    """Trains the model in a process (of world_size processes in data parallel).

    The first process (rank 0) evaluates the model, saves it and writes the logs.
    """
    is_main = rank == 0
    distributed = world_size > 1
    set_nyugyoku_features(use_nyugyoku_features)

    print(f"Using device: {device}" + (f" x {world_size}" if distributed else ""))

    # Create checkpoint directory
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    # Create SummaryWriter if log_dir is specified
    writer = None
    if log_dir and is_main:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir_path)

    # Create model
    model = policy_value_network(network, input2_channels=features2_num()).to(device)

    # Load initial model if specified
    if initial_model:
        load_initial_model(initial_model, model, device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create gradient scaler for mixed precision
    scaler = GradScaler(device.type, enabled=amp)

    # Resume from checkpoint if specified (all processes load the same state)
    start_epoch = 0
    total_steps = 0
    if resume:
        start_epoch, total_steps = load_checkpoint(
            resume, model, optimizer, scaler, device
        )
        start_epoch += 1

    # Train with the model wrapped by DDP, which averages the gradients of the
    # processes. model has the parameters and is saved (and evaluated by the first
    # process, since a forward of DDP communicates in all processes).
    run_model = model
    if distributed:
        run_model = DistributedDataParallel(model, device_ids=[device])
    # Train and evaluate with the compiled model. It shares the parameters with model,
    # which is saved (the state dict of the compiled model has other names).
    if compile_options is not None:
        run_model = compile_model(run_model, device, **compile_options)
    eval_model = model if distributed else run_model

    # Create data loaders
    train_dataloader = create_dataloader(
        train_dir,
        batch_size=batch_size // world_size,
        num_files=num_files,
        num_workers=num_workers,
        shuffle=True,
        use_nyugyoku_features=use_nyugyoku_features,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
    )

    test_dataloader = None
    if test_file and is_main:
        test_dataloader = create_test_dataloader(
            test_file,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
            use_nyugyoku_features=use_nyugyoku_features,
        )

    max_epochs = start_epoch + epochs

    # Training loop
    checkpoint_path = None
    for epoch in range(start_epoch, max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)

        # Training
        start_time = time.time()
        train_metrics = train_epoch(
            run_model,
            train_dataloader,
            optimizer,
            scaler,
            device,
            amp,
            progress=is_main,
        )
        if distributed:
            train_metrics = _average_metrics(train_metrics, device)
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
            eval_metrics = evaluate(eval_model, test_dataloader, device, amp)
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
            if is_main:
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
    if save_torchscript and is_main:
        save_torchscript_model(model, save_torchscript)

    if writer:
        writer.close()

    return checkpoint_path


def gpu_ids(value):
    """argparse type of a comma separated list of CUDA device ids, e.g. "0,1" """
    try:
        ids = [int(id) for id in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid GPU ids {value!r}, e.g. 0,1")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"GPU ids {value!r} have duplicates")
    if len(ids) > 1 and not dist.is_nccl_available():
        raise argparse.ArgumentTypeError(_NCCL_REQUIRED)
    count = torch.cuda.device_count()
    invalid = [id for id in ids if not 0 <= id < count]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"GPU ids {invalid} are not available ({count} GPUs are found)"
        )
    return ids


def add_gpus_argument(parser, name="--gpus"):
    """Adds the argument of the GPUs to train on."""
    parser.add_argument(
        name,
        type=gpu_ids,
        help="Comma separated CUDA device ids to train on, e.g. 0,1. With 2 or more, "
        "the model is trained in data parallel (the batch size is divided by them), "
        "which needs NCCL (Linux).",
    )


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
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loader workers (of each GPU in data parallel)",
    )
    add_gpus_argument(parser)

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
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
