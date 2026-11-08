import argparse
import collections
import json
import re
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from gumbel_dlshogi.common import save_atomically
from gumbel_dlshogi.initial_model import (
    check_model_state,
    check_torchscript_model,
    create_initial_model,
    read_model_settings,
)
from gumbel_dlshogi.network import (
    add_compile_arguments,
    add_network_argument,
    resolve_compile_options,
)
from gumbel_dlshogi.selfplay import (
    add_qtransform_arguments,
    make_qtransform,
    selfplay_multiprocess,
)
from gumbel_dlshogi.train import add_gpus_argument, train


def parse_schedule(schedule_str):
    """Parses a schedule string like "1:100,10:200" into a sorted dict."""
    if not schedule_str:
        return None
    try:
        schedule = collections.OrderedDict()
        for item in schedule_str.split(","):
            cycle, value = item.split(":")
            schedule[int(cycle)] = int(value)
        # Sort by cycle number
        return collections.OrderedDict(sorted(schedule.items()))
    except ValueError:
        raise ValueError(f"Invalid schedule format: {schedule_str}")


def get_scheduled_value(cycle, base_value, schedule):
    """Gets the value for the current cycle based on the schedule."""
    if not schedule:
        return base_value
    current_value = base_value
    for scheduled_cycle, value in schedule.items():
        if cycle >= scheduled_cycle:
            current_value = value
        else:
            break
    return current_value


def model_path(models_dir, cycle):
    """Path of the model saved at the end of the cycle (cycle 0 is the initial model)"""
    return models_dir / f"model_{cycle:08d}.pt"


def record_path(models_dir, cycle):
    """Path of the record written when the cycle has completed

    The record of cycle 0 is the model settings saved with the initial model.
    """
    return model_path(models_dir, cycle).with_suffix(".json")


def write_cycle_record(models_dir, cycle, checkpoint, model_settings):
    """Records that the cycle has completed, with the checkpoint to resume from.

    It is written after the model of the cycle, so the record marks the completion.
    It has the same "model_settings" and "files" as the settings of the initial model,
    so that the model can be checked when it is used as an initial model.
    """
    record = {
        "checkpoint": Path(checkpoint).name,
        "model_settings": model_settings,
        "files": [model_path(models_dir, cycle).name],
    }
    save_atomically(
        record_path(models_dir, cycle),
        lambda path: path.write_text(json.dumps(record, indent=2), encoding="utf-8"),
    )


def read_cycle_record(models_dir, cycle):
    """Returns the record of the completed cycle, or None if it has not completed."""
    path = record_path(models_dir, cycle)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_completed_cycles(models_dir):
    """Returns the cycles with a completion record."""
    return [
        int(match[1])
        for path in models_dir.glob("model_*.json")
        if (match := re.fullmatch(r"model_(\d+)\.json", path.name))
    ]


def invalidate_cycle_records(models_dir, start_cycle):
    """Removes the completion records of start_cycle and later cycles.

    Their models and checkpoints are overwritten by running them again, so the old
    records must not mark them as completed if the run stops on the way.
    """
    for cycle in sorted(find_completed_cycles(models_dir)):
        if cycle >= start_cycle:
            record_path(models_dir, cycle).unlink()
            print(f"Removed the completion record of cycle {cycle} to run it again")


def find_resume_paths(models_dir, checkpoint_dir, cycle, model_settings):
    """Returns the model and checkpoint of the completed cycle to resume from.

    Raises:
        ValueError: If the cycle cannot be resumed from.
    """
    resume_model_path = model_path(models_dir, cycle)
    if not resume_model_path.exists():
        raise ValueError(f"Model for cycle {cycle} not found at {resume_model_path}")
    record = read_cycle_record(models_dir, cycle)
    if record is None:
        raise ValueError(
            f"Cycle {cycle} has no completion record {record_path(models_dir, cycle)}"
        )
    if record["model_settings"] != model_settings:
        raise ValueError(
            f"The model settings differ from those of cycle {cycle}: "
            f"{record['model_settings']}"
        )
    resume_checkpoint = checkpoint_dir / record["checkpoint"]
    if not resume_checkpoint.exists():
        raise ValueError(
            f"The checkpoint {resume_checkpoint} of cycle {cycle} is not found"
        )
    return resume_model_path, resume_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Automate self-play and training cycles."
    )

    # Cycle arguments
    parser.add_argument(
        "--cycles", type=int, default=10, help="Number of cycles to run."
    )
    parser.add_argument(
        "--start_cycle",
        type=int,
        help="The cycle number to start from. If omitted, resume after the last "
        "completed cycle in the workspace, or start from cycle 1 with an initial model "
        "(--initial_model or --initial_model_npz).",
    )
    parser.add_argument(
        "--initial_model",
        type=str,
        help="Path to the initial model (.pt file). If omitted when starting from "
        "cycle 1, an initial model is created in the workspace.",
    )
    parser.add_argument(
        "--initial_model_state",
        type=str,
        help="Path to the initial model state (.pth file) used with --initial_model.",
    )
    parser.add_argument(
        "--initial_model_npz",
        type=str,
        help="Path to a model file (.npz, e.g. of dlshogi) of the network given by "
        "--network. The initial model is created from it in the workspace.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="Directory to store models and data.",
    )

    # Self-play arguments
    parser.add_argument("--selfplay_batch_size", type=int, default=64)
    parser.add_argument("--max_num_considered_actions", type=int, default=16)
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument("--num_positions", type=int, default=1000000)
    parser.add_argument("--selfplay_amp", action="store_true")
    parser.add_argument(
        "--selfplay_compile",
        action="store_true",
        help="Compile the model for self-play with torch.compile",
    )
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--skip_max_moves", action="store_true")
    add_qtransform_arguments(parser)

    # Training arguments
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_file")
    parser.add_argument("--num_files", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_num_workers", type=int, default=0)
    add_gpus_argument(parser, "--train_gpus")
    parser.add_argument("--train_amp", action="store_true")
    parser.add_argument(
        "--train_compile",
        action="store_true",
        help="Compile the model for training with torch.compile",
    )
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    add_network_argument(parser)
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features in the input",
    )

    # torch.compile arguments (for self-play and training)
    add_compile_arguments(parser)

    # Schedule arguments
    parser.add_argument(
        "--num_simulations_schedule",
        type=str,
        help='Schedule for num_simulations, e.g., "10:32,100:64,1000:128"',
    )
    parser.add_argument(
        "--num_files_schedule",
        type=str,
        help='Schedule for num_files, e.g., "10:2,100:4,1000:8"',
    )
    parser.add_argument(
        "--train_batch_size_schedule",
        type=str,
        help='Schedule for train_batch_size, e.g., "10:512,100:1024,1000:2048"',
    )

    args = parser.parse_args()
    if args.train_epochs < 1:
        parser.error("--train_epochs must be at least 1.")
    compile_options = {"backend": args.compile_backend, "mode": args.compile_mode}
    if args.selfplay_compile or args.train_compile:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolve_compile_options(device, **compile_options)
        except ValueError as e:
            parser.error(str(e))

    # Parse schedules
    num_simulations_schedule = parse_schedule(args.num_simulations_schedule)
    num_files_schedule = parse_schedule(args.num_files_schedule)
    train_batch_size_schedule = parse_schedule(args.train_batch_size_schedule)
    if args.train_gpus:
        # Each GPU trains an equal part of the batch in data parallel training
        batch_sizes = [args.train_batch_size]
        if train_batch_size_schedule:
            batch_sizes += train_batch_size_schedule.values()
        invalid = [size for size in batch_sizes if size % len(args.train_gpus)]
        if invalid:
            parser.error(
                f"The train batch sizes {invalid} are not divisible by the number of "
                f"GPUs {len(args.train_gpus)}."
            )

    workspace = Path(args.workspace)
    models_dir = workspace / "models"
    data_dir = workspace / "data"
    log_dir = workspace / "logs"
    checkpoint_dir = workspace / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Determine the cycle to start from
    use_initial_model = (
        args.initial_model or args.initial_model_state or args.initial_model_npz
    )
    if args.start_cycle is None:
        if use_initial_model:
            args.start_cycle = 1
        else:
            args.start_cycle = max(find_completed_cycles(models_dir), default=0) + 1
    elif args.start_cycle > 1 and use_initial_model:
        parser.error(
            "--initial_model, --initial_model_state and --initial_model_npz are "
            "only used from cycle 1."
        )
    if args.initial_model_npz and (args.initial_model or args.initial_model_state):
        parser.error(
            "--initial_model_npz cannot be used with --initial_model or "
            "--initial_model_state."
        )
    if args.start_cycle > args.cycles:
        print(f"All {args.cycles} cycles have already been completed.")
        return
    if args.start_cycle > 1:
        print(f"Resuming from cycle {args.start_cycle}")

    # Model settings shared by the initial model, its check and training
    model_settings = {
        "network": args.network,
        "use_nyugyoku_features": args.use_nyugyoku_features,
    }

    # Determine the starting model path (and the checkpoint when resuming)
    initial_model_state_path = None
    resume_checkpoint = None
    if args.start_cycle == 1:
        initial_model_path = args.initial_model
        initial_model_state_path = args.initial_model_state
        if not initial_model_path:
            if initial_model_state_path:
                parser.error("--initial_model_state requires --initial_model.")
            # Create the initial model with the same settings as self-play and training
            initial_model_path = model_path(models_dir, 0)
            initial_model_state_path = initial_model_path.with_suffix(".pth")
            # The settings are saved last with the names of the saved files, so the
            # initial model is complete if they list both files. It is always created
            # from the npz file, since the existing one may be from another file.
            reuse_initial_model = not args.initial_model_npz and all(
                read_model_settings(path) is not None
                for path in (initial_model_path, initial_model_state_path)
            )
            if reuse_initial_model:
                print(f"Using the existing initial model {initial_model_path}")
            else:
                try:
                    create_initial_model(
                        models_dir,
                        initial_model_path.stem,
                        **model_settings,
                        npz=args.initial_model_npz,
                    )
                except ValueError as e:
                    # Raised only when the npz file does not match the network
                    parser.error(f"{e}. Give the network of the npz model.")

        # Check the initial model against the settings before starting self-play
        try:
            check_torchscript_model(initial_model_path, **model_settings)
            if initial_model_state_path:
                check_model_state(initial_model_state_path, **model_settings)
        except ValueError as e:
            parser.error(
                f"{e}. Use the same settings as the initial model, "
                "or remove it or use another workspace to create a new one."
            )
        latest_model_path = str(initial_model_path)
    else:
        # Resuming from a later cycle
        try:
            resume_model_path, resume_checkpoint = find_resume_paths(
                models_dir, checkpoint_dir, args.start_cycle - 1, model_settings
            )
        except ValueError as e:
            parser.error(f"{e}. Cannot resume.")
        latest_model_path = str(resume_model_path)

    # All checks have passed, so the cycles from start_cycle are run (again)
    invalidate_cycle_records(models_dir, args.start_cycle)
    start_model_path = model_path(models_dir, args.start_cycle)
    if start_model_path.exists():
        print(
            f"{start_model_path} already exists and will be replaced by running "
            f"cycle {args.start_cycle} again."
        )

    for cycle in range(args.start_cycle - 1, args.cycles):
        current_cycle_num = cycle + 1
        print(f"--- Starting Cycle {current_cycle_num}/{args.cycles} ---")

        # Get scheduled parameters for the current cycle
        num_simulations = get_scheduled_value(
            current_cycle_num, args.num_simulations, num_simulations_schedule
        )
        num_files = get_scheduled_value(
            current_cycle_num, args.num_files, num_files_schedule
        )
        train_batch_size = get_scheduled_value(
            current_cycle_num, args.train_batch_size, train_batch_size_schedule
        )
        if num_simulations_schedule or num_files_schedule or train_batch_size_schedule:
            print(f"Current parameters for cycle {current_cycle_num}:")
            print(f"  num_simulations: {num_simulations}")
            print(f"  num_files: {num_files}")
            print(f"  train_batch_size: {train_batch_size}")

        # --- Self-play Phase ---
        selfplay_stats = selfplay_multiprocess(
            model_path=latest_model_path,
            batch_size=args.selfplay_batch_size,
            max_num_considered_actions=args.max_num_considered_actions,
            num_simulations=num_simulations,
            output_dir=str(data_dir),
            num_positions=args.num_positions,
            amp=args.selfplay_amp,
            num_processes=args.num_processes,
            skip_max_moves=args.skip_max_moves,
            use_nyugyoku_features=args.use_nyugyoku_features,
            compile_options=compile_options if args.selfplay_compile else None,
            network=args.network,
            qtransform=make_qtransform(args.value_scale, args.rescale_values),
        )

        # Log self-play stats to TensorBoard
        if selfplay_stats:
            print("--- Self-play Stats ---")
            writer = SummaryWriter(log_dir=str(log_dir))
            for key, value in selfplay_stats.items():
                print(f"  {key}: {value}")
                writer.add_scalar(f"selfplay/{key}", value, current_cycle_num)

            # Log the parameters used for this cycle
            writer.add_scalar(
                "parameters/selfplay/num_simulations",
                num_simulations,
                current_cycle_num,
            )
            writer.add_scalar(
                "parameters/selfplay/value_scale", args.value_scale, current_cycle_num
            )
            writer.add_scalar(
                "parameters/selfplay/rescale_values",
                int(args.rescale_values),
                current_cycle_num,
            )
            writer.add_scalar(
                "parameters/training/num_files", num_files, current_cycle_num
            )
            writer.add_scalar(
                "parameters/training/batch_size", train_batch_size, current_cycle_num
            )

            writer.close()

        # --- Training Phase ---
        print("--- Training Phase ---")
        next_model_path = model_path(models_dir, cycle + 1)

        # Determine which model/checkpoint to use for training
        initial_model_state = None
        resume_path = None
        if cycle == 0:
            # For the very first cycle, use the initial model
            initial_model_state = initial_model_state_path
        else:
            # For subsequent cycles, resume from the checkpoint of the previous cycle
            resume_path = str(resume_checkpoint)

        resume_checkpoint = train(
            train_dir=str(data_dir),
            test_file=args.test_file,
            num_files=num_files,
            initial_model=initial_model_state,
            epochs=args.train_epochs,
            batch_size=train_batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.train_num_workers,
            amp=args.train_amp,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            resume=resume_path,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            save_torchscript=str(next_model_path),
            compile_options=compile_options if args.train_compile else None,
            gpus=args.train_gpus,
            **model_settings,
        )
        write_cycle_record(models_dir, cycle + 1, resume_checkpoint, model_settings)

        latest_model_path = str(next_model_path)

    print("--- All cycles completed ---")


if __name__ == "__main__":
    main()
