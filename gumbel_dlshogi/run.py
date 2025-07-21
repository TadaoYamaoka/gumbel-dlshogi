import argparse
import collections
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from gumbel_dlshogi.selfplay import selfplay_multiprocess
from gumbel_dlshogi.train import train


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
        default=1,
        help="The cycle number to start from.",
    )
    parser.add_argument(
        "--initial_model",
        type=str,
        help="Path to the initial model (.pt file).",
    )
    parser.add_argument(
        "--initial_model_state",
        type=str,
        help="Path to the initial model state (.pth file).",
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
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--skip_max_moves", action="store_true")

    # Training arguments
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_file")
    parser.add_argument("--num_files", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_num_workers", type=int, default=4)
    parser.add_argument("--train_amp", action="store_true")
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--fcl", type=int, default=256)

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

    # Parse schedules
    num_simulations_schedule = parse_schedule(args.num_simulations_schedule)
    num_files_schedule = parse_schedule(args.num_files_schedule)
    train_batch_size_schedule = parse_schedule(args.train_batch_size_schedule)

    workspace = Path(args.workspace)
    models_dir = workspace / "models"
    data_dir = workspace / "data"
    log_dir = workspace / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine the starting model path
    if args.start_cycle == 1:
        if not args.initial_model:
            parser.error("--initial_model is required when starting from cycle 1.")
        latest_model_path = args.initial_model
    else:
        # Resuming from a later cycle
        prev_cycle_num = args.start_cycle - 1
        resume_model_path = models_dir / f"model_{prev_cycle_num:08d}.pt"
        if not resume_model_path.exists():
            parser.error(
                f"Model for cycle {prev_cycle_num} not found at {resume_model_path}. "
                "Cannot resume."
            )
        latest_model_path = str(resume_model_path)

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
                "parameters/training/num_files", num_files, current_cycle_num
            )
            writer.add_scalar(
                "parameters/training/batch_size", train_batch_size, current_cycle_num
            )

            writer.close()

        # --- Training Phase ---
        print("--- Training Phase ---")
        checkpoint_dir = workspace / f"checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        next_model_path = models_dir / f"model_{cycle + 1:08d}.pt"

        # Determine which model/checkpoint to use for training
        initial_model_state = None
        resume_path = None
        if cycle == 0:
            # For the very first cycle, use the initial model provided
            initial_model_state = args.initial_model_state
        else:
            # For subsequent cycles, resume from the last checkpoint of the previous cycle
            checkpoints = sorted(
                checkpoint_dir.glob("*.pth"),
                key=lambda p: int(p.name.split("_")[-1].split(".")[0]),
            )
            resume_path = str(checkpoints[-1])

        train(
            train_dir=str(data_dir),
            test_file=args.test_file,
            num_files=num_files,
            blocks=args.blocks,
            channels=args.channels,
            fcl=args.fcl,
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
        )

        latest_model_path = str(next_model_path)

    print("--- All cycles completed ---")


if __name__ == "__main__":
    main()
