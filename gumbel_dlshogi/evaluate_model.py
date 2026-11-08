"""Evaluates a model on self-play data not used for training.

Generate the data with selfplay.py into a directory that training does not read,
so that the data are separated from the training data by games, e.g.:

    python -m gumbel_dlshogi.selfplay model.pt --output_dir validation --num_processes 4
    python -m gumbel_dlshogi.evaluate_model model.pt validation

It reports, over all the positions, the metrics of the policy and the value with the
same definitions as the training losses, so that models with and without the
moves-left head can be compared:

- policy: the cross entropy with the policy targets (the improved policies of the
  search), and the agreement of the most probable moves
- value: the binary cross entropy and the Brier score with the game results, and the
  accuracy (the result is a win if it is more than 0.5, as the training evaluation)

With the moves-left head, it also reports the mean absolute error (plies) of the
moves left, over the positions with them (not of the games ended by the maximum
number of moves): over all of them, over the winning and losing positions by the
value of the model (where the bonus of the search is applied: |2V - 1| > threshold),
and by the range of the moves left.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from gumbel_dlshogi.common import MOVES_LEFT_INVALID, bounded_float_type
from gumbel_dlshogi.data_loader import create_dataloader
from gumbel_dlshogi.initial_model import read_model_settings
from gumbel_dlshogi.network.moves_left import MOVES_LEFT_SCALE

# Ranges of the moves left (plies) to report the errors by
MOVES_LEFT_BINS = (0, 10, 30, 60, 120, 240, np.inf)


def evaluate(model_path, data_dir, batch_size, use_nyugyoku_features, amp, num_files):
    """Evaluates the TorchScript model on the data.

    The network in the model (base_model of the wrapper for inference) is used, so
    that the value BCE is computed from the logits as the training loss: the win
    rate of the wrapper saturates, e.g. in float16 with mixed precision. The win
    rates are computed from the logits in float32.

    Returns:
        `(metrics, moves_left)`: metrics are the policy and value metrics over all
        the positions. moves_left is None without the head, or `(targets,
        predictions, values)` of the positions with the moves left.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device).eval()
    if not hasattr(model, "base_model"):
        raise ValueError(f"{model_path} is not a model saved by this package")
    network = model.base_model
    dataloader = create_dataloader(
        data_dir,
        batch_size=batch_size,
        num_files=num_files,
        shuffle=False,
        drop_last=False,
        use_nyugyoku_features=use_nyugyoku_features,
    )

    sums = dict.fromkeys(
        ("policy_loss", "policy_agreement", "value_loss", "brier", "value_accuracy"),
        0.0,
    )
    positions = 0
    moves_left = None
    with torch.no_grad():
        for features1, features2, policies, results, targets in dataloader:
            with autocast(device.type, enabled=amp):
                outputs = network(features1.to(device), features2.to(device))
            logits = outputs[0].float().cpu()
            value_logits = outputs[1].float().cpu().reshape(-1)
            values = torch.sigmoid(value_logits)

            # The same as the training losses
            sums["policy_loss"] += float(
                F.cross_entropy(logits, policies, reduction="sum")
            )
            sums["policy_agreement"] += float(
                (logits.argmax(dim=1) == policies.argmax(dim=1)).sum()
            )
            sums["value_loss"] += float(
                F.binary_cross_entropy_with_logits(
                    value_logits, results, reduction="sum"
                )
            )
            sums["brier"] += float(((values - results) ** 2).sum())
            sums["value_accuracy"] += float(((values > 0.5) == (results > 0.5)).sum())
            positions += len(results)

            if len(outputs) == 3:
                if moves_left is None:
                    moves_left = ([], [], [])
                valid = targets != MOVES_LEFT_INVALID
                moves_left[0].append(targets[valid].numpy())
                # The network predicts the plies divided by MOVES_LEFT_SCALE
                predictions = outputs[2].float().cpu().reshape(-1) * MOVES_LEFT_SCALE
                moves_left[1].append(predictions[valid].numpy())
                moves_left[2].append(values[valid].numpy())

    metrics = {key: value / max(positions, 1) for key, value in sums.items()}
    metrics["positions"] = positions
    if moves_left is not None:
        moves_left = tuple(
            np.concatenate(arrays).astype(np.float64) for arrays in moves_left
        )
    return metrics, moves_left


def moves_left_errors(targets, predictions, values, threshold):
    """Returns the errors (plies) of the predictions by the groups of positions.

    Returns:
        A list of `(group, positions, mean absolute error, mean error)`. The mean
        error is the prediction minus the target (positive if it predicts longer).
    """
    errors = predictions - targets
    centered = 2 * values - 1
    groups = [
        ("all", np.ones(len(targets), dtype=bool)),
        (f"winning (2V-1 > {threshold})", centered > threshold),
        (f"losing (2V-1 < -{threshold})", centered < -threshold),
        (f"even (|2V-1| <= {threshold})", np.abs(centered) <= threshold),
    ]
    for low, high in zip(MOVES_LEFT_BINS[:-1], MOVES_LEFT_BINS[1:]):
        in_range = (targets >= low) & (targets < high)
        groups.append((f"moves left [{low}, {high})", in_range))

    results = []
    for name, mask in groups:
        group_errors = errors[mask]
        count = group_errors.size
        if count == 0:
            results.append((name, 0, float("nan"), float("nan")))
        else:
            mae = float(np.abs(group_errors).mean())
            results.append((name, count, mae, float(group_errors.mean())))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on self-play data not used for training"
    )
    parser.add_argument("model", help="TorchScript model (.pt)")
    parser.add_argument("data_dir", help="Self-play data not used for training")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_files", type=int, help="Number of newest files to use")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument(
        "--threshold",
        type=bounded_float_type("threshold", 0, 1, include_maximum=False),
        default=0.6,
        help="Advantage |2V - 1| of the winning and losing positions for the "
        "moves-left errors (the same as --moves_left_threshold of the search)",
    )
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features (read from the settings of the model if saved)",
    )
    args = parser.parse_args()

    model_settings = read_model_settings(args.model)
    use_nyugyoku_features = args.use_nyugyoku_features
    if model_settings is not None:
        use_nyugyoku_features = model_settings["use_nyugyoku_features"]

    try:
        metrics, moves_left = evaluate(
            args.model,
            args.data_dir,
            args.batch_size,
            use_nyugyoku_features,
            args.amp,
            args.num_files,
        )
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))
    if metrics["positions"] == 0:
        parser.error(f"No positions are found in {args.data_dir}")

    print(f"positions: {metrics['positions']}")
    print(f"policy cross entropy: {metrics['policy_loss']:.4f}")
    print(f"policy agreement:     {metrics['policy_agreement']:.4f}")
    print(f"value BCE:            {metrics['value_loss']:.4f}")
    print(f"value Brier score:    {metrics['brier']:.4f}")
    print(f"value accuracy:       {metrics['value_accuracy']:.4f}")

    if moves_left is None:
        return
    targets, predictions, values = moves_left
    print()
    if len(targets) == 0:
        print("moves left: no positions with the moves left")
        return
    print(f"{'moves left group':34s} {'positions':>10s} {'MAE':>8s} {'bias':>8s}")
    for name, count, mae, bias in moves_left_errors(
        targets, predictions, values, args.threshold
    ):
        print(f"{name:34s} {count:10d} {mae:8.2f} {bias:8.2f}")


if __name__ == "__main__":
    main()
