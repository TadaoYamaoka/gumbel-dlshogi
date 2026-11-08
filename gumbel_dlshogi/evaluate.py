"""Plays two models against each other to compare their strength.

Each start position (the initial position, or from a file of start positions) is
played twice, with the colors swapped. The search of a move uses only the model of
the side to move at its root, also for the positions of the other side in the tree.
The games are played concurrently, and the positions are evaluated in batches for
each model.

    python -m gumbel_dlshogi.evaluate model_a.pt model_b.pt --games 100

It reports, for each model, the wins, draws and losses (overall and by color), the
score and the Elo difference, how the games ended, the average plies of the games it
won and lost, and the time per move. The search settings are the same for both
models except the moves-left bonus (--moves_left_beta_a and --moves_left_beta_b).
The Gumbel noise of the search (--gumbel_scale) varies the games; with 0 the search
is deterministic, so the two games of a position between the same model are the same.
"""

import argparse
import math
import time
from collections import deque

import numpy as np
import torch
from cshogi import BLACK, BLACK_WIN, DRAW, WHITE_WIN, Board, move_to_usi
from torch.amp import autocast

from gumbel_dlshogi.common import bounded_float_type
from gumbel_dlshogi.features import make_input_features, set_nyugyoku_features
from gumbel_dlshogi.game import END_REASONS, end_reason, game_result
from gumbel_dlshogi.initial_model import read_model_settings
from gumbel_dlshogi.mcts.action_selection import init_table
from gumbel_dlshogi.selfplay import (
    MoveSearch,
    allocate_input_buffers,
    load_model,
    set_evaluations,
    add_qtransform_arguments,
    make_qtransform,
    model_has_moves_left,
    move_from_label,
)
from gumbel_dlshogi.start_positions import (
    STARTPOS,
    StartPositions,
    add_start_positions_arguments,
    check_roots_file,
)


class Game:
    """A game of the match, whose moves are searched by the model of the side to move.

    Like MoveSearch, the caller evaluates the position of board with the model of
    `mover` and sets the result to step after each call of next().
    """

    def __init__(self, pair, start_sfen, black, searches_settings, gumbel_scale):
        """
        Args:
            pair: The index of the pair of the games from the same start position
            start_sfen: The start position, or None for the initial position
            black: The index of the player (model) playing black
            searches_settings: `(max_num_considered_actions, num_simulations,
                qtransform)` of each player
        """
        self.pair = pair
        self.start_sfen = start_sfen
        self.black = black
        self.board = Board() if start_sfen is None else Board(start_sfen)
        self.searches = [
            MoveSearch(
                self.board,
                max_num_considered_actions,
                num_simulations,
                qtransform=qtransform,
                gumbel_scale=gumbel_scale,
            )
            for max_num_considered_actions, num_simulations, qtransform in (
                searches_settings
            )
        ]
        self.moves = []
        self.reason = None
        self.result = None
        self._update_mover()

    def _update_mover(self):
        # The player of the side to move at the root, fixed during the search (the
        # board moves along the tree while searching)
        self.mover = self.black if self.board.turn == BLACK else 1 - self.black

    @property
    def step(self):
        """The evaluation of the position of board to set (by the model of mover)"""
        return self.searches[self.mover].step

    def next(self):
        """Advances the search of the side to move. Returns True if the game ended."""
        output = self.searches[self.mover].next()
        if output is None:
            return False
        move = move_from_label(self.board, output.action)
        self.moves.append(move_to_usi(move))
        self.board.push(move)
        self.reason = end_reason(self.board)
        if self.reason is not None:
            self.result = game_result(self.reason, self.board.turn)
            return True
        self._update_mover()
        return False

    def score(self, player):
        """The score of the player in the ended game (1 for a win, 0.5 for a draw)."""
        if self.result == DRAW:
            return 0.5
        black_wins = self.result == BLACK_WIN
        return 1.0 if black_wins == (player == self.black) else 0.0

    def record(self, names):
        position = "startpos" if self.start_sfen is None else f"sfen {self.start_sfen}"
        result = {BLACK_WIN: "black_win", WHITE_WIN: "white_win", DRAW: "draw"}
        return (
            f"{position} moves {' '.join(self.moves)}\tblack={names[self.black]}\t"
            f"white={names[1 - self.black]}\tresult={result[self.result]}\t"
            f"reason={self.reason}"
        )


def play_match(
    models,
    qtransforms,
    *,
    num_pairs,
    batch_size,
    max_num_considered_actions,
    num_simulations,
    gumbel_scale,
    start_positions,
    amp,
    device,
):
    """Plays the games of the match, and returns them with the time of each player.

    Returns:
        `(games, times, moves)`: the ended games, the seconds spent by each player
        (searching and evaluating) and the number of the moves of each player.
    """
    init_table(max_num_considered_actions, num_simulations)
    settings = [(max_num_considered_actions, num_simulations, q) for q in qtransforms]

    # Each start position is played twice, with the colors swapped
    pending = deque()
    board = Board()
    for pair in range(num_pairs):
        source = start_positions.set_start_position(board)
        start_sfen = None if source == STARTPOS else board.sfen()
        pending.append((pair, start_sfen, 0))
        pending.append((pair, start_sfen, 1))

    def next_game():
        if not pending:
            return None
        pair, start_sfen, black = pending.popleft()
        return Game(pair, start_sfen, black, settings, gumbel_scale)

    slots = [next_game() for _ in range(min(batch_size, len(pending)))]
    x1, x2 = allocate_input_buffers(len(slots))
    features1, features2 = x1.numpy(), x2.numpy()
    ended = []
    times = [0.0, 0.0]
    moves = [0, 0]

    while any(game is not None for game in slots):
        # Evaluate the positions waiting for it (the start of a search or a leaf),
        # with the model of the side to move at the root of the search
        for player, model in enumerate(models):
            indices = [
                i
                for i, game in enumerate(slots)
                if game is not None and game.mover == player
            ]
            if not indices:
                continue
            start = time.perf_counter()
            # Packed at the front of the buffers, to transfer a contiguous view of
            # the (pinned) buffers without a copy
            for j, i in enumerate(indices):
                make_input_features(slots[i].board, features1[j], features2[j])
            n = len(indices)
            with torch.no_grad(), autocast(device.type, enabled=amp):
                outputs = model(x1[:n].to(device), x2[:n].to(device))
            set_evaluations([slots[i] for i in indices], outputs)
            times[player] += time.perf_counter() - start

        # Advance the searches with the evaluations
        for i, game in enumerate(slots):
            if game is None:
                continue
            mover = game.mover
            num_moves = len(game.moves)
            start = time.perf_counter()
            if game.next():
                ended.append(game)
                slots[i] = next_game()
            times[mover] += time.perf_counter() - start
            moves[mover] += len(game.moves) - num_moves
    return ended, times, moves


def elo(score):
    """Returns the Elo difference for the score (inf for 0 or 1)."""
    if score <= 0:
        return -math.inf
    if score >= 1:
        return math.inf
    # Adding 0.0 turns -0.0 to 0.0 (an even score)
    return -400 * math.log10(1 / score - 1) + 0.0


# The 97.5% quantiles of the t distribution by the degrees of freedom (1 to 30)
_T_QUANTILES = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def _t_quantile(df):
    """Returns the 97.5% quantile of the t distribution (for a 95% interval)."""
    if df <= len(_T_QUANTILES):
        return _T_QUANTILES[df - 1]
    # The Cornish-Fisher expansion from the normal quantile, accurate above 30
    z = 1.959964
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def score_interval(games, player):
    """Returns an approximate 95% interval of the score of the player.

    The two games from the same start position are not independent, so the unit is
    the pair: the t interval of the mean of the scores of the pairs. It is an
    approximation (e.g. about 94% of coverage with 10 pairs of only wins or losses
    of both games). Without variance (fewer than 2 pairs, or all the pairs of the
    same score), the pairs do not tell the uncertainty, so it is [0, 1].
    """
    pairs = {}
    for game in games:
        pairs.setdefault(game.pair, []).append(game.score(player))
    scores = np.array([np.mean(pair) for pair in pairs.values()])
    if len(scores) < 2 or np.all(scores == scores[0]):
        return 0.0, 1.0
    margin = (
        _t_quantile(len(scores) - 1) * scores.std(ddof=1) / math.sqrt(len(scores))
    )
    return max(scores.mean() - margin, 0.0), min(scores.mean() + margin, 1.0)


def summarize(games, player, times, moves):
    """Returns the lines of the results of the player."""
    scores = np.array([game.score(player) for game in games])
    n = len(scores)
    score = scores.mean()
    low, high = score_interval(games, player)
    num_pairs = len({game.pair for game in games})
    lines = [
        f"games {n}: wins {int((scores == 1).sum())}, draws "
        f"{int((scores == 0.5).sum())}, losses {int((scores == 0).sum())}, "
        f"score {score:.3f}, Elo {elo(score):+.0f} "
        f"(approx. 95% by {num_pairs} pairs: score {low:.3f}-{high:.3f}, "
        f"Elo {elo(low):+.0f} to {elo(high):+.0f})"
    ]
    for is_black, color in ((True, "black"), (False, "white")):
        color_scores = [
            game.score(player)
            for game in games
            if (game.black == player) == is_black
        ]
        if color_scores:
            lines.append(
                f"  as {color}: games {len(color_scores)}, "
                f"score {np.mean(color_scores):.3f}"
            )
    for outcome, value in (("won", 1.0), ("drawn", 0.5), ("lost", 0.0)):
        plies = [len(game.moves) for game in games if game.score(player) == value]
        if plies:
            lines.append(
                f"  average plies of the games {outcome}: {np.mean(plies):.1f}"
            )
    lines.append(
        f"  time per move: {1000 * times[player] / max(moves[player], 1):.1f} ms "
        f"({moves[player]} moves)"
    )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Play two models against each other to compare their strength"
    )
    parser.add_argument("model_a", help="TorchScript model (.pt)")
    parser.add_argument("model_b", help="TorchScript model (.pt)")
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of the start positions, each played twice with the colors "
        "swapped",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of the games played at once"
    )
    parser.add_argument("--max_num_considered_actions", type=int, default=16)
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument(
        "--gumbel_scale",
        type=bounded_float_type("gumbel_scale", 0),
        default=1.0,
        help="Scale of the Gumbel noise of the search (0 for a deterministic search)",
    )
    add_qtransform_arguments(parser)
    for player in ("a", "b"):
        parser.add_argument(
            f"--moves_left_beta_{player}",
            type=bounded_float_type(f"moves_left_beta_{player}", 0),
            default=0.0,
            help=f"Moves-left bonus of the search of model {player.upper()} (it "
            "needs the moves-left head unless 0)",
        )
    parser.add_argument(
        "--moves_left_tau",
        type=bounded_float_type("moves_left_tau", 0, include_minimum=False),
        default=20.0,
    )
    parser.add_argument(
        "--moves_left_threshold",
        type=bounded_float_type("moves_left_threshold", 0, 1, include_maximum=False),
        default=0.6,
    )
    add_start_positions_arguments(parser)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, help="Seed of the noise and the positions")
    parser.add_argument("--kifu", help="Write the games to this file")
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features (read from the settings of the models if saved)",
    )
    args = parser.parse_args()
    if args.games < 1 or args.batch_size < 1:
        parser.error("--games and --batch_size must be at least 1.")
    if args.roots is not None:
        try:
            check_roots_file(args.roots)
        except ValueError as e:
            parser.error(str(e))

    paths = (args.model_a, args.model_b)
    betas = (args.moves_left_beta_a, args.moves_left_beta_b)
    # The input features are set per process, so both models must use the same
    features = set()
    for path, beta in zip(paths, betas):
        settings = read_model_settings(path)
        features.add(
            args.use_nyugyoku_features
            if settings is None
            else settings["use_nyugyoku_features"]
        )
        if beta > 0 and not model_has_moves_left(path):
            parser.error(f"The moves-left bonus needs the moves-left head: {path}")
    if len(features) > 1:
        parser.error("The models use different input features (nyugyoku features)")
    set_nyugyoku_features(features.pop())

    if args.seed is not None:
        np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(path, device, None, None) for path in paths]
    qtransforms = [
        make_qtransform(
            args.value_scale,
            args.rescale_values,
            moves_left_beta=beta,
            moves_left_tau=args.moves_left_tau,
            moves_left_threshold=args.moves_left_threshold,
        )
        for beta in betas
    ]
    start_positions = StartPositions(
        args.roots,
        args.roots_ratio,
        None if args.seed is None else np.random.default_rng(args.seed),
    )

    start = time.perf_counter()
    games, times, moves = play_match(
        models,
        qtransforms,
        num_pairs=args.games,
        batch_size=args.batch_size,
        max_num_considered_actions=args.max_num_considered_actions,
        num_simulations=args.num_simulations,
        gumbel_scale=args.gumbel_scale,
        start_positions=start_positions,
        amp=args.amp,
        device=device,
    )
    elapsed = time.perf_counter() - start

    names = ("A", "B")
    if args.kifu:
        with open(args.kifu, "w", encoding="utf-8") as f:
            for game in games:
                f.write(game.record(names) + "\n")

    for name, path, beta in zip(names, paths, betas):
        print(f"{name}: {path}" + (f" (moves_left_beta {beta})" if beta else ""))
    print(f"{len(games)} games in {elapsed:.1f} s")
    for player, name in enumerate(names):
        lines = summarize(games, player, times, moves)
        print(f"{name}: " + lines[0])
        for line in lines[1:]:
            print(line)
    print("end reasons (games, average plies):")
    for reason in END_REASONS:
        plies = [len(game.moves) for game in games if game.reason == reason]
        if plies:
            print(f"  {reason}: {len(plies)}, {np.mean(plies):.1f}")


if __name__ == "__main__":
    main()
