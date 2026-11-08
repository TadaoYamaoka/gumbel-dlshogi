import argparse
import functools
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from queue import Queue

import numpy as np
import torch
from cshogi import (
    BLACK,
    BLACK_WIN,
    DRAW,
    NOT_REPETITION,
    REPETITION_DRAW,
    REPETITION_LOSE,
    REPETITION_WIN,
    WHITE_WIN,
    Board,
    HuffmanCodedPos,
    move_to_usi,
)
from torch.amp import autocast
from tqdm import tqdm

from gumbel_dlshogi.common import (
    MOVES_LEFT_FILE_EXT,
    MOVES_LEFT_INVALID,
    POLICY_FILE_EXT,
    POSITION_FILE_EXT,
    TEMPORARY_FILE_EXT,
    dtypeMovePolicy,
    dtypeMovesLeft,
    dtypePosition,
)
from gumbel_dlshogi.features import (
    FEATURES1_NUM,
    MOVE_LABELS_NUM,
    features2_num,
    make_input_features,
    make_move_label,
    set_nyugyoku_features,
)
from gumbel_dlshogi.initial_model import (
    check_torchscript_model,
    read_model_settings,
)
from gumbel_dlshogi.mcts import base
from gumbel_dlshogi.mcts.action_selection import init_table
from gumbel_dlshogi.mcts.base import EvaluationStep, RootFnOutput
from gumbel_dlshogi.mcts.policies import _mask_invalid_actions, gumbel_muzero_policy
from gumbel_dlshogi.mcts.qtransforms import (
    qtransform_completed_by_mix_value,
    qtransform_with_moves_left,
)
from gumbel_dlshogi.mcts.search import MAX_MOVES
from gumbel_dlshogi.network.moves_left import MOVES_LEFT_PREFIX
from gumbel_dlshogi.start_positions import (
    SOURCES,
    STARTPOS,
    StartPositions,
    add_start_positions_arguments,
    check_roots_file,
)
from gumbel_dlshogi.network import (
    add_compile_arguments,
    add_network_argument,
    compile_model,
    inference_wrapper,
    load_torchscript_state_dict,
    policy_value_network,
    resolve_compile_options,
)


@dataclass
class TrainingData:
    hcps: np.ndarray
    policy_outputs: list[base.PolicyOutput]
    turn: int
    is_mate_in_3ply: bool
    is_game_over: bool
    is_nyugyoku: bool
    is_draw: int
    is_max_moves: bool
    source: str  # Source of the start position (STARTPOS or ROOTS)


def _get_invalid_actions(board: Board) -> np.ndarray:
    """指定された局面の非合法手のマスクを生成する"""
    invalid_actions = np.ones(MOVE_LABELS_NUM, dtype=np.int32)
    legal_move_labels = [
        make_move_label(move, board.turn) for move in board.legal_moves
    ]
    if legal_move_labels:
        invalid_actions[legal_move_labels] = 0
    return invalid_actions


class Actor:
    def __init__(
        self,
        max_num_considered_actions,
        num_simulations,
        queue: Queue,
        debug=False,
        qtransform=qtransform_completed_by_mix_value,
        start_positions=None,
        target_qtransform=None,
    ):
        self.board = Board()
        self.max_num_considered_actions = max_num_considered_actions
        self.num_simulations = num_simulations
        self.qtransform = qtransform
        self.target_qtransform = target_qtransform
        self.start_positions = (
            StartPositions() if start_positions is None else start_positions
        )
        # The first game starts at the first call of next()
        self.step = None
        self.generator = None
        self.queue = queue
        self.debug = debug

        self.hcps = np.empty(MAX_MOVES, dtype=HuffmanCodedPos)
        self.policy_outputs = []

    def _start_game(self):
        """Starts a new game, waiting for the evaluation of the start position.

        The caller evaluates the position of board and sets the result to step
        before the next call of next(), which starts the search.
        """
        self.source = self.start_positions.set_start_position(self.board)
        self.start_sfen = self.board.sfen()
        self.step = EvaluationStep()
        self.generator = None
        self.policy_outputs = []

    def next(self):
        if self.step is None:
            self._start_game()
            return

        if self.generator is None:
            self.root = RootFnOutput(
                prior_logits=self.step.prior_logits,
                value=self.step.value,
                moves_left=self.step.moves_left,
            )
            invalid_actions = _get_invalid_actions(self.board)
            self.generator = gumbel_muzero_policy(
                self.board,
                self.root,
                self.num_simulations,
                invalid_actions,
                self.max_num_considered_actions,
                qtransform=self.qtransform,
                target_qtransform=self.target_qtransform,
            )

        try:
            self.step = next(self.generator)
        except StopIteration as e:
            policy_output = e.value

            self.board.to_hcp(np.asarray(self.hcps[len(self.policy_outputs)]))
            self.policy_outputs.append(policy_output)

            action = policy_output.action
            move = next(
                (
                    move
                    for move in self.board.legal_moves
                    if make_move_label(move, self.board.turn) == action
                ),
                None,
            )
            assert move is not None
            self.board.push(move)
            self.step = EvaluationStep()
            self.generator = None

            is_nyugyoku = False
            is_draw = NOT_REPETITION
            is_max_moves = False
            is_mate_in_3ply = False
            if (
                (is_game_over := self.board.is_game_over())
                or (is_nyugyoku := self.board.is_nyugyoku())
                or (is_draw := self.board.is_draw())
                in (REPETITION_DRAW, REPETITION_WIN, REPETITION_LOSE)
                or (is_max_moves := self.board.move_number >= MAX_MOVES)
                or (is_mate_in_3ply := self.board.mate_move(3) != 0)
            ):
                if self.debug:
                    # The history starts from the start position
                    position = (
                        "startpos"
                        if self.source == STARTPOS
                        else f"sfen {self.start_sfen}"
                    )
                    moves = " ".join(move_to_usi(move) for move in self.board.history)
                    print(f"{position} moves {moves}")
                self.queue.put(
                    TrainingData(
                        hcps=self.hcps[: len(self.policy_outputs)].copy(),
                        policy_outputs=self.policy_outputs,
                        turn=self.board.turn,
                        is_mate_in_3ply=is_mate_in_3ply,
                        is_game_over=is_game_over,
                        is_nyugyoku=is_nyugyoku,
                        is_draw=is_draw,
                        is_max_moves=is_max_moves,
                        source=self.source,
                    )
                )

                self._start_game()


def _allocate_input_buffers(batch_size):
    """推論用の入力バッファ x1, x2 を確保する"""
    # pin_memory は CUDA がある場合のみ使える
    pin_memory = torch.cuda.is_available()
    x1 = torch.empty(
        (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=pin_memory
    )
    x2 = torch.empty(
        (batch_size, features2_num(), 9, 9), dtype=torch.float32, pin_memory=pin_memory
    )
    return x1, x2


def _step_actors(actors, x1, x2):
    """各 Actor を1ステップ進め、評価する局面の入力特徴量をバッファに書き込む"""
    features1, features2 = x1.numpy(), x2.numpy()
    for i, actor in enumerate(actors):
        actor.next()
        make_input_features(actor.board, features1[i], features2[i])


def _set_evaluations(actors, outputs):
    """推論結果を各 Actor に設定する

    outputs は (logits, value) か、残り手数ヘッドがあれば (logits, value, moves_left)
    """
    # GPU から CPU へはバッチ単位でまとめて転送する
    logits = outputs[0].float().cpu().numpy()
    values = outputs[1].float().cpu().numpy().reshape(-1)
    moves_left = None
    if len(outputs) == 3:
        moves_left = outputs[2].float().cpu().numpy().reshape(-1)

    for i, actor in enumerate(actors):
        actor.step.prior_logits = _mask_invalid_actions(
            logits[i], _get_invalid_actions(actor.board)
        )
        actor.step.value = values[i]
        actor.step.moves_left = None if moves_left is None else moves_left[i]


def write_training_data(
    queue: Queue,
    output_dir: str,
    num_positions: int,
    skip_max_moves: bool = False,
    stats_dict=None,
):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # The files are written with temporary names, and published when completed. The
    # position file is published last, since the data loader finds the files by it.
    paths = [
        os.path.join(output_dir, timestamp + ext)
        for ext in (POLICY_FILE_EXT, MOVES_LEFT_FILE_EXT, POSITION_FILE_EXT)
    ]
    policy_path, moves_left_path, position_path = (
        path + TEMPORARY_FILE_EXT for path in paths
    )

    nyugyoku_count = 0
    draw_count = 0
    max_moves_count = 0
    total_moves = 0
    total_games = 0
    # Games, positions and results of each source of the start position
    source_stats = {
        source: {"games": 0, "positions": 0, BLACK_WIN: 0, WHITE_WIN: 0, DRAW: 0}
        for source in SOURCES
    }

    with tqdm(total=num_positions, unit="pos") as pbar:
        while pbar.n < num_positions:
            training_data = queue.get()
            if training_data is None:
                break

            if (
                training_data.is_mate_in_3ply
                or training_data.is_nyugyoku
                or training_data.is_draw == REPETITION_WIN
            ):
                if training_data.turn == BLACK:
                    result = BLACK_WIN
                else:
                    result = WHITE_WIN
                if training_data.is_nyugyoku:
                    nyugyoku_count += 1
            elif training_data.is_game_over or training_data.is_draw == REPETITION_LOSE:
                if training_data.turn == BLACK:
                    result = WHITE_WIN
                else:
                    result = BLACK_WIN
            else:
                result = DRAW
                if training_data.is_draw == REPETITION_DRAW:
                    draw_count += 1
                else:  # max_moves
                    max_moves_count += 1
                    if skip_max_moves:
                        continue

            num_positions_in_game = len(training_data.policy_outputs)
            positions = np.empty(num_positions_in_game, dtype=dtypePosition)
            positions["hcp"] = training_data.hcps
            positions["result"] = result
            # Plies until the end of the game: T - t for the position s_t
            if training_data.is_max_moves:
                moves_left = np.full(
                    num_positions_in_game, MOVES_LEFT_INVALID, dtype=dtypeMovesLeft
                )
            else:
                moves_left = np.arange(
                    num_positions_in_game, 0, -1, dtype=dtypeMovesLeft
                )
            move_policies = []
            for i, policy_output in enumerate(training_data.policy_outputs):
                # 確率が0でない手だけを保存する(0の手は読み込み時に0で復元される)
                labels = np.flatnonzero(policy_output.action_weights)
                move_policy = np.empty(len(labels), dtype=dtypeMovePolicy)
                move_policy["label"] = labels
                move_policy["prob"] = policy_output.action_weights[labels]
                move_policies.append(move_policy)
                positions["num_moves"][i] = len(labels)

            # 局面ファイルは方策ファイルと残り手数ファイルを書いた後に追記する
            with open(policy_path, "ab") as f:
                np.concatenate(move_policies).tofile(f)
            with open(moves_left_path, "ab") as f:
                moves_left.tofile(f)
            with open(position_path, "ab") as f:
                positions.tofile(f)

            total_moves += len(training_data.policy_outputs)
            total_games += 1
            stats = source_stats[training_data.source]
            stats["games"] += 1
            stats["positions"] += len(training_data.policy_outputs)
            stats[result] += 1

            pbar.update(len(training_data.policy_outputs))
            pbar.set_postfix(
                {
                    "Games": total_games,
                    "AverageMoves": total_moves / total_games,
                    "Nyugyoku": nyugyoku_count,
                    "Draw": draw_count,
                    "MaxMoves": max_moves_count,
                }
            )

    # Publish the files (closed above) if a game is saved, the position file last
    if total_games > 0:
        for path in paths:
            os.replace(path + TEMPORARY_FILE_EXT, path)

    # Store final stats
    if stats_dict is not None:
        stats_dict["positions"] = total_moves
        stats_dict["games"] = total_games
        stats_dict["average_moves"] = (
            total_moves / total_games if total_games > 0 else 0
        )
        stats_dict["nyugyoku_count"] = nyugyoku_count
        stats_dict["draw_count"] = draw_count
        stats_dict["max_moves_count"] = max_moves_count
        for source, stats in source_stats.items():
            games = stats["games"]
            stats_dict[f"{source}/games"] = games
            stats_dict[f"{source}/positions"] = stats["positions"]
            stats_dict[f"{source}/average_moves"] = (
                stats["positions"] / games if games > 0 else 0
            )
            stats_dict[f"{source}/black_wins"] = stats[BLACK_WIN]
            stats_dict[f"{source}/white_wins"] = stats[WHITE_WIN]
            stats_dict[f"{source}/draws"] = stats[DRAW]


def _bounded_float_type(
    name,
    minimum,
    maximum=math.inf,
    *,
    include_minimum=True,
    include_maximum=True,
):
    """Returns an argparse type of a finite number in the range."""
    range_text = (
        ("[" if include_minimum else "(")
        + f"{minimum}, {maximum}"
        + ("]" if include_maximum and math.isfinite(maximum) else ")")
    )

    def parse(value):
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        above = number >= minimum if include_minimum else number > minimum
        below = number <= maximum if include_maximum else number < maximum
        if not (math.isfinite(number) and above and below):
            raise argparse.ArgumentTypeError(
                f"{name} must be a finite number in {range_text}: {value!r}"
            )
        return number

    return parse


def add_qtransform_arguments(parser):
    """Adds the arguments of the Q-value transformation of the search."""
    parser.add_argument(
        "--value_scale",
        # A negative value reverses the order of the Q-values, and NaN breaks the
        # search. 0 uses only the prior policy, for comparison.
        type=_bounded_float_type("value_scale", 0),
        default=0.1,
        help="c_scale of the Q-values in the search, 0 or more. The default is the "
        "same as mctx (the paper uses 0.1 on Atari and 1.0 on Go and chess).",
    )
    parser.add_argument(
        "--rescale_values",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale the Q-values to [0, 1] by their min and max at each node",
    )


def make_qtransform(
    value_scale: float,
    rescale_values: bool,
    *,
    moves_left_beta: float = 0.0,
    moves_left_tau: float = 20.0,
    moves_left_threshold: float = 0.6,
) -> base.QTransform:
    """Returns the Q-value transformation of the search with the settings.

    With moves_left_beta > 0, the bonus by the moves left is added to it (see
    qtransform_with_moves_left), which needs a model with the moves-left head.
    With 0, it is the same as without the moves-left head.
    """
    qtransform = functools.partial(
        qtransform_completed_by_mix_value,
        value_scale=value_scale,
        rescale_values=rescale_values,
    )
    if moves_left_beta == 0:
        return qtransform
    return functools.partial(
        qtransform_with_moves_left,
        qtransform=qtransform,
        beta=moves_left_beta,
        tau=moves_left_tau,
        threshold=moves_left_threshold,
    )


def make_qtransforms(args):
    """Returns the Q-value transformations of the search and of the training targets.

    The targets use the moves-left bonus only with --moves_left_target.
    """
    qtransform = make_qtransform(
        args.value_scale,
        args.rescale_values,
        moves_left_beta=args.moves_left_beta,
        moves_left_tau=args.moves_left_tau,
        moves_left_threshold=args.moves_left_threshold,
    )
    # The same object lets the policy reuse the values for the targets
    if args.moves_left_beta == 0 or args.moves_left_target:
        return qtransform, qtransform
    return qtransform, make_qtransform(args.value_scale, args.rescale_values)


def add_moves_left_search_arguments(parser):
    """Adds the arguments of the moves-left bonus of the search."""
    parser.add_argument(
        "--moves_left_beta",
        type=_bounded_float_type("moves_left_beta", 0),
        default=0.0,
        help="Maximum of the moves-left bonus in the search (logits), 0 or more. "
        "0 disables it. It needs a model with the moves-left head.",
    )
    parser.add_argument(
        "--moves_left_tau",
        type=_bounded_float_type("moves_left_tau", 0, include_minimum=False),
        default=20.0,
        help="Difference of the moves left (plies) for the maximum bonus",
    )
    parser.add_argument(
        "--moves_left_threshold",
        type=_bounded_float_type("moves_left_threshold", 0, 1, include_maximum=False),
        default=0.6,
        help="Advantage |2V - 1| from which the moves-left bonus is applied, in [0, 1)",
    )
    parser.add_argument(
        "--moves_left_target",
        action="store_true",
        help="Also use the moves-left bonus for the training targets",
    )


def model_has_moves_left(model_path):
    """Whether the TorchScript model has the moves-left head (by its settings)."""
    model_settings = read_model_settings(model_path)
    if model_settings is not None:
        return model_settings["moves_left"]
    return _has_moves_left_parameters(load_torchscript_state_dict(model_path))


def _check_compile(
    model_path, device, compile_options, network, use_nyugyoku_features
):
    """Checks the options to compile the model, and returns the network of the model.

    It is called before starting self-play (and the worker processes), so that an
    error does not stop only the workers. The network is read from the settings saved
    with the model if not given, and checked against them (e.g. the activation,
    which the parameters do not determine).

    Raises:
        ValueError: If the options or the network are invalid.
    """
    resolve_compile_options(device, **compile_options)
    if network is None:
        model_settings = read_model_settings(model_path)
        if model_settings is None:
            raise ValueError(
                f"The network of {model_path} is unknown, so give it to compile"
            )
        network = model_settings["network"]
    check_torchscript_model(
        model_path,
        network=network,
        use_nyugyoku_features=use_nyugyoku_features,
        moves_left=_has_moves_left_parameters(load_torchscript_state_dict(model_path)),
    )
    return network


def _has_moves_left_parameters(state_dict):
    """Whether the parameters of the network include the moves-left head."""
    return any(key.startswith(MOVES_LEFT_PREFIX) for key in state_dict)


def _load_model(model_path, device, compile_options, network):
    """Loads the TorchScript model for inference, or compiles it with torch.compile.

    torch.compile does not compile TorchScript, so the network given by its name
    (checked by _check_compile) is built again with the parameters of the TorchScript
    model.

    Args:
        compile_options: None to use the TorchScript model, or the arguments of
            compile_model(), e.g. {"backend": None, "mode": None}
    """
    if compile_options is None:
        return torch.jit.load(model_path, map_location=device).eval()

    state_dict = load_torchscript_state_dict(model_path)
    model = policy_value_network(
        network,
        input2_channels=features2_num(),
        moves_left=_has_moves_left_parameters(state_dict),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return compile_model(inference_wrapper(model).eval(), device, **compile_options)


def selfplay(
    model_path,
    batch_size,
    max_num_considered_actions,
    num_simulations,
    output_dir,
    num_positions,
    amp,
    skip_max_moves=False,
    debug=False,
    use_nyugyoku_features=False,
    compile_options=None,
    network=None,
    qtransform=qtransform_completed_by_mix_value,
    roots=None,
    roots_ratio=1.0,
    target_qtransform=None,
):
    set_nyugyoku_features(use_nyugyoku_features)
    if roots is not None:
        check_roots_file(roots)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compile_options is not None:
        network = _check_compile(
            model_path, device, compile_options, network, use_nyugyoku_features
        )
    model = _load_model(model_path, device, compile_options, network)
    start_positions = StartPositions(roots, roots_ratio)

    init_table(max_num_considered_actions, num_simulations)

    queue = Queue()
    writer_thread = threading.Thread(
        target=write_training_data,
        args=(queue, output_dir, num_positions, skip_max_moves),
        daemon=True,
    )
    writer_thread.start()

    actors = [
        Actor(
            max_num_considered_actions,
            num_simulations,
            queue,
            debug=debug,
            qtransform=qtransform,
            start_positions=start_positions,
            target_qtransform=target_qtransform,
        )
        for _ in range(batch_size)
    ]

    x1, x2 = _allocate_input_buffers(batch_size)

    try:
        while writer_thread.is_alive():
            _step_actors(actors, x1, x2)

            # Evaluate
            with torch.no_grad(), autocast(device.type, enabled=amp):
                # A model with the moves-left head also returns the moves left
                outputs = model(x1.to(device), x2.to(device))

            _set_evaluations(actors, outputs)
    except KeyboardInterrupt:
        queue.put(None)
        writer_thread.join()


def selfplay_worker_mp(
    lock,
    stop_event,
    model_path,
    batch_size,
    max_num_considered_actions,
    num_simulations,
    queue,
    amp,
    debug,
    use_nyugyoku_features,
    compile_options,
    network,
    qtransform,
    roots,
    roots_ratio,
    target_qtransform,
):
    """マルチプロセス用の自己対局ワーカー"""
    # 特徴量の設定はプロセス単位なので、ワーカーごとに設定する
    set_nyugyoku_features(use_nyugyoku_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with lock:
        model = _load_model(model_path, device, compile_options, network)

    init_table(max_num_considered_actions, num_simulations)
    # The file is opened in each process, which shares its pages with the others
    start_positions = StartPositions(roots, roots_ratio)

    actors = [
        Actor(
            max_num_considered_actions,
            num_simulations,
            queue,
            debug=debug,
            qtransform=qtransform,
            start_positions=start_positions,
            target_qtransform=target_qtransform,
        )
        for _ in range(batch_size)
    ]

    x1, x2 = _allocate_input_buffers(batch_size)

    while not stop_event.is_set():
        _step_actors(actors, x1, x2)

        # Evaluate with lock
        with lock, torch.no_grad(), autocast(device.type, enabled=amp):
            # A model with the moves-left head also returns the moves left
            outputs = model(x1.to(device), x2.to(device))

        _set_evaluations(actors, outputs)


def selfplay_multiprocess(
    model_path,
    batch_size,
    max_num_considered_actions,
    num_simulations,
    output_dir,
    num_positions,
    amp,
    num_processes,
    skip_max_moves=False,
    debug=False,
    use_nyugyoku_features=False,
    compile_options=None,
    network=None,
    qtransform=qtransform_completed_by_mix_value,
    roots=None,
    roots_ratio=1.0,
    target_qtransform=None,
):
    """マルチプロセスで自己対局を実行する"""
    import multiprocessing as mp

    # Check the options here, since an error in the workers does not stop the writer
    if roots is not None:
        check_roots_file(roots)
    if compile_options is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = _check_compile(
            model_path, device, compile_options, network, use_nyugyoku_features
        )

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # context has already been set

    manager = mp.Manager()
    queue = manager.Queue()
    stats_dict = manager.dict()
    lock = mp.Lock()
    stop_event = mp.Event()

    writer_process = mp.Process(
        target=write_training_data,
        args=(queue, output_dir, num_positions, skip_max_moves, stats_dict),
    )
    writer_process.start()

    processes = []
    for _ in range(num_processes):
        p = mp.Process(
            target=selfplay_worker_mp,
            args=(
                lock,
                stop_event,
                model_path,
                batch_size,
                max_num_considered_actions,
                num_simulations,
                queue,
                amp,
                debug,
                use_nyugyoku_features,
                compile_options,
                network,
                qtransform,
                roots,
                roots_ratio,
                target_qtransform,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    try:
        # Writerが終了したら、全局面生成完了
        while writer_process.is_alive():
            writer_process.join(timeout=1)
            # The workers run until stop_event is set, so an exited one has failed
            failed = [p for p in processes if not p.is_alive()]
            if failed and writer_process.is_alive():
                raise RuntimeError(
                    f"{len(failed)} self-play processes have exited unexpectedly "
                    f"(exit codes: {[p.exitcode for p in failed]})"
                )
    except KeyboardInterrupt:
        print("\nTerminating self-play processes.")
    finally:
        stop_event.set()  # 全プロセスに終了を通知

        for p in processes:
            p.join()
        if writer_process.is_alive():
            queue.put(None)
            writer_process.join()
        print("All processes terminated.")

    return dict(stats_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_considered_actions", type=int, default=16)
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument("--output_dir", default="training_data")
    parser.add_argument("--num_positions", type=int, default=1000000)
    parser.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="Number of self-play processes. If 0, run in single process mode.",
    )
    parser.add_argument(
        "--skip_max_moves",
        action="store_true",
        help="Skip training data when game ends due to maximum moves limit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for additional logging",
    )
    parser.add_argument(
        "--use_nyugyoku_features",
        action="store_true",
        help="Use nyugyoku features in the input",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model with torch.compile"
    )
    add_network_argument(
        parser,
        default=None,
        help="Network of the model to compile it. Defaults to the network in the "
        "settings saved with the model.",
    )
    add_compile_arguments(parser)
    add_qtransform_arguments(parser)
    add_moves_left_search_arguments(parser)
    add_start_positions_arguments(parser)
    args = parser.parse_args()
    if args.moves_left_beta > 0 and not model_has_moves_left(args.model_path):
        parser.error(
            f"--moves_left_beta needs a model with the moves-left head: "
            f"{args.model_path}"
        )
    qtransform, target_qtransform = make_qtransforms(args)
    if args.roots is not None:
        try:
            check_roots_file(args.roots)
        except ValueError as e:
            parser.error(str(e))
    compile_options = (
        {"backend": args.compile_backend, "mode": args.compile_mode}
        if args.compile
        else None
    )

    if args.num_processes > 0:
        selfplay_multiprocess(
            args.model_path,
            args.batch_size,
            args.max_num_considered_actions,
            args.num_simulations,
            args.output_dir,
            args.num_positions,
            args.amp,
            args.num_processes,
            args.skip_max_moves,
            debug=args.debug,
            use_nyugyoku_features=args.use_nyugyoku_features,
            compile_options=compile_options,
            network=args.network,
            qtransform=qtransform,
            target_qtransform=target_qtransform,
            roots=args.roots,
            roots_ratio=args.roots_ratio,
        )
    else:
        selfplay(
            args.model_path,
            args.batch_size,
            args.max_num_considered_actions,
            args.num_simulations,
            args.output_dir,
            args.num_positions,
            args.amp,
            args.skip_max_moves,
            debug=args.debug,
            use_nyugyoku_features=args.use_nyugyoku_features,
            compile_options=compile_options,
            network=args.network,
            qtransform=qtransform,
            target_qtransform=target_qtransform,
            roots=args.roots,
            roots_ratio=args.roots_ratio,
        )
