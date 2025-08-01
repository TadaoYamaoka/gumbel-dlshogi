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
from pydlshogi2.features import (
    FEATURES_NUM,
    MOVE_LABELS_NUM,
    make_input_features,
    make_move_label,
)
from torch.cuda.amp import autocast
from tqdm import tqdm

from gumbel_dlshogi.common import dtypeTrainingData
from gumbel_dlshogi.mcts import base
from gumbel_dlshogi.mcts.action_selection import init_table
from gumbel_dlshogi.mcts.base import EvaluationStep, RootFnOutput
from gumbel_dlshogi.mcts.policies import _mask_invalid_actions, gumbel_muzero_policy
from gumbel_dlshogi.mcts.search import MAX_MOVES


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
        self, max_num_considered_actions, num_simulations, queue: Queue, debug=False
    ):
        self.board = Board()
        self.max_num_considered_actions = max_num_considered_actions
        self.num_simulations = num_simulations
        self.step = None
        self.generator = None
        self.queue = queue
        self.debug = debug

        self.hcps = np.empty(MAX_MOVES, dtype=HuffmanCodedPos)
        self.policy_outputs = []

    def next(self):
        if self.step is None:
            self.step = EvaluationStep()
            return

        if self.generator is None:
            self.root = RootFnOutput(
                prior_logits=self.step.prior_logits, value=self.step.value
            )
            invalid_actions = _get_invalid_actions(self.board)
            self.generator = gumbel_muzero_policy(
                self.board,
                self.root,
                self.num_simulations,
                invalid_actions,
                self.max_num_considered_actions,
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
                    print(
                        f"startpos moves {' '.join(move_to_usi(move) for move in self.board.history)}"
                    )
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
                    )
                )

                self.board.reset()
                self.policy_outputs = []


def write_training_data(
    queue: Queue,
    output_dir: str,
    num_positions: int,
    skip_max_moves: bool = False,
    stats_dict=None,
):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}.data"
    filepath = os.path.join(output_dir, filename)

    nyugyoku_count = 0
    draw_count = 0
    max_moves_count = 0
    total_moves = 0
    total_games = 0

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

            data = np.empty(len(training_data.policy_outputs), dtype=dtypeTrainingData)
            for i, (hcp, policy_output) in enumerate(
                zip(training_data.hcps, training_data.policy_outputs)
            ):
                data[i]["hcp"] = hcp
                data[i]["policy"] = policy_output.action_weights
                data[i]["result"] = result

            with open(filepath, "ab") as f:
                data.tofile(f)

            total_moves += len(training_data.policy_outputs)
            total_games += 1

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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()

    init_table(max_num_considered_actions, num_simulations)

    queue = Queue()
    writer_thread = threading.Thread(
        target=write_training_data,
        args=(queue, output_dir, num_positions, skip_max_moves),
        daemon=True,
    )
    writer_thread.start()

    actors = [
        Actor(max_num_considered_actions, num_simulations, queue, debug)
        for _ in range(batch_size)
    ]

    torch_features = torch.empty(
        (batch_size, FEATURES_NUM, 9, 9),
        dtype=torch.float32,
        pin_memory=True,
        requires_grad=False,
    )
    input_features = torch_features.numpy()

    try:
        while writer_thread.is_alive():
            for i, actor in enumerate(actors):
                actor.next()
                make_input_features(actor.board, input_features[i])

            # Evaluate
            with autocast(enabled=amp):
                with torch.no_grad():
                    logits, value = model(torch_features.to(device))

            for i, actor in enumerate(actors):
                actor.step.prior_logits = logits[i].cpu().numpy()
                actor.step.value = value[i].cpu().numpy()[0]

                invalid_actions = _get_invalid_actions(actor.board)
                actor.step.prior_logits = _mask_invalid_actions(
                    actor.step.prior_logits, invalid_actions
                )
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
):
    """マルチプロセス用の自己対局ワーカー"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path)
    with lock:
        model.to(device)
    model.eval()

    init_table(max_num_considered_actions, num_simulations)

    actors = [
        Actor(max_num_considered_actions, num_simulations, queue, debug)
        for _ in range(batch_size)
    ]

    torch_features = torch.empty(
        (batch_size, FEATURES_NUM, 9, 9),
        dtype=torch.float32,
        pin_memory=True,
        requires_grad=False,
    )
    input_features = torch_features.numpy()

    while not stop_event.is_set():
        for i, actor in enumerate(actors):
            actor.next()
            make_input_features(actor.board, input_features[i])

        # Evaluate with lock
        with lock:
            with autocast(enabled=amp):
                with torch.no_grad():
                    logits, value = model(torch_features.to(device))

        for i, actor in enumerate(actors):
            actor.step.prior_logits = logits[i].cpu().numpy()
            actor.step.value = value[i].cpu().numpy()[0]

            invalid_actions = _get_invalid_actions(actor.board)
            actor.step.prior_logits = _mask_invalid_actions(
                actor.step.prior_logits, invalid_actions
            )


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
):
    """マルチプロセスで自己対局を実行する"""
    import multiprocessing as mp

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
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    try:
        writer_process.join()
        # Writerが終了したら、全局面生成完了
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
    import argparse

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
    args = parser.parse_args()

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
        )
