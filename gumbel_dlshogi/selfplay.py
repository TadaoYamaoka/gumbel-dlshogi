import os
import threading
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Literal

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
    WHITE,
    WHITE_WIN,
    Board,
    HuffmanCodedPos,
    dtypeHcp,
)

from gumbel_dlshogi.features import (
    FEATURES_NUM,
    MOVE_LABELS_NUM,
    make_input_features,
    make_move_label,
)
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
    is_game_over: bool
    is_nyugyoku: bool
    is_draw: int
    is_max_moves: bool


dtypeTrainingData = np.dtype(
    [
        ("hcp", dtypeHcp),
        ("result", np.uint8),
        ("policy", np.dtype((np.float32, MOVE_LABELS_NUM))),
    ]
)


class Actor:
    def __init__(self, max_num_considered_actions, num_simulations, queue: Queue):
        self.board = Board()
        self.max_num_considered_actions = max_num_considered_actions
        self.num_simulations = num_simulations
        self.step = None
        self.generator = None
        self.queue = queue

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
            invalid_actions = np.ones(MOVE_LABELS_NUM, dtype=np.int32)
            invalid_actions[
                [
                    make_move_label(move, self.board.turn)
                    for move in self.board.legal_moves
                ]
            ] = 0
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
            if (
                is_game_over := self.board.is_game_over()
                or (is_nyugyoku := self.board.is_nyugyoku())
                or (is_draw := self.board.is_draw())
                in (REPETITION_DRAW, REPETITION_WIN, REPETITION_LOSE)
                or (is_max_moves := self.board.move_number >= MAX_MOVES)
            ):
                self.queue.put(
                    TrainingData(
                        hcps=self.hcps[: len(self.policy_outputs)].copy(),
                        policy_outputs=self.policy_outputs,
                        turn=self.board.turn,
                        is_game_over=is_game_over,
                        is_nyugyoku=is_nyugyoku,
                        is_draw=is_draw,
                        is_max_moves=is_max_moves,
                    )
                )

                self.board.reset()
                self.policy_outputs = []


def write_training_data(queue: Queue, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}.data"
    filepath = os.path.join(output_dir, filename)

    while True:
        training_data = queue.get()
        if training_data is None:
            break

        if training_data.is_game_over or training_data.is_draw == REPETITION_LOSE:
            if training_data.turn == BLACK:
                result = WHITE_WIN
            else:
                result = BLACK_WIN
        elif training_data.is_nyugyoku or training_data.is_draw == REPETITION_WIN:
            if training_data.turn == BLACK:
                result = BLACK_WIN
            else:
                result = WHITE_WIN
        elif training_data.is_draw == REPETITION_DRAW:
            result = DRAW

        data = np.empty(len(training_data.policy_outputs), dtype=dtypeTrainingData)
        for i, (hcp, policy_output) in enumerate(
            zip(training_data.hcps, training_data.policy_outputs)
        ):
            data[i]["hcp"] = hcp
            data[i]["result"] = result
            data[i]["policy"] = policy_output.action_weights

        with open(filepath, "ab") as f:
            data.tofile(f)


def selfplay(
    model_path, batch_size, max_num_considered_actions, num_simulations, output_dir
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()

    init_table(max_num_considered_actions, num_simulations)

    queue = Queue()
    writer_thread = threading.Thread(
        target=write_training_data, args=(queue, output_dir), daemon=True
    )
    writer_thread.start()

    actors = [
        Actor(max_num_considered_actions, num_simulations, queue)
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
        while True:
            for i, actor in enumerate(actors):
                actor.next()
                make_input_features(actor.board, input_features[i])

            # Evaluate
            with torch.no_grad():
                logits, value = model(torch_features.to(device))

            for i, actor in enumerate(actors):
                actor.step.prior_logits = logits[i].cpu().numpy()
                actor.step.value = value[i].cpu().numpy()[0]

                invalid_actions = np.ones(MOVE_LABELS_NUM, dtype=np.int32)
                invalid_actions[
                    [
                        make_move_label(move, actor.board.turn)
                        for move in actor.board.legal_moves
                    ]
                ] = 0
                actor.step.prior_logits = _mask_invalid_actions(
                    actor.step.prior_logits, invalid_actions
                )
    except KeyboardInterrupt:
        queue.put(None)
        writer_thread.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_considered_actions", type=int, default=16)
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument("--output_dir", default="training_data")
    args = parser.parse_args()

    selfplay(
        args.model_path,
        args.batch_size,
        args.max_num_considered_actions,
        args.num_simulations,
        args.output_dir,
    )
