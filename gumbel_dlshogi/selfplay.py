import numpy as np
import torch
from cshogi import REPETITION_DRAW, REPETITION_LOSE, REPETITION_WIN, Board

from gumbel_dlshogi.features import (
    FEATURES_NUM,
    MOVE_LABELS_NUM,
    make_input_features,
    make_move_label,
)
from gumbel_dlshogi.mcts.action_selection import init_table
from gumbel_dlshogi.mcts.base import EvaluationStep, RootFnOutput
from gumbel_dlshogi.mcts.policies import _mask_invalid_actions, gumbel_muzero_policy


class Actor:
    def __init__(self, max_num_considered_actions, num_simulations):
        self.board = Board()
        self.max_num_considered_actions = max_num_considered_actions
        self.num_simulations = num_simulations
        self.step = None
        self.generator = None

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

            if (
                self.board.is_game_over()
                or self.board.is_nyugyoku()
                or self.board.is_draw()
                in (REPETITION_DRAW, REPETITION_WIN, REPETITION_LOSE)
            ):
                self.board.reset()


def selfplay(model_path, batch_size, max_num_considered_actions, num_simulations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()

    init_table(max_num_considered_actions, num_simulations)

    actors = [
        Actor(max_num_considered_actions, num_simulations) for _ in range(batch_size)
    ]

    torch_features = torch.empty(
        (batch_size, FEATURES_NUM, 9, 9),
        dtype=torch.float32,
        pin_memory=True,
        requires_grad=False,
    )
    input_features = torch_features.numpy()

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_considered_actions", type=int, default=16)
    parser.add_argument("--num_simulations", type=int, default=32)
    args = parser.parse_args()

    selfplay(
        args.model_path,
        args.batch_size,
        args.max_num_considered_actions,
        args.num_simulations,
    )
