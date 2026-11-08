from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from gumbel_dlshogi.mcts import tree


@dataclass
class EvaluationStep:
    """A position to evaluate, yielded by the search (mctx uses `recurrent_fn`).

    The caller sets the outputs of the network for the position:
    prior_logits: `[num_actions]` the logits produced by a policy network, with
      the illegal moves masked. None for a terminal position.
    value: `[]` the win rate of the side to move.
    """

    prior_logits: Optional[np.ndarray] = None
    value: Optional[np.float32] = None


@dataclass
class RootFnOutput:
    """The output of a representation network.

    prior_logits: `[num_actions]` the logits produced by a policy network.
    value: `[]` an approximate value of the current state.
    """

    prior_logits: np.ndarray
    value: np.float32


@dataclass
class PolicyOutput:
    """The output of a policy.

    action: `[]` the proposed action.
    action_weights: `[num_actions]` the targets used to train a policy network.
      The action weights sum to one. Usually, the policy network is trained by
      cross-entropy:
      `cross_entropy(labels=stop_gradient(action_weights), logits=prior_logits)`.
    """

    action: np.int32
    action_weights: np.ndarray


QTransform = Callable[[tree.Tree, int], np.ndarray]
