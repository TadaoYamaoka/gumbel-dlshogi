from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EvaluationStep:
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

    action: `[B]` the proposed action.
    action_weights: `[B, num_actions]` the targets used to train a policy network.
      The action weights sum to one. Usually, the policy network is trained by
      cross-entropy:
      `cross_entropy(labels=stop_gradient(action_weights), logits=prior_logits)`.
    """

    action: np.int32
    action_weights: np.ndarray
