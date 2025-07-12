from dataclasses import dataclass
from typing import Optional

import numpy as np

from gumbel_dlshogi.mcts import qtransforms, seq_halving
from gumbel_dlshogi.mcts.qtransforms import softmax
from gumbel_dlshogi.mcts.tree import Tree


def init_table(max_num_considered_actions: int, num_simulations: int):
    global table
    table = np.array(
        seq_halving.get_table_of_considered_visits(
            max_num_considered_actions, num_simulations
        )
    )


@dataclass
class GumbelMuZeroExtraData:
    """Extra data for Gumbel MuZero search."""

    root_gumbel: np.ndarray


def gumbel_muzero_root_action_selection(
    tree: Tree, node_index: int, max_num_considered_actions: int
):
    visit_counts = tree.children_visits[node_index]
    prior_logits = tree.children_prior_logits[node_index]
    completed_qvalues = qtransforms.qtransform_completed_by_mix_value(tree, node_index)

    num_valid_actions = np.sum(1 - tree.root_invalid_actions).astype(np.int32)
    num_considered = np.minimum(max_num_considered_actions, num_valid_actions)
    # At the root, the simulation_index is equal to the sum of visit counts.
    simulation_index = np.sum(visit_counts, -1)
    considered_visit = table[num_considered, simulation_index]
    gumbel = tree.extra_data.root_gumbel
    to_argmax = seq_halving.score_considered(
        considered_visit, gumbel, prior_logits, completed_qvalues, visit_counts
    )

    # Masking the invalid actions at the root.
    return masked_argmax(to_argmax, tree.root_invalid_actions)


def gumbel_muzero_interior_action_selection(
    tree: Tree,
    node_index: int,
) -> np.ndarray:
    """Selects the action with a deterministic action selection.

    The action is selected based on the visit counts to produce visitation
    frequencies similar to softmax(prior_logits + qvalues).

    Args:
        rng_key: random number generator state.
        tree: _unbatched_ MCTS tree state.
        node_index: scalar index of the node from which to take an action.
        depth: the scalar depth of the current node. The root has depth zero.
        qtransform: function to obtain completed Q-values for a node.

    Returns:
        action: the action selected from the given node.
    """
    visit_counts = tree.children_visits[node_index]
    prior_logits = tree.children_prior_logits[node_index]
    completed_qvalues = qtransforms.qtransform_completed_by_mix_value(tree, node_index)

    # The `prior_logits + completed_qvalues` provide an improved policy,
    # because the missing qvalues are replaced by v_{prior_logits}(node).
    to_argmax = _prepare_argmax_input(
        probs=softmax(prior_logits + completed_qvalues),
        visit_counts=visit_counts,
    )

    return np.argmax(to_argmax).astype(np.int32)


def masked_argmax(
    to_argmax: np.ndarray, invalid_actions: Optional[np.ndarray]
) -> np.ndarray:
    """Returns a valid action with the highest `to_argmax`."""
    if invalid_actions is not None:
        # The usage of the -inf inside the argmax does not lead to NaN.
        # Do not use -inf inside softmax, logsoftmax or cross-entropy.
        to_argmax = np.where(invalid_actions, -np.inf, to_argmax)
    # If all actions are invalid, the argmax returns action 0.
    return np.argmax(to_argmax).astype(np.int32)


def _prepare_argmax_input(probs, visit_counts):
    """Prepares the input for the deterministic selection.

    When calling argmax(_prepare_argmax_input(...)) multiple times
    with updated visit_counts, the produced visitation frequencies will
    approximate the probs.

    For the derivation, see Section 5 "Planning at non-root nodes" in
    "Policy improvement by planning with Gumbel":
    https://openreview.net/forum?id=bERaNdoegnO

    Args:
        probs: a policy or an improved policy. Shape `[num_actions]`.
        visit_counts: the existing visit counts. Shape `[num_actions]`.

    Returns:
        The input to an argmax. Shape `[num_actions]`.
    """
    to_argmax = probs - visit_counts / (1 + np.sum(visit_counts))
    return to_argmax
