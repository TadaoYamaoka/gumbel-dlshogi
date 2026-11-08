import numpy as np

from gumbel_dlshogi.mcts import base
from gumbel_dlshogi.mcts.tree import Tree


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def qtransform_completed_by_mix_value(
    tree: Tree,
    node_index: int,
    *,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rescale_values: bool = True,
    epsilon: float = 1e-8,
):
    """Returns completed qvalues.

    The missing Q-values of the unvisited actions are replaced by the
    mixed value, defined in Appendix D of
    "Policy improvement by planning with Gumbel":
    https://openreview.net/forum?id=bERaNdoegnO

    The Q-values are transformed by a linear transformation:
      `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

    Args:
      tree: _unbatched_ MCTS tree state.
      node_index: scalar index of the parent node.
      value_scale: scale for the Q-values (c_scale). The default is the same as
        mctx (the paper uses 0.1 on Atari and 1.0 on Go and chess).
      maxvisit_init: offset to the `max(visit_counts)` in the scaling factor
        (c_visit).
      rescale_values: if True, scale the qvalues by `1 / (max_q - min_q)`.
      epsilon: the minimum denominator when using `rescale_values`.

    Returns:
      Completed Q-values. Shape `[num_actions]`.

    Differences from mctx: `use_mixed_value` is omitted (always the mixed value,
    the default of mctx), and the prior probabilities are the softmax stored when
    the node is expanded (`tree.children_prior_probs`) instead of computing it here.
    """
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]

    # Computing the mixed value and producing completed_qvalues.
    raw_value = tree.raw_values[node_index]
    prior_probs = tree.children_prior_probs[node_index]
    value = _compute_mixed_value(
        raw_value, qvalues=qvalues, visit_counts=visit_counts, prior_probs=prior_probs
    )
    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value
    )

    # Scaling the Q-values.
    if rescale_values:
        completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
    maxvisit = np.max(visit_counts)
    visit_scale = maxvisit_init + maxvisit
    return visit_scale * value_scale * completed_qvalues


def _rescale_qvalues(qvalues, epsilon):
    """Rescales the given completed Q-values to be from the [0, 1] interval."""
    min_value = np.min(qvalues)
    max_value = np.max(qvalues)
    return (qvalues - min_value) / np.maximum(max_value - min_value, epsilon)


def _complete_qvalues(qvalues, *, visit_counts, value):
    """Returns completed Q-values, with the `value` for unvisited actions."""

    # The missing qvalues are replaced by the value.
    completed_qvalues = np.where(visit_counts > 0, qvalues, value)
    return completed_qvalues


def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
    """Interpolates the raw_value and weighted qvalues.

    Args:
      raw_value: an approximate value of the state. Shape `[]`.
      qvalues: Q-values for all actions. Shape `[num_actions]`. The unvisited
        actions have undefined Q-value.
      visit_counts: the visit counts for all actions. Shape `[num_actions]`.
      prior_probs: the action probabilities, produced by the policy network for
        each action. Shape `[num_actions]`.

    Returns:
      An estimator of the state value. Shape `[]`.
    """
    # Only the visited actions contribute, so compute over them alone. mctx masks
    # the unvisited ones instead, which gives the same value.
    visited = np.flatnonzero(visit_counts)
    if len(visited) == 0:
        return raw_value
    sum_visit_counts = np.sum(visit_counts[visited])
    # Ensuring non-nan weighted_q, even if the visited actions have zero
    # prior probability.
    visited_probs = np.maximum(np.finfo(prior_probs.dtype).tiny, prior_probs[visited])
    weighted_q = np.sum(visited_probs * qvalues[visited]) / np.sum(visited_probs)
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)


def qtransform_with_moves_left(
    tree: Tree,
    node_index: int,
    *,
    qtransform: base.QTransform,
    beta: float,
    tau: float,
    threshold: float,
):
    """Returns the transformed Q-values plus a bonus by the moves left.

    It prefers the shorter wins and the longer losses. The bonus is added after
    `qtransform` (after the rescaling and the scaling), in the units of the logits,
    so that its effect does not depend on the range of the Q-values of the node.
    All the values are from the side to move at the node:

      v = 2 V - 1 (V: the value of the node), s = sign(v)
      w = clip((|v| - threshold) / (1 - threshold), 0, 1)
      bonus(a) = beta * s * w * clip((M_node - 1 - M_child(a)) / tau, -1, 1)

    only for the visited actions on the same side as the node, s (2 Q(a) - 1) >
    threshold, and 0 for the others (so 0 for a draw with Q = 0.5). M_node - 1 is
    the expected moves left of the children (in plies). The bonus is at most beta,
    and it does not change the values of the tree. This is not in mctx.

    Args:
      qtransform: the Q-value transformation to add the bonus to.
      beta: the maximum of the bonus.
      tau: the difference of the moves left (plies) for the maximum bonus.
      threshold: the advantage |2 V - 1| from which the bonus is applied.
    """
    qvalues = qtransform(tree, node_index)
    centered_value = 2 * float(tree.node_values[node_index]) - 1
    if abs(centered_value) <= threshold:
        return qvalues

    direction = np.sign(centered_value)
    weight = min((abs(centered_value) - threshold) / (1 - threshold), 1.0)
    eligible = np.flatnonzero(
        (tree.children_visits[node_index] > 0)
        & (direction * (2 * tree.qvalues(node_index) - 1) > threshold)
    )
    if len(eligible) == 0:
        return qvalues

    # Read the children only for the eligible (visited) actions
    children = tree.children_index[node_index, eligible]
    children_moves_left = tree.node_moves_left[children]
    expected_moves_left = tree.node_moves_left[node_index] - 1
    difference = np.clip((expected_moves_left - children_moves_left) / tau, -1, 1)
    qvalues = qvalues.copy()
    qvalues[eligible] += beta * direction * weight * difference
    return qvalues
