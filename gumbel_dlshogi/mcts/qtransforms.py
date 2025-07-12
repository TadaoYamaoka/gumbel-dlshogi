import numpy as np

from gumbel_dlshogi.mcts.tree import Tree


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def qtransform_completed_by_mix_value(
    tree: Tree,
    node_index: int,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    epsilon: float = 1e-8,
):
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]

    # Computing the mixed value and producing completed_qvalues.
    raw_value = tree.raw_values[node_index]
    prior_probs = softmax(tree.children_prior_logits[node_index])
    value = _compute_mixed_value(
        raw_value, qvalues=qvalues, visit_counts=visit_counts, prior_probs=prior_probs
    )
    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value
    )

    # Scaling the Q-values.
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
    sum_visit_counts = np.sum(visit_counts)
    # Ensuring non-nan weighted_q, even if the visited actions have zero
    # prior probability.
    prior_probs = np.maximum(np.finfo(prior_probs.dtype).tiny, prior_probs)
    # Summing the probabilities of the visited actions.
    sum_probs = np.sum(np.where(visit_counts > 0, prior_probs, 0.0))
    weighted_q = np.sum(
        np.where(
            visit_counts > 0,
            prior_probs * qvalues / np.where(visit_counts > 0, sum_probs, 1.0),
            0.0,
        )
    )
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)
