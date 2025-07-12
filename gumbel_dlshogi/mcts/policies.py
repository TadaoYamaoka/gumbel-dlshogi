from typing import Optional

import numpy as np
from cshogi import Board

from gumbel_dlshogi.mcts import action_selection, base, qtransforms, search, seq_halving
from gumbel_dlshogi.mcts.qtransforms import softmax


def gumbel_muzero_policy(
    board: Board,
    root: base.RootFnOutput,
    num_simulations: int,
    invalid_actions: Optional[np.ndarray] = None,
    max_num_considered_actions: int = 16,
    gumbel_scale: float = 1.0,
):
    """Runs Gumbel MuZero search and returns the `PolicyOutput`.

    This policy implements Full Gumbel MuZero from
    "Policy improvement by planning with Gumbel".
    https://openreview.net/forum?id=bERaNdoegnO

    At the root of the search tree, actions are selected by Sequential Halving
    with Gumbel. At non-root nodes (aka interior nodes), actions are selected by
    the Full Gumbel MuZero deterministic action selection.

    Args:
      root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
        `prior_logits` are from a policy network. The shapes are
        `([num_actions], [], [...])`, respectively.
      num_simulations: the number of simulations.
      invalid_actions: a mask with invalid actions. Invalid actions
        have ones, valid actions have zeros in the mask. Shape `[num_actions]`.
      max_num_considered_actions: the maximum number of actions expanded at the
        root node. A smaller number of actions will be expanded if the number of
        valid actions is smaller.
      gumbel_scale: scale for the Gumbel noise. Evalution on perfect-information
        games can use gumbel_scale=0.0.

    Returns:
      `PolicyOutput` containing the proposed action, action_weights and the used
      search tree.
    """
    # Masking invalid actions.
    root.prior_logits = _mask_invalid_actions(root.prior_logits, invalid_actions)

    # Generating Gumbel.
    gumbel = gumbel_scale * np.random.gumbel(size=root.prior_logits.shape)

    # Allocate all necessary storage.
    tree = search.instantiate_tree_from_root(
        root,
        num_simulations=num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel),
    )

    # Searching.
    yield from search.search(
        board,
        tree,
        root=root,
        max_num_considered_actions=max_num_considered_actions,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
    )
    summary = tree.summary()

    # Acting with the best action from the most visited actions.
    # The "best" action has the highest `gumbel + logits + q`.
    # Inside the minibatch, the considered_visit can be different on states with
    # a smaller number of valid actions.
    considered_visit = np.max(summary.visit_counts)
    # The completed_qvalues include imputed values for unvisited actions.
    completed_qvalues = qtransforms.qtransform_completed_by_mix_value(
        tree, tree.ROOT_INDEX
    )
    to_argmax = seq_halving.score_considered(
        considered_visit,
        gumbel,
        root.prior_logits,
        completed_qvalues,
        summary.visit_counts,
    )
    action = action_selection.masked_argmax(to_argmax, invalid_actions)

    # Producing action_weights usable to train the policy network.
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions
    )
    action_weights = softmax(completed_search_logits)
    return base.PolicyOutput(
        action=action, action_weights=action_weights, search_tree=tree
    )


def _mask_invalid_actions(logits, invalid_actions):
    """Returns logits with zero mass to invalid actions."""
    if invalid_actions is None:
        return logits
    logits = logits - np.max(logits)
    # At the end of an episode, all actions can be invalid. A softmax would then
    # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
    # a finite `min_logit` for the invalid actions.
    min_logit = np.finfo(logits.dtype).min
    return np.where(invalid_actions, min_logit, logits)
