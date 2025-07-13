from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass
class Tree:
    """State of a search tree.

    The `Tree` dataclass is used to hold and inspect search data for a batch of
    inputs. In the fields below `B` denotes the batch dimension, `N` represents
    the number of nodes in the tree, and `num_actions` is the number of discrete
    actions.

    node_visits: `[N]` the visit counts for each node.
    raw_values: `[N]` the raw value for each node.
    node_values: `[N]` the cumulative search value for each node.
    parents: `[N]` the node index for the parents for each node.
    action_from_parent: `[N]` action to take from the parent to reach each
      node.
    children_index: `[N, num_actions]` the node index of the children for each
      action.
    children_prior_logits: `[N, num_actions]` the action prior logits of each
      node.
    children_visits: `[N, num_actions]` the visit counts for children for
      each action.
    children_values: `[N, num_actions]` the value of the next node after the
      action.
    root_invalid_actions: `[num_actions]` a mask with invalid actions at the
      root. In the mask, invalid actions have ones, and valid actions have zeros.
    extra_data: `[...]` extra data passed to the search.
    """

    node_visits: np.ndarray  # [N]
    raw_values: np.ndarray  # [N]
    node_values: np.ndarray  # [N]
    parents: np.ndarray  # [N]
    action_from_parent: np.ndarray  # [N]
    children_index: np.ndarray  # [N, num_actions]
    children_prior_logits: np.ndarray  # [N, num_actions]
    children_visits: np.ndarray  # [N, num_actions]
    children_values: np.ndarray  # [N, num_actions]
    root_invalid_actions: np.ndarray  # [num_actions]
    extra_data: Any

    # The following attributes are class variables (and should not be set on
    # Tree instances).
    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self):
        return self.children_index.shape[-1]

    @property
    def num_simulations(self):
        return self.node_visits.shape[-1] - 1

    def qvalues(self, indices):
        """Compute q-values for any node indices in the tree."""

        return self.children_values[indices]

    def summary(self) -> SearchSummary:
        """Extract summary statistics for the root node."""
        # Get state and action values for the root nodes.
        value = self.node_values[Tree.ROOT_INDEX]
        qvalue = self.qvalues(Tree.ROOT_INDEX)
        # Extract visit counts and induced probabilities for the root nodes.
        visit_counts = self.children_visits[Tree.ROOT_INDEX].astype(value.dtype)
        total_counts = np.sum(visit_counts, axis=-1, keepdims=True)
        visit_probs = visit_counts / np.maximum(total_counts, 1)
        visit_probs = np.where(total_counts > 0, visit_probs, 1 / self.num_actions)
        # Return relevant stats.
        return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
            visit_counts=visit_counts,
            visit_probs=visit_probs,
            value=value,
            qvalue=qvalue,
        )


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@dataclass
class SearchSummary:
    """Stats from MCTS search."""

    visit_counts: np.ndarray
    visit_probs: np.ndarray
    value: np.float32
    qvalue: np.float32
