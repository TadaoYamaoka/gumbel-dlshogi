from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from cshogi import REPETITION_DRAW, REPETITION_LOSE, REPETITION_WIN, Board

from gumbel_dlshogi.features import make_move_label
from gumbel_dlshogi.mcts import base
from gumbel_dlshogi.mcts.action_selection import (
    gumbel_muzero_interior_action_selection,
    gumbel_muzero_root_action_selection,
)
from gumbel_dlshogi.mcts.tree import Tree

MAX_MOVES = 512


def search(
    board: Board,
    tree: Tree,
    root: base.RootFnOutput,
    max_num_considered_actions: int,
    num_simulations: int,
    invalid_actions: Optional[np.ndarray] = None,
):
    """Performs a full search and returns sampled actions.

    In the shape descriptions, `B` denotes the batch dimension.

    Args:
      root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
        `prior_logits` are from a policy network. The shapes are
        `([num_actions], [], [...])`, respectively.
      num_simulations: the number of simulations.
      max_depth: maximum search tree depth allowed during simulation, defined as
        the number of edges from the root to a leaf node.
      invalid_actions: a mask with invalid actions at the root. In the
        mask, invalid actions have ones, and valid actions have zeros.
        Shape `[num_actions]`.
      extra_data: extra data passed to `tree.extra_data`. Shape `[...]`.

    Returns:
      `SearchResults` containing outcomes of the search, e.g. `visit_counts`
      `[num_actions]`.
    """

    # Do simulation, expansion, and backward steps.
    if invalid_actions is None:
        invalid_actions = np.zeros_like(root.prior_logits)

    for sim in range(num_simulations):
        parent_index, action = simulate(board, tree, max_num_considered_actions)
        # A node first expanded on simulation `i`, will have node index `i`.
        # Node 0 corresponds to the root node.
        next_node_index = tree.children_index[parent_index, action]
        next_node_index = np.where(
            next_node_index == Tree.UNVISITED, sim + 1, next_node_index
        )
        step = base.EvaluationStep()
        yield step
        expand(
            tree,
            step,
            parent_index,
            action,
            next_node_index,
        )
        backward(board, tree, next_node_index)


class _SimulationState(NamedTuple):
    """The state for the simulation while loop."""

    node_index: int
    action: int
    next_node_index: int
    depth: int
    is_continuing: bool


def simulate(
    board: Board, tree: Tree, max_num_considered_actions: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Traverses the tree until reaching an unvisited action or `max_depth`.

    Each simulation starts from the root and keeps selecting actions traversing
    the tree until a leaf or `max_depth` is reached.

    Args:
      rng_key: random number generator state, the key is consumed.
      tree: _unbatched_ MCTS tree state.
      action_selection_fn: function used to select an action during simulation.
      max_depth: maximum search tree depth allowed during simulation.

    Returns:
      `(parent_index, action)` tuple, where `parent_index` is the index of the
      node reached at the end of the simulation, and the `action` is the action to
      evaluate from the `parent_index`.
    """

    def body_fun(state):
        # Preparing the next simulation state.
        node_index = state.next_node_index
        if state.depth == 0:
            action = gumbel_muzero_root_action_selection(
                tree, node_index, max_num_considered_actions
            )
        else:
            action = gumbel_muzero_interior_action_selection(tree, node_index)
        next_node_index = tree.children_index[node_index, action]
        # The returned action will be visited.
        depth = state.depth + 1
        is_visited = next_node_index != Tree.UNVISITED
        is_continuing = is_visited

        move = next(
            (
                move
                for move in board.legal_moves
                if make_move_label(move, board.turn) == action
            ),
            None,
        )
        assert move is not None
        board.push(move)
        if (
            board.is_game_over()
            or board.is_nyugyoku()
            or board.is_draw() in (REPETITION_DRAW, REPETITION_WIN, REPETITION_LOSE)
            or board.move_number >= MAX_MOVES
        ):
            is_continuing = False

        return _SimulationState(
            node_index=node_index,
            action=action,
            next_node_index=next_node_index,
            depth=depth,
            is_continuing=is_continuing,
        )

    node_index = np.array(Tree.ROOT_INDEX, dtype=np.int32)
    depth = 0
    state = _SimulationState(
        node_index=tree.NO_PARENT,
        action=tree.NO_PARENT,
        next_node_index=node_index,
        depth=depth,
        is_continuing=True,
    )
    while state.is_continuing:
        state = body_fun(state)

    # Returning a node with a selected action.
    # The action can be already visited.
    return state.node_index, state.action


def expand(
    tree: Tree,
    step: base.EvaluationStep,
    parent_index: np.int32,
    action: np.int32,
    next_node_index: np.int32,
):
    """Create and evaluate child nodes from given nodes and unvisited actions.

    Args:
      tree: the MCTS tree state to update.
      parent_index: the index of the parent node, from which the action will be
        expanded.
      action: the action to expand.
      next_node_index: the index of the newly expanded node. This can be the index
        of an existing node, if `max_depth` is reached.
    """

    # Evaluate and create a new node.
    update_tree_node(tree, next_node_index, step.prior_logits, step.value)

    # Return updated tree topology.
    tree.children_index[parent_index, action] = next_node_index
    tree.parents[next_node_index] = parent_index
    tree.action_from_parent[next_node_index] = action


def backward(board: Board, tree: Tree, leaf_index: np.int32):
    """Goes up and updates the tree until all nodes reached the root.

    Args:
      tree: the MCTS tree state to update, without the batch size.
      leaf_index: the node index from which to do the backward.
    """

    def cond_fun(loop_state):
        _, index = loop_state
        return index != Tree.ROOT_INDEX

    def body_fun(loop_state):
        # Here we update the value of our parent, so we start by reversing.
        leaf_value, index = loop_state
        parent = tree.parents[index]
        count = tree.node_visits[parent]
        action = tree.action_from_parent[index]
        parent_value = (tree.node_values[parent] * count + leaf_value) / (count + 1.0)
        children_value = tree.node_values[index]
        children_count = tree.children_visits[parent, action] + 1

        tree.node_values[parent] = parent_value
        tree.node_visits[parent] = count + 1
        tree.children_values[parent, action] = children_value
        tree.children_visits[parent, action] = children_count

        board.pop()

        return leaf_value, parent

    leaf_index = np.asarray(leaf_index, dtype=np.int32)
    loop_state = (tree.node_values[leaf_index], leaf_index)
    while cond_fun(loop_state):
        loop_state = body_fun(loop_state)


def update_tree_node(
    tree: Tree,
    node_index: np.int32,
    prior_logits: np.ndarray,
    value: np.float32,
):
    """Updates the tree at node index.

    Args:
      tree: `Tree` to whose node is to be updated.
      node_index: the index of the expanded node.
      prior_logits: the prior logits to fill in for the new node, of shape
        `[num_actions]`.
      value: the value to fill in for the new node.

    Returns:
      The new tree with updated nodes.
    """

    new_visit = tree.node_visits[node_index] + 1

    tree.children_prior_logits[node_index] = prior_logits
    tree.raw_values[node_index] = value
    tree.node_values[node_index] = value
    tree.node_visits[node_index] = new_visit


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: np.ndarray,
    extra_data: Any,
) -> Tree:
    """Initializes tree state at search root."""
    num_actions = root.prior_logits.shape[0]
    num_nodes = num_simulations + 1

    # Create a new empty tree state and fill its root.
    tree = Tree(
        node_visits=np.zeros(num_nodes, dtype=np.int32),
        raw_values=np.zeros(num_nodes, dtype=np.float32),
        node_values=np.zeros(num_nodes, dtype=np.float32),
        parents=np.full(num_nodes, Tree.NO_PARENT, dtype=np.int32),
        action_from_parent=np.full(num_nodes, Tree.NO_PARENT, dtype=np.int32),
        children_index=np.full(
            (num_nodes, num_actions), Tree.UNVISITED, dtype=np.int32
        ),
        children_prior_logits=np.zeros(
            (num_nodes, num_actions), dtype=root.prior_logits.dtype
        ),
        children_values=np.zeros((num_nodes, num_actions), dtype=np.float32),
        children_visits=np.zeros((num_nodes, num_actions), dtype=np.int32),
        root_invalid_actions=root_invalid_actions,
        extra_data=extra_data,
    )

    root_index = Tree.ROOT_INDEX
    update_tree_node(tree, root_index, root.prior_logits, root.value)
    return tree
