from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
from cshogi import REPETITION_DRAW, REPETITION_LOSE, REPETITION_WIN, Board

from gumbel_dlshogi.features import make_move_label
from gumbel_dlshogi.mcts import base
from gumbel_dlshogi.mcts.action_selection import (
    gumbel_muzero_interior_action_selection,
    gumbel_muzero_root_action_selection,
)
from gumbel_dlshogi.mcts.qtransforms import qtransform_completed_by_mix_value, softmax
from gumbel_dlshogi.mcts.tree import Tree

MAX_MOVES = 512


def search(
    board: Board,
    tree: Tree,
    max_num_considered_actions: int,
    num_simulations: int,
    qtransform: base.QTransform = qtransform_completed_by_mix_value,
):
    """Performs a full search, updating the tree in place.

    Unlike mctx, which calls `recurrent_fn` to evaluate the leaves, this is a
    generator: it yields an `EvaluationStep` for each leaf to evaluate while
    `board` is at the leaf, and the caller sets its `prior_logits` and `value`
    before resuming. Terminal positions are not yielded; they are evaluated by
    the game result (`get_terminal_value`). There is no `max_depth`: a simulation
    continues until an unvisited action or a terminal position.

    mctx creates the tree from `root` and `invalid_actions` inside the search.
    Here the tree is created by the caller (`instantiate_tree_from_root`), which
    stores the mask of the invalid actions at the root (`root_invalid_actions`).

    Args:
      board: the position of the root. It is moved along the simulations and is
        back at the root after the search.
      tree: the tree created by `instantiate_tree_from_root`.
      max_num_considered_actions: the maximum number of actions considered by
        Sequential Halving at the root.
      num_simulations: the number of simulations.
      qtransform: function to obtain the completed and scaled Q-values for a node.
    """

    # Do simulation, expansion, and backward steps.
    for sim in range(num_simulations):
        parent_index, action, terminal_value = simulate(
            board, tree, max_num_considered_actions, qtransform
        )
        # A node first expanded on simulation `i`, will have node index `i`.
        # Node 0 corresponds to the root node.
        next_node_index = tree.children_index[parent_index, action]
        next_node_index = np.where(
            next_node_index == Tree.UNVISITED, sim + 1, next_node_index
        )
        if terminal_value is None:
            step = base.EvaluationStep()
            yield step
        else:
            # Terminal nodes use the game result instead of the network evaluation.
            # No action is selected from them, so no prior logits are needed. The
            # game ends there, so no moves are left.
            step = base.EvaluationStep(value=terminal_value, moves_left=np.float32(0))
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
    terminal_value: Optional[np.float32]


def get_terminal_value(board: Board) -> Optional[np.float32]:
    """Returns the game result from the side to move, or None if not terminal."""
    if board.is_game_over():
        return np.float32(0.0)
    if board.is_nyugyoku():
        return np.float32(1.0)
    draw = board.is_draw()
    if draw == REPETITION_DRAW:
        return np.float32(0.5)
    if draw == REPETITION_WIN:
        return np.float32(1.0)
    if draw == REPETITION_LOSE:
        return np.float32(0.0)
    if board.move_number >= MAX_MOVES:
        return np.float32(0.5)
    if board.mate_move(3) != 0:
        return np.float32(1.0)
    return None


def simulate(
    board: Board,
    tree: Tree,
    max_num_considered_actions: int,
    qtransform: base.QTransform = qtransform_completed_by_mix_value,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.float32]]:
    """Traverses the tree until reaching an unvisited action or a terminal position.

    Each simulation starts from the root and keeps selecting actions traversing
    the tree, moving `board` along them. The actions are selected by
    `gumbel_muzero_root_action_selection` at the root and by
    `gumbel_muzero_interior_action_selection` at the other nodes.

    Args:
      board: the position of the root, moved to the reached position.
      tree: _unbatched_ MCTS tree state.
      max_num_considered_actions: the maximum number of actions considered by
        Sequential Halving at the root.
      qtransform: function to obtain the completed and scaled Q-values for a node.

    Returns:
      `(parent_index, action, terminal_value)` tuple, where `parent_index` is the
      index of the parent of the final selected action, the `action` is the
      action to evaluate from the `parent_index` (`board` is at the position after
      it), and `terminal_value` is the game result from the side to move if that
      position is terminal, else None.
    """

    def body_fun(state):
        # Preparing the next simulation state.
        node_index = state.next_node_index
        if state.depth == 0:
            action = gumbel_muzero_root_action_selection(
                tree,
                node_index,
                max_num_considered_actions=max_num_considered_actions,
                qtransform=qtransform,
            )
        else:
            action = gumbel_muzero_interior_action_selection(
                tree, node_index, qtransform=qtransform
            )
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
        terminal_value = get_terminal_value(board)
        if terminal_value is not None:
            is_continuing = False

        return _SimulationState(
            node_index=node_index,
            action=action,
            next_node_index=next_node_index,
            depth=depth,
            is_continuing=is_continuing,
            terminal_value=terminal_value,
        )

    node_index = np.array(Tree.ROOT_INDEX, dtype=np.int32)
    depth = 0
    state = _SimulationState(
        node_index=tree.NO_PARENT,
        action=tree.NO_PARENT,
        next_node_index=node_index,
        depth=depth,
        is_continuing=True,
        terminal_value=None,
    )
    while state.is_continuing:
        state = body_fun(state)

    # Returning a node with a selected action.
    # The action can be already visited.
    return state.node_index, state.action, state.terminal_value


def expand(
    tree: Tree,
    step: base.EvaluationStep,
    parent_index: np.int32,
    action: np.int32,
    next_node_index: np.int32,
):
    """Creates the child node with the evaluation, or updates the reached node.

    Args:
      tree: the MCTS tree state to update.
      step: the evaluation of the reached position: the network outputs, or the
        game result (without `prior_logits`) for a terminal position.
      parent_index: the index of the parent node, from which the action will be
        expanded.
      action: the action to expand.
      next_node_index: the index of the newly expanded node. This can be the index
        of an existing node, if the reached position is terminal (terminal nodes
        are not traversed, so they are reached again).
    """

    # Evaluate and create a new node.
    update_tree_node(
        tree, next_node_index, step.prior_logits, step.value, step.moves_left
    )

    # Return updated tree topology.
    tree.children_index[parent_index, action] = next_node_index
    tree.parents[next_node_index] = parent_index
    tree.action_from_parent[next_node_index] = action


def backward(board: Board, tree: Tree, leaf_index: np.int32):
    """Goes up and updates the tree until all nodes reached the root.

    The values are win rates of the side to move in [0, 1], so a value is turned
    to the parent's side by `1 - value` (mctx uses the discount -1 for the values
    in [-1, 1] of two-player games). The moves left are averaged in the same way,
    adding a ply for each level instead of turning them.

    Args:
      board: the reached position, moved back to the root.
      tree: the MCTS tree state to update, without the batch size.
      leaf_index: the node index from which to do the backward.
    """

    def cond_fun(loop_state):
        _, _, index = loop_state
        return index != Tree.ROOT_INDEX

    def body_fun(loop_state):
        # Here we update the value of our parent, so we start by reversing.
        leaf_value, leaf_moves_left, index = loop_state
        leaf_value = 1 - leaf_value
        # The parent is a ply further from the end of the game
        leaf_moves_left = leaf_moves_left + 1
        parent = tree.parents[index]
        count = tree.node_visits[parent]
        action = tree.action_from_parent[index]
        parent_value = (tree.node_values[parent] * count + leaf_value) / (count + 1.0)
        parent_moves_left = (
            tree.node_moves_left[parent] * count + leaf_moves_left
        ) / (count + 1.0)
        children_value = tree.node_values[index]
        children_count = tree.children_visits[parent, action] + 1

        tree.node_values[parent] = parent_value
        tree.node_moves_left[parent] = parent_moves_left
        tree.node_visits[parent] = count + 1
        tree.children_values[parent, action] = 1 - children_value
        tree.children_visits[parent, action] = children_count

        board.pop()

        return leaf_value, leaf_moves_left, parent

    leaf_index = np.asarray(leaf_index, dtype=np.int32)
    loop_state = (
        tree.node_values[leaf_index],
        tree.node_moves_left[leaf_index],
        leaf_index,
    )
    while cond_fun(loop_state):
        loop_state = body_fun(loop_state)


def update_tree_node(
    tree: Tree,
    node_index: np.int32,
    prior_logits: Optional[np.ndarray],
    value: np.float32,
    moves_left: Optional[np.float32] = None,
):
    """Updates the tree at node index in place.

    Args:
      tree: `Tree` to whose node is to be updated.
      node_index: the index of the expanded node.
      prior_logits: the prior logits to fill in for the new node, of shape
        `[num_actions]`. None for terminal nodes, from which no action is selected.
        Unlike mctx, their softmax is also stored (`children_prior_probs`), since
        the Q-value transformation uses it at each selection.
      value: the value to fill in for the new node.
      moves_left: the moves left to fill in for the new node (0 for a terminal
        node, which keeps 0 when it is reached again). None without the moves-left
        head, which is filled in as 0.
    """

    new_visit = tree.node_visits[node_index] + 1

    if prior_logits is not None:
        tree.children_prior_logits[node_index] = prior_logits
        # Update the prior probabilities together with the logits
        tree.children_prior_probs[node_index] = softmax(prior_logits)
    tree.raw_values[node_index] = value
    tree.node_values[node_index] = value
    moves_left = 0 if moves_left is None else moves_left
    tree.raw_moves_left[node_index] = moves_left
    tree.node_moves_left[node_index] = moves_left
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
        children_prior_probs=np.zeros(
            (num_nodes, num_actions), dtype=root.prior_logits.dtype
        ),
        raw_moves_left=np.zeros(num_nodes, dtype=np.float32),
        node_moves_left=np.zeros(num_nodes, dtype=np.float32),
        children_values=np.zeros((num_nodes, num_actions), dtype=np.float32),
        children_visits=np.zeros((num_nodes, num_actions), dtype=np.int32),
        root_invalid_actions=root_invalid_actions,
        extra_data=extra_data,
    )

    root_index = Tree.ROOT_INDEX
    # The first visit of the root has its predicted moves left, like its value
    update_tree_node(
        tree, root_index, root.prior_logits, root.value, root.moves_left
    )
    return tree
