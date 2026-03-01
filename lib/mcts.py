"""
AlphaZero-style MCTS for Microscope board game.

Bypasses the gym environment - simulates directly on board copies
using game logic from util/t7g.py. Each tree edge is a complete move
(piece + direction) encoded in 1225-action space.

Performance features:
  - Transposition table: duplicate board positions share statistics and network
    evaluations, so the same position discovered via different paths is only
    evaluated by the network once
  - Cycle detection: if a board position appears twice in the current selection
    path, the simulation terminates with value=0 (draw by repetition)
  - Lazy child expansion: child MCTSNodes created only when first visited
  - Batched network inference: multiple leaf nodes evaluated in one forward pass
  - Virtual loss: discourages re-selecting the same path within a batch
  - Tree reuse: subtree from previous move reused for the next search

Usage:
    from lib.dual_network import DualHeadNetwork
    from lib.mcts import MCTS

    network = DualHeadNetwork()
    mcts = MCTS(network, num_simulations=100)
    action_probs = mcts.search(board, turn=True)
    mcts.advance_tree(selected_action)   # reuse graph next move
"""
from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import (
    action_masks, action_to_move, count_cells,
    apply_move, board_to_obs, check_terminal, new_board,
    BLUE, GREEN, CLEAR,
    Board,  # npt.NDArray[np.bool_], shape (7, 7, 2)
)

# Virtual loss added to discourage parallel paths from converging on the same leaf
VIRTUAL_LOSS = 3


# ============================================================
# Game simulation
# ============================================================

def get_legal_moves(board: Board, turn: bool) -> list[int]:
    """Get list of valid action indices in 1225-action space."""
    masks = action_masks(board, turn)
    return list(np.where(masks)[0])


# ============================================================
# MCTS Node
# ============================================================

class MCTSNode:
    """
    Single node in the MCTS search graph (transposition table entry).

    A node represents a unique (board, turn) position. The same node may be
    reachable via multiple paths from the root — this is the transposition case.
    Visit statistics are shared across all paths that reach this position.
    """

    __slots__ = [
        'board', 'turn', 'action', 'children',
        'visit_count', 'value_sum', 'prior',
        'edge_visits', 'is_terminal', 'terminal_value', 'is_expanded',
        'move_priors', 'network_value',
    ]

    board: Board
    turn: bool
    action: int | None              # action that first created this node (informational)
    children: dict[int, MCTSNode]   # action_1225 -> MCTSNode (lazily populated)
    visit_count: int
    value_sum: float
    prior: float
    edge_visits: dict[int, int]     # action -> simulation count through this edge
    is_terminal: bool
    terminal_value: float | None
    is_expanded: bool
    move_priors: dict[int, float] | None  # set during _expand_batch
    network_value: float                  # set during _expand_batch

    def __init__(
        self,
        board: Board,
        turn: bool,
        action: int | None = None,
        prior: float = 0.0,
    ) -> None:
        self.board = board
        self.turn = turn
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.edge_visits = {}
        self.is_expanded = False
        self.move_priors = None
        self.network_value = 0.0

        # Check terminal state
        self.is_terminal, self.terminal_value = check_terminal(board, turn)

    @property
    def q_value(self) -> float:
        """Average value from this node's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ============================================================
# MCTS Search
# ============================================================

class MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search with neural network guidance.

    The search graph uses a transposition table so that identical board
    positions discovered via different move sequences share the same node and
    its visit statistics. Cycles are detected per-simulation and treated as
    draws (value = 0).

    Optimizations vs naive implementation:
      - Transposition table: O(1) lookup prevents duplicate network evaluations
        and merges visit counts across transposed paths
      - Cycle detection: repeated positions in one selection path → value=0
      - Lazy child expansion: nodes created only when first selected
      - Batched inference: `inference_batch_size` leaves evaluated per forward pass
      - Virtual loss: steers batch selections to different branches
      - Graph reuse: call advance_tree(action) to carry the graph across moves
    """

    def __init__(
        self,
        network: nn.Module,
        num_simulations: int = 100,
        c_puct: float = 2.0,
        dirichlet_alpha: float = 0.1,
        dirichlet_epsilon: float = 0.25,
        inference_batch_size: int = 8,
    ) -> None:
        """
        Args:
            network: DualHeadNetwork (or any nn.Module with the same interface)
            num_simulations: MCTS rollouts per search call
            c_puct: exploration constant for PUCT formula
            dirichlet_alpha: Dirichlet noise concentration at root
            dirichlet_epsilon: weight of Dirichlet noise at root
            inference_batch_size: leaves per network forward pass
        """
        self.network: nn.Module = network
        self.num_simulations: int = num_simulations
        self.c_puct: float = c_puct
        self.dirichlet_alpha: float = dirichlet_alpha
        self.dirichlet_epsilon: float = dirichlet_epsilon
        self.inference_batch_size: int = inference_batch_size
        self.root: MCTSNode | None = None
        self.transposition_table: dict[bytes, MCTSNode] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, board: Board, turn: bool) -> npt.NDArray[np.float32]:
        """
        Run MCTS from position, return action probability distribution.

        Args:
            board: 7x7x2 numpy bool array
            turn: True=Blue, False=Green

        Returns:
            action_probs: 1225-element numpy array of visit-count probabilities
        """
        # --- Graph reuse: reset only when the board position changes ---
        root_key = self._board_key(board, turn)
        if self.root is None or not np.array_equal(self.root.board, board):
            self.transposition_table = {}
            self.root = MCTSNode(board, turn)
            self.transposition_table[root_key] = self.root
        root = self.root

        # Expand root if not already done
        if not root.is_terminal and not root.is_expanded:
            self._expand_batch([root])

        # Add Dirichlet noise to root move_priors for exploration
        if root.move_priors and self.dirichlet_epsilon > 0:
            actions = list(root.move_priors.keys())
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(actions)
            )
            for i, action in enumerate(actions):
                root.move_priors[action] = (
                    (1 - self.dirichlet_epsilon) * root.move_priors[action]
                    + self.dirichlet_epsilon * noise[i]
                )

        # --- Run simulations in batches ---
        sims_done = 0
        while sims_done < self.num_simulations:
            batch_leaves = []
            batch_nodes = []    # path node lists for each batch leaf
            batch_edges = []    # path edge lists for each batch leaf

            batch_target = min(
                self.inference_batch_size,
                self.num_simulations - sims_done
            )

            for _ in range(batch_target):
                nodes, edges, is_cycle = self._select_path(root)
                leaf = nodes[-1]

                if is_cycle:
                    # Repeated position in this path → draw by repetition
                    self._backpropagate(nodes, edges, 0.0)
                    sims_done += 1
                elif leaf.is_terminal:
                    self._backpropagate(nodes, edges, leaf.terminal_value)
                    sims_done += 1
                else:
                    # Apply virtual loss so other batch sims avoid this path
                    self._apply_virtual_loss(nodes)
                    batch_leaves.append(leaf)
                    batch_nodes.append(nodes)
                    batch_edges.append(edges)

            # EXPAND batch + BACKPROPAGATE
            if batch_leaves:
                self._expand_batch(batch_leaves)
                for leaf, nodes, edges in zip(batch_leaves, batch_nodes, batch_edges):
                    self._remove_virtual_loss(nodes)
                    self._backpropagate(nodes, edges, leaf.network_value)
                    sims_done += 1

        # Build action probability distribution from per-edge visit counts.
        # Using edge_visits (not child.visit_count) correctly handles the rare
        # case where two different root moves lead to the same transposed position.
        action_probs = np.zeros(1225, dtype=np.float32)
        for action, count in root.edge_visits.items():
            action_probs[action] = count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def advance_tree(self, action: int) -> None:
        """
        Reuse the subgraph rooted at the child for `action`.

        The transposition table is preserved across the advance so previously
        explored positions retain their statistics. Call this after selecting
        and applying a move.
        """
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = None
            self.transposition_table = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _board_key(self, board: Board, turn: bool) -> bytes:
        """Compact hashable key for the transposition table."""
        return board.tobytes() + bytes([turn])

    def _select_path(
        self, root: MCTSNode
    ) -> tuple[list[MCTSNode], list[int], bool]:
        """
        Traverse from root to a leaf using PUCT selection.

        Returns:
            nodes: list[MCTSNode] from root to leaf (inclusive)
            edges: list[int] of actions taken; edges[i] goes from nodes[i] to nodes[i+1]
            is_cycle: True if the leaf position already appeared earlier in the path
        """
        nodes = [root]
        edges = []
        visited_keys = {self._board_key(root.board, root.turn)}
        node = root

        while node.is_expanded and not node.is_terminal:
            action = self._best_action(node)
            child = self._get_or_create_child(node, action)
            edges.append(action)
            nodes.append(child)

            key = self._board_key(child.board, child.turn)
            if key in visited_keys:
                return nodes, edges, True   # cycle detected
            visited_keys.add(key)
            node = child

        return nodes, edges, False

    def _best_action(self, node: MCTSNode) -> int:
        """Return the action with the highest PUCT score from this node."""
        best_score = -float('inf')
        best_action = None
        sqrt_parent = math.sqrt(node.visit_count)

        for action, prior in node.move_priors.items():
            child = node.children.get(action)
            q = 0.0 if child is None else -child.q_value   # negate: child is opponent
            visits = 0 if child is None else child.visit_count

            score = q + self.c_puct * prior * sqrt_parent / (1 + visits)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _get_or_create_child(self, node: MCTSNode, action: int) -> MCTSNode:
        """
        Return the child for `action`, creating it lazily if needed.

        Checks the transposition table before allocating a new node — if this
        board position was already reached via a different path, the existing
        node (with its accumulated statistics) is reused.
        """
        if action not in node.children:
            child_board = apply_move(node.board, action, node.turn)
            child_turn = not node.turn
            key = self._board_key(child_board, child_turn)

            if key in self.transposition_table:
                child = self.transposition_table[key]
            else:
                child = MCTSNode(
                    child_board,
                    turn=child_turn,
                    action=action,
                    prior=node.move_priors[action],
                )
                self.transposition_table[key] = child

            node.children[action] = child

        return node.children[action]

    def _expand_batch(self, nodes: list[MCTSNode]) -> None:
        """
        Evaluate a batch of leaf nodes in a single network forward pass.

        Sets node.move_priors, node.network_value, node.is_expanded.
        Handles terminal / no-legal-moves cases without a network call.
        Skips nodes that are already expanded (can happen with transpositions
        when two batch simulations reach the same unexpanded position).
        """
        to_evaluate = []
        legal_moves_list = []

        for node in nodes:
            if node.is_terminal or node.is_expanded:
                continue   # already handled

            legal_moves = get_legal_moves(node.board, node.turn)
            if not legal_moves:
                # No moves = loss for current player
                node.is_terminal = True
                node.terminal_value = -1.0
                node.network_value = -1.0
                node.is_expanded = True
            else:
                to_evaluate.append(node)
                legal_moves_list.append(legal_moves)

        if not to_evaluate:
            return

        # --- Single batched forward pass ---
        obs_batch = np.stack(
            [board_to_obs(n.board, n.turn) for n in to_evaluate]
        )
        obs_tensor = torch.from_numpy(obs_batch).to(
            next(self.network.parameters()).device
        )

        self.network.eval()
        with torch.no_grad():
            policy_logits, values = self.network(obs_tensor)
            policy_probs_batch = F.softmax(policy_logits, dim=-1).cpu().numpy()
            values_flat = values.cpu().numpy().flatten()

        # Assign results to nodes
        for node, policy_probs, value, legal_moves in zip(
            to_evaluate, policy_probs_batch, values_flat, legal_moves_list
        ):
            # Mask policy to legal moves and renormalize
            priors = {m: float(policy_probs[m]) for m in legal_moves}
            total = sum(priors.values())
            if total > 0:
                priors = {m: p / total for m, p in priors.items()}
            else:
                uniform = 1.0 / len(legal_moves)
                priors = {m: uniform for m in legal_moves}

            node.move_priors = priors
            node.network_value = float(value)
            node.is_expanded = True

    def _apply_virtual_loss(self, nodes: list[MCTSNode]) -> None:
        """Add pessimistic counts to all nodes on the selection path."""
        for node in nodes:
            node.visit_count += VIRTUAL_LOSS
            node.value_sum -= VIRTUAL_LOSS

    def _remove_virtual_loss(self, nodes: list[MCTSNode]) -> None:
        """Undo virtual loss before applying real backpropagation."""
        for node in nodes:
            node.visit_count -= VIRTUAL_LOSS
            node.value_sum += VIRTUAL_LOSS

    def _backpropagate(
        self, nodes: list[MCTSNode], edges: list[int], value: float
    ) -> None:
        """
        Update visit counts, values, and edge visit counts from leaf to root.

        nodes[i] --edges[i]--> nodes[i+1], with value from nodes[-1]'s perspective.
        Sign flips at each step to account for alternating player perspectives.
        """
        for i in range(len(nodes) - 1, -1, -1):
            node = nodes[i]
            node.visit_count += 1
            node.value_sum += value
            if i < len(edges):
                action = edges[i]
                node.edge_visits[action] = node.edge_visits.get(action, 0) + 1
            value = -value  # flip perspective at each level

    def select_action(
        self, action_probs: npt.NDArray[np.float32], temperature: float = 1.0
    ) -> int:
        """
        Select action from MCTS probability distribution.

        Args:
            action_probs: 1225-element probability distribution
            temperature: 1.0 = proportional to visits, 0 = greedy

        Returns:
            selected action index (1225-space)
        """
        if temperature == 0:
            return int(np.argmax(action_probs))

        probs = action_probs ** (1.0 / temperature)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            return int(np.argmax(action_probs))

        return int(np.random.choice(len(probs), p=probs))