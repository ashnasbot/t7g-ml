"""
AlphaZero-style MCTS for Microscope board game.

Bypasses the gym environment - simulates directly on board copies
using game logic from util/t7g.py. Each tree edge is a complete move
(piece + direction) encoded in 1225-action space.

Performance features:
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
    mcts.advance_tree(selected_action)   # reuse tree next move
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from lib.t7g import (
    action_masks, action_to_move, count_cells,
    apply_move, board_to_obs, check_terminal, new_board,
    BLUE, GREEN, CLEAR
)

# Virtual loss added to discourage parallel paths from converging on the same leaf
VIRTUAL_LOSS = 3


# ============================================================
# Game simulation
# ============================================================

def get_legal_moves(board, turn):
    """Get list of valid action indices in 1225-action space."""
    masks = action_masks(board, turn)
    return list(np.where(masks)[0])


# ============================================================
# MCTS Node
# ============================================================

class MCTSNode:
    """Single node in the MCTS tree."""

    __slots__ = [
        'board', 'turn', 'parent', 'action', 'children',
        'visit_count', 'value_sum', 'prior',
        'is_terminal', 'terminal_value', 'is_expanded',
        # Lazy expansion: priors stored here; children created on first visit
        'move_priors',    # dict[action -> prior prob] for all legal moves
        'network_value',  # cached network value estimate (for backprop)
    ]

    def __init__(self, board, turn, parent=None, action=None, prior=0.0):
        self.board = board
        self.turn = turn
        self.parent = parent
        self.action = action        # action that led to this node
        self.children = {}          # action_1225 -> MCTSNode (lazily populated)
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.move_priors = None     # set during _expand_batch
        self.network_value = 0.0   # set during _expand_batch

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

    Optimizations vs naive implementation:
      - Lazy child expansion: nodes created only when first selected
      - Batched inference: `inference_batch_size` leaves evaluated per forward pass
      - Virtual loss: steers batch selections to different tree branches
      - Tree reuse: call advance_tree(action) to carry the tree across moves
    """

    def __init__(self, network, num_simulations=100, c_puct=2.0,
                 dirichlet_alpha=0.1, dirichlet_epsilon=0.25,
                 inference_batch_size=8):
        """
        Args:
            network: DualHeadNetwork
            num_simulations: MCTS rollouts per search call
            c_puct: exploration constant for PUCT formula
            dirichlet_alpha: Dirichlet noise concentration at root
            dirichlet_epsilon: weight of Dirichlet noise at root
            inference_batch_size: leaves per network forward pass
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.inference_batch_size = inference_batch_size
        self.root = None  # persistent root for tree reuse

    def search(self, board, turn):
        """
        Run MCTS from position, return action probability distribution.

        Args:
            board: 7x7x2 numpy bool array
            turn: True=Blue, False=Green

        Returns:
            action_probs: 1225-element numpy array of visit-count probabilities
        """
        # --- Tree reuse: reuse existing subtree if board matches ---
        if self.root is None or not np.array_equal(self.root.board, board):
            self.root = MCTSNode(board, turn)
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
            batch_target = min(
                self.inference_batch_size,
                self.num_simulations - sims_done
            )

            for _ in range(batch_target):
                node = root

                # SELECT: traverse tree using PUCT until a leaf
                while node.is_expanded and not node.is_terminal:
                    node = self._select_child(node)

                if node.is_terminal:
                    self._backpropagate(node, node.terminal_value)
                    sims_done += 1
                else:
                    # Apply virtual loss so parallel selections avoid this path
                    self._apply_virtual_loss(node)
                    batch_leaves.append(node)

            # EXPAND batch + BACKPROPAGATE
            if batch_leaves:
                self._expand_batch(batch_leaves)
                for node in batch_leaves:
                    self._remove_virtual_loss(node)
                    self._backpropagate(node, node.network_value)
                    sims_done += 1

        # Build action probability distribution from visit counts
        action_probs = np.zeros(1225, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def advance_tree(self, action):
        """
        Reuse the subtree rooted at the child for `action`.

        Call this after selecting and applying a move so the next search
        builds on existing visit counts instead of starting from scratch.
        """
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None   # detach → allows GC of old branches
        else:
            self.root = None          # child not visited yet; start fresh

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_child(self, node):
        """Select child with highest PUCT score, creating it lazily if needed."""
        best_score = -float('inf')
        best_action = None
        sqrt_parent = math.sqrt(node.visit_count)

        for action, prior in node.move_priors.items():
            child = node.children.get(action)
            if child is None:
                q = 0.0
                visits = 0
            else:
                q = -child.q_value   # negate: child value is opponent's perspective
                visits = child.visit_count

            score = q + self.c_puct * prior * sqrt_parent / (1 + visits)
            if score > best_score:
                best_score = score
                best_action = action

        # Lazily create the child node only now (on first selection)
        if best_action not in node.children:
            child_board = apply_move(node.board, best_action, node.turn)
            child = MCTSNode(
                child_board,
                turn=not node.turn,
                parent=node,
                action=best_action,
                prior=node.move_priors[best_action],
            )
            node.children[best_action] = child

        return node.children[best_action]

    def _expand_batch(self, nodes):
        """
        Evaluate a batch of leaf nodes in a single network forward pass.

        Sets node.move_priors, node.network_value, node.is_expanded.
        Handles terminal / no-legal-moves cases without a network call.
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

    def _apply_virtual_loss(self, node):
        """Add pessimistic counts up the tree to discourage re-selection."""
        n = node
        while n is not None:
            n.visit_count += VIRTUAL_LOSS
            n.value_sum -= VIRTUAL_LOSS   # value = -1 per virtual visit
            n = n.parent

    def _remove_virtual_loss(self, node):
        """Undo virtual loss before applying real backpropagation."""
        n = node
        while n is not None:
            n.visit_count -= VIRTUAL_LOSS
            n.value_sum += VIRTUAL_LOSS
            n = n.parent

    def _backpropagate(self, node, value):
        """Update visit counts and values from leaf to root."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value   # flip perspective at each level
            node = node.parent

    def select_action(self, action_probs, temperature=1.0):
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
