"""
Gumbel AlphaZero Monte Carlo Graph Search (MCGS) for Microscope board game.

Shared transposition table so positions reachable via multiple paths
accumulate visits from all paths, giving tighter Q estimates.

Replaces Dirichlet noise + visit-count policy with Gumbel search:
  - Gumbel top-K root sampling   — diverse candidates without spending sims
  - Sequential Halving           — allocate budget across K candidates, halve each round
  - Completed-Q policy           — softmax(gumbel + Q) provably beats the prior at any sim count

Internal nodes (below the root) keep PUCT unchanged.

Reference: Danihelka et al. 2022, "Policy improvement by planning with Gumbel"
"""
from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import (
    action_masks, apply_move, board_to_obs, check_terminal,
    Board,
)

# Sentinel action for forced-pass positions (no legal moves but game not terminal).
# Never appears in the action_probs returned by search().
PASS_ACTION = 1225


# ============================================================
# Game simulation
# ============================================================

def get_legal_moves(board: Board, turn: bool) -> list[int]:
    """Get list of valid action indices in 1225-action space."""
    return list(np.where(action_masks(board, turn))[0])


# ============================================================
# MCGS Node
# ============================================================

class MCGSNode:
    """
    Single node in the MCGS search graph (transposition table entry).

    A node represents a unique (board + turn) position. The same node may be
    reachable via multiple paths from the root — this is the transposition case.
    Visit statistics are shared across all paths that reach this position.
    """

    __slots__ = [
        'board', 'turn', 'children', 'key',
        'visit_count', 'value_sum',
        'is_terminal', 'terminal_value', 'is_expanded',
        'move_priors', 'network_value',
    ]

    def __init__(
        self,
        board: Board,
        turn: bool,
        prior: float = 0.0,  # kept for API compatibility; not stored
    ) -> None:
        self.board = board
        self.turn = turn
        self.key: bytes = board.tobytes() + (b'\x01' if turn else b'\x00')
        self.children: dict[int, MCGSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.move_priors: dict[int, float] | None = None
        self.network_value = 0.0
        self.is_terminal, self.terminal_value = check_terminal(board, turn)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ============================================================
# MCGS Search
# ============================================================

class MCGS:
    """
    Gumbel AlphaZero Monte Carlo Graph Search with transposition table.

    At the root:
      1. Sample K actions with Gumbel top-K (log-prior + Gumbel noise)
      2. Sequential Halving allocates the simulation budget across K candidates
      3. Policy extracted via completed-Q: softmax(gumbel_score + Q)

    Below the root: standard PUCT selection unchanged.
    """

    def __init__(
        self,
        network: nn.Module,
        num_simulations: int = 100,
        c_puct: float = 0.75,
        gumbel_k: int = 8,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gumbel_k = gumbel_k
        self.root: MCGSNode | None = None
        self.transposition_table: dict[bytes, MCGSNode] = {}
        self._device = next(network.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, board: Board, turn: bool) -> npt.NDArray[np.float32]:
        """
        Run Gumbel MCGS from position, return improved action probability distribution.

        Returns:
            action_probs: 1225-element array (non-zero only over the K sampled actions)
        """
        root_key = self._board_key(board, turn)
        if self.root is None or self.root.turn != turn or not np.array_equal(self.root.board, board):
            # Fresh root, but preserve the transposition table: Q estimates
            # cached earlier in this game are still valid and save re-expansion.
            self.root = MCGSNode(board, turn)
            self.transposition_table[root_key] = self.root
        root = self.root

        if not root.is_terminal and not root.is_expanded:
            self._expand_batch([root])

        if root.is_terminal or not root.move_priors:
            return np.zeros(1225, dtype=np.float32)

        legal = [a for a in root.move_priors if a != PASS_ACTION]
        if not legal:
            return np.zeros(1225, dtype=np.float32)

        # ── Gumbel top-K ──────────────────────────────────────────────
        # score(a) = log π(a) + Gumbel noise; sample K without replacement
        K = min(self.gumbel_k, len(legal))
        gumbel: dict[int, float] = {
            a: (math.log(max(root.move_priors[a], 1e-9))
                - math.log(-math.log(np.random.uniform())))
            for a in legal
        }
        top_k = sorted(legal, key=lambda a: gumbel[a], reverse=True)[:K]

        # ── Sequential Halving ────────────────────────────────────────
        # R rounds; each round allocates n_r sims per active action, then halves.
        # Each step runs one path per active action and batch-expands their leaves
        # in a single forward pass (batch size = len(active)).  This recovers GPU
        # efficiency: O(n_r) forward passes per round instead of O(n_r * K).
        #
        # Subtle weakening: paths in a batch share Q estimates at selection time
        # (backprop hasn't run yet), so they may converge on the same leaf.
        # Deduplication in _expand_paths handles correctness; the effect is a
        # slight reduction in unique expansions per batch, negligible in practice.
        N = self.num_simulations
        R = max(1, math.ceil(math.log2(K)))
        active = list(top_k)
        sims_done = 0

        for _ in range(R):
            if len(active) <= 1 or sims_done >= N:
                break
            n_r = max(1, N // (R * len(active)))
            for _ in range(n_r):
                if sims_done >= N:
                    break
                paths = [self._select_path(root, a) for a in active]
                self._expand_paths(paths)
                for nodes, is_cycle in paths:
                    self._backprop_path(nodes, is_cycle)
                sims_done += len(active)
            # Keep top half by Q-value estimate
            active.sort(key=lambda a: self._action_q(root, a), reverse=True)
            active = active[:max(1, len(active) // 2)]

        # Tail: drain remaining budget on the survivor, batched for GPU efficiency
        while sims_done < N:
            batch = min(K, N - sims_done)
            paths = [self._select_path(root, active[0]) for _ in range(batch)]
            self._expand_paths(paths)
            for nodes, is_cycle in paths:
                self._backprop_path(nodes, is_cycle)
            sims_done += batch

        # ── Completed-Q policy ────────────────────────────────────────
        # logit(a) = gumbel(a) + Q̄(a)   [unvisited actions use root network value]
        logits = np.array([gumbel[a] + self._action_q(root, a) for a in top_k])
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        action_probs = np.zeros(1225, dtype=np.float32)
        for a, p in zip(top_k, probs):
            action_probs[a] = float(p)
        return action_probs

    def advance_tree(self, action: int) -> None:
        """Reuse the subgraph rooted at the child for `action`.

        On a root miss the transposition table is intentionally preserved.
        All cached Q estimates remain valid for positions in this game, so
        retaining them saves re-expansion work.  The cost is that pre-loaded
        children have uneven visit counts going into Sequential Halving, which
        weakens the equal-budget guarantee — but the free Q information
        outweighs this in practice.
        """
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = None  # transposition table kept deliberately

    def select_action(
        self, action_probs: npt.NDArray[np.float32], temperature: float = 1.0
    ) -> int:
        """Select action from MCGS probability distribution."""
        if temperature == 0:
            return int(np.argmax(action_probs))
        probs = action_probs ** (1.0 / temperature)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            return int(np.argmax(action_probs))
        return int(np.random.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _board_key(self, board: Board, turn: bool) -> bytes:
        return board.tobytes() + bytes([turn])

    def _action_q(self, root: MCGSNode, action: int) -> float:
        """Q-value for a root action from the current player's perspective."""
        child = root.children.get(action)
        if child is None or child.visit_count == 0:
            return root.network_value
        return -child.q_value  # negate: child is opponent's perspective

    def _expand_paths(
        self, paths: list[tuple[list[MCGSNode], bool]]
    ) -> None:
        """Batch-expand unique unexpanded leaves across a set of simulation paths."""
        seen: set[int] = set()
        leaves: list[MCGSNode] = []
        for nodes, is_cycle in paths:
            if is_cycle:
                continue
            leaf = nodes[-1]
            if not leaf.is_terminal and not leaf.is_expanded and id(leaf) not in seen:
                leaves.append(leaf)
                seen.add(id(leaf))
        if leaves:
            self._expand_batch(leaves)

    def _backprop_path(self, nodes: list[MCGSNode], is_cycle: bool) -> None:
        """Backpropagate a single simulation path."""
        leaf = nodes[-1]
        if is_cycle:
            self._backpropagate(nodes, 0.0)
        elif leaf.is_terminal:
            assert leaf.terminal_value is not None
            self._backpropagate(nodes, leaf.terminal_value)
        else:
            self._backpropagate(nodes, leaf.network_value)

    def _select_path(
        self, root: MCGSNode, forced_first_action: int | None = None
    ) -> tuple[list[MCGSNode], bool]:
        """
        Traverse from root to a leaf, optionally forcing the first action.

        Returns (nodes, is_cycle).
        """
        nodes = [root]
        visited_keys = {root.key}
        node = root

        while node.is_expanded and not node.is_terminal:
            if forced_first_action is not None:
                action = forced_first_action
                forced_first_action = None
            else:
                action = self._best_action(node)

            child = self._get_or_create_child(node, action)
            nodes.append(child)

            if child.key in visited_keys:
                return nodes, True
            visited_keys.add(child.key)
            node = child

        return nodes, False

    def _best_action(self, node: MCGSNode) -> int:
        """Return the action with the highest PUCT score."""
        assert node.move_priors is not None
        best_score = -float('inf')
        best_action = None
        sqrt_parent = math.sqrt(node.visit_count)

        for action, prior in node.move_priors.items():
            child = node.children.get(action)
            q = 0.0 if child is None else -child.q_value
            visits = 0 if child is None else child.visit_count
            score = q + self.c_puct * prior * sqrt_parent / (1 + visits)
            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None
        return best_action

    def _get_or_create_child(self, node: MCGSNode, action: int) -> MCGSNode:
        """Return the child for `action`, creating and transposition-checking if needed."""
        assert node.move_priors is not None
        if action not in node.children:
            if action == PASS_ACTION:
                child_board, child_turn = node.board, not node.turn
            else:
                child_board = apply_move(node.board, action, node.turn)
                child_turn = not node.turn
            key = self._board_key(child_board, child_turn)

            if key in self.transposition_table:
                child = self.transposition_table[key]
            else:
                child = MCGSNode(child_board, child_turn)
                self.transposition_table[key] = child

            node.children[action] = child
        return node.children[action]

    def _expand_batch(self, nodes: list[MCGSNode]) -> None:
        """Evaluate a batch of leaf nodes in a single network forward pass."""
        to_evaluate = []
        legal_moves_list = []

        for node in nodes:
            if node.is_terminal or node.is_expanded:
                continue
            legal_moves = get_legal_moves(node.board, node.turn)
            if not legal_moves:
                node.move_priors = {PASS_ACTION: 1.0}
                node.network_value = 0.0
                node.is_expanded = True
            else:
                to_evaluate.append(node)
                legal_moves_list.append(legal_moves)

        if not to_evaluate:
            return

        obs_batch = np.stack([board_to_obs(n.board, n.turn) for n in to_evaluate])
        obs_tensor = torch.from_numpy(obs_batch).to(self._device)
        self.network.eval()
        with torch.no_grad():
            policy_logits, values = self.network(obs_tensor)
            policy_probs_batch = F.softmax(policy_logits, dim=-1).cpu().numpy()
            values_flat = values.cpu().numpy().flatten()

        for node, policy_probs, value, legal_moves in zip(
            to_evaluate, policy_probs_batch, values_flat, legal_moves_list
        ):
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

    def _backpropagate(self, nodes: list[MCGSNode], value: float) -> None:
        """Update visit counts and values from leaf to root, flipping sign each level."""
        for node in reversed(nodes):
            node.visit_count += 1
            node.value_sum += value
            value = -value
