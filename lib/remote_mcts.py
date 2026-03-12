"""
RemoteMCTS: MCTS subclass that delegates network inference to a
central InferenceServer process via multiprocessing queues.

All search logic (PUCT selection, backpropagation, tree reuse,
transposition table, cycle detection, virtual loss) is inherited from
MCTS unchanged.  Only _expand_batch is overridden to use queue-based
IPC instead of a local network forward pass.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from typing import List

import numpy as np

from lib.mcts import MCTS, MCTSNode, get_legal_moves
from lib.t7g import board_to_obs


class RemoteMCTS(MCTS):
    """
    MCTS using a central inference server instead of a local network.

    Args:
        worker_id:      unique integer (0..num_workers-1) assigned by the pool
        request_queue:  shared queue for sending obs batches to the server
        result_queue:   this worker's private queue for receiving results
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: "mp.Queue[object]",
        result_queue: "mp.Queue[object]",
        num_simulations: int = 100,
        c_puct: float = 2.0,
        dirichlet_alpha: float = 0.1,
        dirichlet_epsilon: float = 0.25,
        inference_batch_size: int = 16,
    ) -> None:
        # Initialise all MCTS instance state without a local network
        self.network = None  # type: ignore[assignment]
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.inference_batch_size = inference_batch_size
        self.root = None
        self.transposition_table: dict = {}
        # Queue handles
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue

    def _expand_batch(self, nodes: List[MCTSNode]) -> None:
        """Expand leaf nodes via the central inference server."""
        to_evaluate: List[MCTSNode] = []
        legal_moves_list = []

        for node in nodes:
            if node.is_terminal or node.is_expanded:
                continue

            legal_moves = get_legal_moves(node.board, node.turn)
            if not legal_moves:
                # Current player has no moves but the game is not terminal
                # (check_terminal already ran in MCTSNode.__init__ and returned False,
                # meaning the opponent can still move).  Mirror MCTS._expand_batch:
                # expose a single PASS_ACTION child so the tree continues searching.
                node.move_priors = {1225: 1.0}  # PASS_ACTION = 1225
                node.network_value = 0.0
                node.is_expanded = True
            else:
                to_evaluate.append(node)
                legal_moves_list.append(legal_moves)

        if not to_evaluate:
            return

        obs_batch = np.stack(
            [board_to_obs(n.board, n.turn) for n in to_evaluate]
        )

        # Send to server and wait for results synchronously
        self.request_queue.put((self.worker_id, obs_batch))
        try:
            policy_probs_batch, values_flat = self.result_queue.get(timeout=30)  # type: ignore[misc]
        except queue.Empty:
            raise RuntimeError(
                f"Worker {self.worker_id}: inference server did not respond within 30s — "
                "it may have crashed"
            )

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
