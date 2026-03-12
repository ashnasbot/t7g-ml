"""
Central GPU inference server for batched MCTS leaf evaluation.

Workers put (worker_id, obs_batch) on the shared request_queue.
The server drains the queue each round, concatenates observations from
concurrent workers into one large batch, runs a single forward pass,
then puts (policy_probs, values) on each worker's private result queue.

This replaces the previous design where each worker held its own
network copy and ran independent small-batch forward passes.  With N
workers all routing through one server, a typical forward pass sees
N × inference_batch_size observations, saturating the GPU far better.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from lib.dual_network import DualHeadNetwork


class InferenceServer(mp.Process):
    """
    Dedicated GPU inference process.

    Collects leaf observation batches from multiple MCTS worker
    processes, combines them into one large batch per GPU call, and
    returns (policy_probs, values) on per-worker result queues.
    """

    def __init__(
        self,
        state_dict: dict,
        request_queue: "mp.Queue[object]",
        result_queues: "List[mp.Queue[object]]",
        num_workers: int,
    ) -> None:
        super().__init__(daemon=True, name="InferenceServer")
        self.state_dict = state_dict
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.num_workers = num_workers

    def run(self) -> None:
        # Device selection (mirrors worker logic)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                import torch_directml  # type: ignore[import-untyped]
                device = torch_directml.device()
            except ImportError:
                device = torch.device("cpu")

        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")  # TF32 matmul on Ampere+

        network = DualHeadNetwork(num_actions=1225)
        network.load_state_dict(self.state_dict)
        network.to(device)
        network.eval()

        active_workers = self.num_workers
        while active_workers > 0:
            # Block until the first request (or shutdown signal) arrives
            try:
                item = self.request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                active_workers -= 1
                continue

            pending = [item]

            # Brief collection window: give other workers time to submit
            # their requests so they land in the same forward pass.
            time.sleep(0.002)

            # Drain: collect any requests that arrived during the window
            while True:
                try:
                    item = self.request_queue.get_nowait()
                    if item is None:
                        active_workers -= 1
                    else:
                        pending.append(item)
                except queue.Empty:
                    break

            if not pending:
                continue

            worker_ids = [req[0] for req in pending]  # type: ignore[index]
            obs_list = [req[1] for req in pending]    # type: ignore[index]
            sizes = [obs.shape[0] for obs in obs_list]

            obs_tensor = torch.from_numpy(
                np.concatenate(obs_list, axis=0)
            ).to(device)

            with torch.no_grad():
                policy_logits, values = network(obs_tensor)
                policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()

            # Distribute results back to each requesting worker
            offset = 0
            for worker_id, size in zip(worker_ids, sizes):
                self.result_queues[worker_id].put((
                    policy_probs[offset:offset + size],
                    values_np[offset:offset + size],
                ))
                offset += size
