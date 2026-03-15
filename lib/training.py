"""
Network training step for AlphaZero self-play.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from lib.t7g import apply_obs_symmetry, SYMMETRY_INV_PERMS


def train_network(
    network: torch.nn.Module,
    replay_buffer: deque,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    epochs: int = 5,
    device: str | torch.device = 'cpu',
    curriculum: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    curriculum_ratio: float = 0.25,
) -> dict:
    """
    Train *network* on a random sample from *replay_buffer*.

    One random D4 symmetry is applied per batch to augment data cheaply.

    If *curriculum* is provided it must be a ``(obs, policy, value)`` tuple of
    numpy arrays (pre-loaded from ``value_curriculum.npz``).  Each batch is
    filled with ``(1 - curriculum_ratio)`` replay examples followed by
    ``curriculum_ratio`` curriculum examples.  **Policy loss is computed only
    on the replay portion** — curriculum policy targets are uniform and must
    not flatten the policy gradient.  Value loss covers the full batch so the
    curriculum's accurate outcome labels train the value head.

    Returns
    -------
    dict with keys ``policy_loss``, ``value_loss``, ``total_loss``
    (averages across all batches and epochs).
    """
    if len(replay_buffer) < batch_size:
        return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

    network.train()

    n_curriculum = int(batch_size * curriculum_ratio) if curriculum is not None else 0
    n_replay     = batch_size - n_curriculum

    # Pre-convert replay buffer to numpy in RAM (fast), then move one batch at
    # a time to the device to avoid OOMing GPU VRAM (~500 MB for full policy).
    buffer_list = list(replay_buffer)
    obs_np    = np.array([ex[0] for ex in buffer_list])
    policy_np = np.array([ex[1] for ex in buffer_list])
    value_np  = np.array([ex[2] for ex in buffer_list], dtype=np.float32)
    n = len(buffer_list)

    if curriculum is not None:
        cur_obs, cur_pol, cur_val = curriculum
        n_cur = len(cur_obs)

    epoch_losses: list[dict[str, float]] = []

    for _ in range(epochs):
        ep_policy = 0.0
        ep_value  = 0.0
        ep_batches = 0

        indices = np.random.permutation(n)
        for start in range(0, n - n_replay + 1, n_replay):
            replay_idx = indices[start:start + n_replay]
            k = int(np.random.randint(0, 8))

            # ── Replay portion ────────────────────────────────────────────
            batch_obs    = np.ascontiguousarray(apply_obs_symmetry(obs_np[replay_idx], k))
            batch_policy = policy_np[replay_idx][:, SYMMETRY_INV_PERMS[k]]
            batch_value  = value_np[replay_idx]

            # ── Curriculum portion (appended after replay) ────────────────
            if n_curriculum > 0:
                cur_idx      = np.random.randint(0, n_cur, n_curriculum)
                batch_obs    = np.concatenate([
                    batch_obs, np.ascontiguousarray(apply_obs_symmetry(cur_obs[cur_idx], k))
                ])
                batch_value  = np.concatenate([batch_value, cur_val[cur_idx]])
                # curriculum policy not concatenated — policy loss uses [:n_replay] only

            t_obs    = torch.from_numpy(batch_obs).to(device)
            t_policy = torch.from_numpy(batch_policy).to(device)
            t_value  = torch.from_numpy(batch_value).to(device).unsqueeze(-1)

            optimizer.zero_grad()
            pred_logits, pred_value = network(t_obs)

            # Policy loss: replay samples only (curriculum targets are uniform)
            log_probs   = F.log_softmax(pred_logits, dim=-1)
            policy_loss = -torch.sum(t_policy * log_probs[:n_replay], dim=-1).mean()

            # Value loss: full batch (curriculum provides accurate value labels)
            value_loss  = F.mse_loss(pred_value, t_value)

            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            ep_policy  += policy_loss.item()
            ep_value   += value_loss.item()
            ep_batches += 1

        if ep_batches > 0:
            epoch_losses.append({
                "policy_loss": ep_policy / ep_batches,
                "value_loss":  ep_value  / ep_batches,
            })

    if not epoch_losses:
        return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "epoch_losses": []}

    avg_pol = float(np.mean([e["policy_loss"] for e in epoch_losses]))
    avg_val = float(np.mean([e["value_loss"] for e in epoch_losses]))
    return {
        "policy_loss":  avg_pol,
        "value_loss":   avg_val,
        "total_loss":  avg_pol + avg_val,
        "epoch_losses": epoch_losses,
    }
