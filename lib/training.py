"""
Network training step for AlphaZero self-play.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from lib.t7g import apply_obs_symmetry, SYMMETRY_INV_PERMS, SYMMETRY_INV_PERMS_49


def train_network(
    network: torch.nn.Module,
    replay_buffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    epochs: int = 5,
    device: str | torch.device = 'cpu',
    desc: str | None = None,
) -> dict:
    """
    Train *network* on a random sample from *replay_buffer*.

    One random D4 symmetry is applied per batch to augment data cheaply.

    Returns
    -------
    dict with keys ``policy_loss``, ``value_loss``, ``total_loss``
    (averages across all batches and epochs).
    """
    if len(replay_buffer) < batch_size:
        return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

    network.train()

    buffer_list = list(replay_buffer)
    n = len(buffer_list)

    epoch_losses: list[dict[str, float]] = []

    epoch_iter = tqdm(range(epochs), desc=desc, unit="epoch") if desc else range(epochs)
    for _ in epoch_iter:
        ep_policy = 0.0
        ep_value  = 0.0
        ep_batches = 0

        indices = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            replay_idx = indices[start:start + batch_size]
            k = int(np.random.randint(0, 8))

            batch_obs    = np.ascontiguousarray(apply_obs_symmetry(
                np.array([buffer_list[i][0] for i in replay_idx]), k))
            raw_policy   = np.array([buffer_list[i][1] for i in replay_idx])
            # Select correct symmetry permutation based on policy space size.
            inv_perm     = (SYMMETRY_INV_PERMS_49[k] if raw_policy.shape[1] == 49
                            else SYMMETRY_INV_PERMS[k])
            batch_policy = raw_policy[:, inv_perm]
            batch_value  = np.array([buffer_list[i][2] for i in replay_idx], dtype=np.float32)

            t_obs    = torch.from_numpy(batch_obs).to(device)
            t_policy = torch.from_numpy(batch_policy).to(device)
            t_value  = torch.from_numpy(batch_value).to(device).unsqueeze(-1)

            optimizer.zero_grad()
            pred_logits, pred_value = network(t_obs)

            log_probs   = F.log_softmax(pred_logits, dim=-1)
            policy_loss = -torch.sum(t_policy * log_probs, dim=-1).mean()
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
