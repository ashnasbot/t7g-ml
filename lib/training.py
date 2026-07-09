"""
Network training step and replay buffer for AlphaZero self-play.
"""
from __future__ import annotations

from collections import deque
from itertools import chain

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from lib.t7g import apply_obs_symmetry, SYMMETRY_INV_PERMS, SYMMETRY_INV_PERMS_49


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class _IterBuffer:
    """
    Rolling window replay buffer sized by iteration count, not example count.

    Stores the last *maxiters* iterations as separate batches so the window
    stays proportional to current game length.  Early training produces long
    branchy games (~17k examples/iter); late training produces short focused
    games (~1-2k examples/iter).  A fixed example-count deque would hold 60+
    stale iterations in the latter case; this stays at exactly N.
    """

    def __init__(self, maxiters: int) -> None:
        self._batches: deque = deque(maxlen=maxiters)

    def append_batch(self, batch: list) -> None:
        self._batches.append(batch)

    def __len__(self) -> int:
        return sum(len(b) for b in self._batches)

    def __iter__(self):
        return chain.from_iterable(self._batches)


def train_network(
    network: torch.nn.Module,
    replay_buffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    epochs: int = 5,
    device: str | torch.device = 'cpu',
    desc: str | None = None,
    entropy_coeff: float = 0.0,
    value_coef: float = 1.0,
    margin_coef: float = 0.0,
) -> dict:
    """
    Train *network* on a random sample from *replay_buffer*.

    Full 8× D4 symmetry augmentation: every batch is expanded to all eight
    rotations/reflections of each example, giving the value head a
    full-group gradient per step (cleaner than 1× random in a short-run
    regime like ours).

    Buffer examples are (obs, policy, value) or (obs, policy, value, margin).
    When margin_coef > 0, examples carrying a 4th element train the auxiliary
    margin head (final material margin / 49, side-to-move perspective) with an
    extra MSE term; 3-tuple examples (BC warmup, MM-mix) are masked out of
    that loss.

    Returns
    -------
    dict with keys ``policy_loss``, ``value_loss``, ``margin_loss``,
    ``total_loss`` (averages across all batches and epochs).
    """
    if len(replay_buffer) < batch_size:
        return {"policy_loss": 0.0, "value_loss": 0.0, "margin_loss": 0.0,
                "total_loss": 0.0, "sign_acc": 0.0}

    network.train()

    buffer_list = list(replay_buffer)
    n = len(buffer_list)

    epoch_losses: list[dict[str, float]] = []

    epoch_iter = tqdm(range(epochs), desc=desc, unit="epoch") if desc else range(epochs)
    for _ in epoch_iter:
        ep_policy   = 0.0
        ep_value    = 0.0
        ep_margin   = 0.0
        ep_sign_acc = 0.0
        ep_batches  = 0

        indices = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            replay_idx = indices[start:start + batch_size]

            base_obs    = np.array([buffer_list[i][0] for i in replay_idx])
            base_policy = np.array([buffer_list[i][1] for i in replay_idx])
            base_value  = np.array([buffer_list[i][2] for i in replay_idx], dtype=np.float32)
            base_margin = np.array(
                [buffer_list[i][3] if len(buffer_list[i]) > 3 else 0.0
                 for i in replay_idx], dtype=np.float32)
            base_has_m  = np.array(
                [1.0 if len(buffer_list[i]) > 3 else 0.0 for i in replay_idx],
                dtype=np.float32)

            # Expand to all 8 D4 symmetries: each example appears in the
            # batch under every rotation/reflection.  Policy actions are
            # permuted to match; value is invariant.
            inv_perms = (SYMMETRY_INV_PERMS_49 if base_policy.shape[1] == 49
                         else SYMMETRY_INV_PERMS)
            obs_chunks    = [apply_obs_symmetry(base_obs, k) for k in range(8)]
            policy_chunks = [base_policy[:, inv_perms[k]] for k in range(8)]
            batch_obs    = np.ascontiguousarray(np.concatenate(obs_chunks,    axis=0))
            batch_policy = np.concatenate(policy_chunks, axis=0)
            batch_value  = np.tile(base_value, 8)
            batch_margin = np.tile(base_margin, 8)  # margin is D4-invariant
            batch_has_m  = np.tile(base_has_m, 8)

            t_obs    = torch.from_numpy(batch_obs).to(device)
            t_policy = torch.from_numpy(batch_policy).to(device)
            t_value  = torch.from_numpy(batch_value).to(device).unsqueeze(-1)
            t_margin = torch.from_numpy(batch_margin).to(device).unsqueeze(-1)
            t_has_m  = torch.from_numpy(batch_has_m).to(device).unsqueeze(-1)

            optimizer.zero_grad()
            pred_logits, pred_value, pred_margin = network(t_obs)

            log_probs   = F.log_softmax(pred_logits, dim=-1)
            has_policy  = t_policy.sum(dim=-1) > 1e-6
            if has_policy.any():
                policy_loss = -torch.sum(t_policy[has_policy] * log_probs[has_policy], dim=-1).mean()
            else:
                policy_loss = t_policy.sum() * 0.0  # zero with correct device/grad_fn
            value_loss  = F.mse_loss(pred_value, t_value)
            # Masked MSE: only examples that carry a margin target contribute.
            m_sq        = (pred_margin - t_margin) ** 2 * t_has_m
            margin_loss = m_sq.sum() / t_has_m.sum().clamp(min=1.0)
            entropy     = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1).mean()

            (policy_loss + value_coef * value_loss + margin_coef * margin_loss
             - entropy_coeff * entropy).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                sign_acc = (pred_value.sign() == t_value.sign()).float().mean().item()

            ep_policy   += policy_loss.item()
            ep_value    += value_loss.item()
            ep_margin   += margin_loss.item()
            ep_sign_acc += sign_acc
            ep_batches  += 1

        if ep_batches > 0:
            epoch_losses.append({
                "policy_loss": ep_policy   / ep_batches,
                "value_loss":  ep_value    / ep_batches,
                "margin_loss": ep_margin   / ep_batches,
                "sign_acc":    ep_sign_acc / ep_batches,
            })

    if not epoch_losses:
        return {"policy_loss": 0.0, "value_loss": 0.0, "margin_loss": 0.0,
                "total_loss": 0.0, "sign_acc": 0.0}

    avg_pol  = float(np.mean([e["policy_loss"] for e in epoch_losses]))
    avg_val  = float(np.mean([e["value_loss"]  for e in epoch_losses]))
    avg_marg = float(np.mean([e["margin_loss"] for e in epoch_losses]))
    avg_sign = float(np.mean([e["sign_acc"]    for e in epoch_losses]))
    return {
        "policy_loss":  avg_pol,
        "value_loss":   avg_val,
        "margin_loss":  avg_marg,
        "total_loss":   avg_pol + avg_val,
        "sign_acc":     avg_sign,
    }
