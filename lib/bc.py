"""
Behavioral cloning (BC) pre-training for AlphaZero MCTS.

Generates supervised demonstrations by having minimax play against itself,
then trains the network to imitate those moves before self-play begins.
"""
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move,
    apply_obs_symmetry, SYMMETRY_PERMS,
)


# ---------------------------------------------------------------------------
# BC data generation
# ---------------------------------------------------------------------------

def _worker_bc_game(args: tuple) -> list:
    """
    Play one BC game in a worker process and return augmented training examples.
    Pure CPU — no network involved.

    Non-expert moves use a shallow minimax (blunder_depth) rather than pure
    random selection.  Depth-based blunders are more coherent than random moves
    — they look like plausible-but-suboptimal play, produce cleaner value
    targets, and better represent the kind of imperfect positions the network
    will encounter during self-play.  A small random_rate is kept on top for
    positions that never arise in structured play at all.
    """
    minimax_depth, noise, blunder_depth, random_rate = args
    board = new_board()
    turn = bool(np.random.randint(2))
    game_examples = []
    move_count = 0

    # Play 2-3 random opening moves to diversify starting positions
    num_random = np.random.randint(2, 4)
    for _ in range(num_random):
        is_terminal, _ = check_terminal(board, turn)
        if is_terminal:
            break
        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            turn = not turn
            continue
        board = apply_move(board, int(np.random.choice(legal)), turn)
        turn = not turn

    winner = 0.0  # fallback; will be overwritten
    while True:
        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            winner = terminal_value if turn else -terminal_value
            break

        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            turn = not turn
            continue

        expert_action = find_best_move(board.tobytes(), minimax_depth, turn)
        if expert_action in (-1, 1225):
            expert_action = int(np.random.choice(legal))

        obs = board_to_obs(board, turn)
        game_examples.append((obs, expert_action, turn))

        r = np.random.random()
        if r < random_rate:
            played = int(np.random.choice(legal))       # true random for coverage
        elif r < noise:
            blunder = find_best_move(board.tobytes(), blunder_depth, turn)
            played = blunder if blunder not in (-1, 1225) else expert_action  # shallow blunder
        else:
            played = expert_action
        board = apply_move(board, played, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    # Apply all 8 D4 symmetries (8× augmentation per position)
    examples = []
    for obs, action, ex_turn in game_examples:
        value_target = float(winner if ex_turn else -winner)
        obs_batch = obs[np.newaxis]
        for k in range(8):
            aug_obs = np.ascontiguousarray(apply_obs_symmetry(obs_batch, k)[0])
            aug_action = int(SYMMETRY_PERMS[k][action])
            policy_target = np.zeros(1225, dtype=np.float32)
            policy_target[aug_action] = 1.0
            examples.append((aug_obs, policy_target, value_target))
    return examples


def generate_bc_data(
    num_games: int,
    minimax_depth: int,
    noise: float = 0.10,
    blunder_depth: int = 2,
    random_rate: float = 0.02,
) -> list:
    """
    Generate supervised examples by playing minimax against itself (parallel).

    Non-expert moves are drawn in priority order:
      1. With probability *random_rate*: uniformly random legal move
      2. With probability *noise - random_rate*: depth-*blunder_depth* minimax move
      3. Otherwise: full-depth expert move

    Each position is augmented with all 8 D4 symmetries.

    Returns
    -------
    list of (obs, policy_one_hot, value_target) tuples.
    """
    num_workers = min(16, num_games)
    task_args = [(minimax_depth, noise, blunder_depth, random_rate)] * num_games

    examples: list = []
    pbar = tqdm(total=num_games, desc="BC data", unit="game")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for game_examples in pool.imap_unordered(_worker_bc_game, task_args):
            examples.extend(game_examples)
            pbar.update(1)
            pbar.set_postfix(examples=len(examples))
    pbar.close()
    return examples


# ---------------------------------------------------------------------------
# BC pre-training
# ---------------------------------------------------------------------------

def pretrain_bc(
    network: torch.nn.Module,
    bc_data: list,
    epochs: int = 10,
    batch_size: int = 256,
    device: str | torch.device = 'cpu',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    writer=None,
) -> None:
    """
    Train *network* on minimax demonstrations before MCTS self-play begins.

    Uses a fresh Adam optimizer so the caller's optimizer/scheduler state is
    not affected.  BC can use a higher LR than self-play (1e-3 is fine here)
    because the labels are clean one-hot minimax moves.
    """
    if not bc_data:
        return

    obs_np    = np.array([ex[0] for ex in bc_data])
    policy_np = np.array([ex[1] for ex in bc_data])
    value_np  = np.array([ex[2] for ex in bc_data], dtype=np.float32)
    n = len(bc_data)

    opt = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    network.train()

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_pol, total_val, num_batches = 0.0, 0.0, 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start:start + batch_size]
            batch_obs    = torch.from_numpy(obs_np[idx]).to(device)
            batch_policy = torch.from_numpy(policy_np[idx]).to(device)
            batch_value  = torch.from_numpy(value_np[idx]).to(device).unsqueeze(-1)

            opt.zero_grad()
            pred_logits, pred_value = network(batch_obs)
            log_probs   = F.log_softmax(pred_logits, dim=-1)
            policy_loss = -torch.sum(batch_policy * log_probs, dim=-1).mean()
            value_loss  = F.mse_loss(pred_value, batch_value)
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            opt.step()

            total_pol += policy_loss.item()
            total_val += value_loss.item()
            num_batches += 1

        if num_batches:
            avg_pol = total_pol / num_batches
            avg_val = total_val / num_batches
            print(f"    Epoch {epoch + 1}/{epochs}:  policy={avg_pol:.4f}  value={avg_val:.4f}")
            if writer is not None:
                writer.add_scalar("bc/policy_loss", avg_pol, epoch)
                writer.add_scalar("bc/value_loss",  avg_val, epoch)
