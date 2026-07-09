"""
Behavioral cloning (BC) pre-training for AlphaZero MCTS.

Generates supervised demonstrations by having minimax play against itself,
then trains the network to imitate those moves before self-play begins.
"""
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move, soft_policy_from_mm,
    apply_obs_symmetry, SYMMETRY_PERMS, SYMMETRY_PERMS_49, ACTION_TO_DEST,
)


# ---------------------------------------------------------------------------
# BC data generation
# ---------------------------------------------------------------------------

def _worker_bc_game(args: tuple) -> list:
    """
    Play one BC game in a worker process and return augmented training examples.
    Pure CPU - no network involved.

    Non-expert moves use a shallow minimax (blunder_depth) rather than pure
    random selection.  Depth-based blunders are more coherent than random moves
    - they look like plausible-but-suboptimal play, produce cleaner value
    targets, and better represent the kind of imperfect positions the network
    will encounter during self-play.  A small random_rate is kept on top for
    positions that never arise in structured play at all.

    If two_stage is True, policy targets are 49-dim destination one-hots
    (marginalized from the expert 1225-action via ACTION_TO_DEST).
    """
    minimax_depth, noise, blunder_depth, random_rate, two_stage = args
    board = new_board()
    turn = bool(np.random.randint(2))
    game_examples = []
    move_count = 0

    # Random prefix to diversify starting positions across all game stages.
    # Range covers the full typical self-play game length so BC positions
    # are representative of what an untrained model will encounter.
    num_random = np.random.randint(0, 101)
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
        if two_stage:
            dest = int(ACTION_TO_DEST[action])
            for k in range(8):
                aug_obs  = np.ascontiguousarray(apply_obs_symmetry(obs_batch, k)[0])
                aug_dest = int(SYMMETRY_PERMS_49[k][dest])
                policy_target = np.zeros(49, dtype=np.float32)
                policy_target[aug_dest] = 1.0
                examples.append((aug_obs, policy_target, value_target))
        else:
            for k in range(8):
                aug_obs    = np.ascontiguousarray(apply_obs_symmetry(obs_batch, k)[0])
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
    two_stage: bool = False,
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
    num_workers = min(4, num_games)
    task_args = [(minimax_depth, noise, blunder_depth, random_rate, two_stage)] * num_games

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
    value_only: bool = False,
) -> None:
    """
    Train *network* on minimax demonstrations before MCTS self-play begins.

    Uses a fresh Adam optimizer so the caller's optimizer/scheduler state is
    not affected.  BC can use a higher LR than self-play (1e-3 is fine here)
    because the labels are clean one-hot minimax moves.

    value_only : if True, skip the policy loss entirely.  Use this when BC
        data is generated at a high minimax depth where policy targets would be
        too sharp and suppress exploration.  Value-only BC bootstraps the value
        head without biasing the policy prior.
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
            pred_logits, pred_value, _ = network(batch_obs)
            value_loss = F.mse_loss(pred_value, batch_value)
            if value_only:
                loss = value_loss
                policy_loss = torch.tensor(0.0)
            else:
                log_probs   = F.log_softmax(pred_logits, dim=-1)
                policy_loss = -torch.sum(batch_policy * log_probs, dim=-1).mean()
                loss = policy_loss + value_loss
            loss.backward()
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


# ---------------------------------------------------------------------------
# BC warmup (soft-policy, no D4 augmentation - used before self-play begins)
# ---------------------------------------------------------------------------

BC_WARMUP_DEPTH  = 3
BC_WARMUP_TEMP   = 0.075   # concentrate policy mass on best moves
BC_RANDOM_OPENS  = 2       # random moves per side at game start
BC_BLUNDER_RATE  = 0.02    # probability of random legal move instead of MM best
MAX_MOVES        = 500     # truncation guard


def _bc_warmup_game(args: tuple) -> list:
    """Play one MM vs MM game and return soft-policy training examples.

    Policy targets are soft distributions from score_root_moves (not one-hot),
    value targets are game outcomes from the active player's perspective.
    """
    game_idx, depth = args
    rng   = np.random.default_rng(game_idx)
    board = new_board()
    turn  = True
    history: list[tuple] = []
    outcome = 0.0
    move_count = 0

    for _ in range(MAX_MOVES):
        is_t, val = check_terminal(board, turn)
        if is_t:
            outcome = (float(val) if turn else -float(val)) if val is not None else 0.0
            break

        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            outcome = -1.0
            break

        in_opening = move_count < BC_RANDOM_OPENS * 2
        if in_opening or rng.random() < BC_BLUNDER_RATE:
            action = int(rng.choice(legal))
        else:
            action = find_best_move(board.tobytes(), depth, turn)
            if action in (-1, 1225):
                action = int(rng.choice(legal))

        obs    = board_to_obs(board, turn)
        policy = soft_policy_from_mm(board, depth, turn, BC_WARMUP_TEMP)
        history.append((obs, policy, turn))
        board  = apply_move(board, action, turn)
        turn   = not turn
        move_count += 1

    return [
        (obs, policy, outcome if was_blue else -outcome)
        for obs, policy, was_blue in history
    ]


def generate_bc_warmup_data(
    num_games: int,
    depth: int = BC_WARMUP_DEPTH,
    cache_path: str | None = None,
) -> list:
    """
    Generate behaviour-cloning examples from MM vs MM self-play.

    Each position becomes one example with a soft MM policy target and the
    game outcome as value.  No D4 augmentation - targets are already
    well-distributed across positions.

    Games run in a thread pool (ctypes releases the GIL for true CPU
    parallelism).  If *cache_path* is given, tries to load from disk first
    and saves after generation.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading BC data from cache: {cache_path}")
        data = np.load(cache_path)
        examples = list(zip(data['obs'], data['policy'], data['value'].tolist()))
        print(f"  Loaded {len(examples)} BC examples")
        return examples

    n_workers = max(1, (os.cpu_count() or 4) - 1)
    examples: list = []
    pbar = tqdm(total=num_games, desc=f"BC warmup (MM{depth})", unit="game")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for game_examples in ex.map(_bc_warmup_game, [(i, depth) for i in range(num_games)]):
            examples.extend(game_examples)
            pbar.update(1)
            pbar.set_postfix(examples=len(examples))
    pbar.close()
    print(f"  BC warmup: {num_games} games -> {len(examples)} examples")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            obs=np.stack([e[0] for e in examples]),
            policy=np.stack([e[1] for e in examples]),
            value=np.array([e[2] for e in examples], dtype=np.float32),
        )
        print(f"  Saved BC cache: {cache_path}")

    return examples
