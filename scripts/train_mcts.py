"""
AlphaZero-style self-play training for Microscope board game.

Trains a dual-head neural network (policy + value) via MCTS self-play:
1. Generate games using MCTS-guided self-play
2. Train network on (board, policy_target, value_target) examples
3. Evaluate against minimax baseline
4. Repeat

Usage:
    python scripts/train_mcts.py

    # Resume from checkpoint:
    python scripts/train_mcts.py --checkpoint models/mcts/iter_050.pt
"""
import argparse
import multiprocessing
import os
import time
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from lib.dual_network import DualHeadNetwork
from lib.mcts import MCTS
from lib.t7g import new_board, apply_move, check_terminal, board_to_obs
from lib.t7g import find_best_move, count_cells, action_masks
from lib.t7g import apply_obs_symmetry, SYMMETRY_INV_PERMS, SYMMETRY_PERMS


# ============================================================
# Configuration
# ============================================================

NUM_ITERATIONS = 200
GAMES_PER_ITERATION = 50
MCTS_SIMULATIONS = 750
EVAL_SIMULATIONS = 750
BATCH_SIZE = 256
EPOCHS_PER_ITERATION = 5
REPLAY_BUFFER_SIZE = 25_000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE_FRACTION = 0.35  # fraction of avg game length to use temperature>0
TEMPERATURE_MIN      = 10    # floor: always at least this many stochastic moves
C_PUCT = 1.5                 # PUCT exploration constant (lower = more exploitation)
DIRICHLET_ALPHA = 0.2        # root noise alpha (~10/branching_factor; avg ~55 legal moves)
EVAL_INTERVAL = 5            # evaluate every N iterations
EVAL_GAMES = 50
CHECKPOINT_INTERVAL = 10
CHECKPOINT_DIR = "models/mcts"
BC_GAMES = 500               # minimax self-play games for behavioral cloning pre-train
BC_DEPTH = 4                 # minimax depth used as the "expert" for BC labels
BC_EPOCHS = 10               # training epochs over the BC dataset
GATE_GAMES = 100             # games per gate evaluation (new vs best network)
GATE_THRESHOLD = 0.45        # reject if new network wins < this fraction
GATE_SIMULATIONS = 250       # sims/move for gate games (fast — just needs to be discriminating)


# ============================================================
# Self-play game generation
# ============================================================

def self_play_game(mcts: MCTS, temperature_threshold: int = 15):
    """
    Play one game via MCTS self-play, collecting training examples.

    Returns:
        examples: list of (obs, policy_target, turn) tuples
        winner: +1.0 if Blue wins, -1.0 if Green wins, 0.0 for draw
        move_count: number of moves played
        elapsed: wall time in seconds
        truncated: True if game ended via the hard move-count cap
    """
    board = new_board()
    # Randomise who moves first so the training data is not systematically
    # biased toward whichever colour has the first-mover advantage.
    turn = bool(np.random.randint(2))
    examples = []
    move_count = 0
    truncated = False
    board_history = {}  # state_key -> visit count for repetition detection
    legal_move_counts = []  # branching factor sample per position
    game_start = time.time()

    while True:
        # 3-fold repetition: same board + same player to move = looping
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

        # Check terminal
        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            # terminal_value is from current player's perspective
            # Convert to Blue's perspective for consistent value targets
            assert terminal_value is not None
            winner = terminal_value if turn else -terminal_value
            break

        # If current player has no moves, pass turn (no training example generated)
        masks = action_masks(board, turn)
        legal_count = int(masks.sum())
        if legal_count == 0:
            mcts.advance_tree(1225)   # resets tree if pass not pre-searched (rare)
            turn = not turn
            continue
        legal_move_counts.append(legal_count)

        # Run MCTS search
        action_probs = mcts.search(board, turn)

        # Store training example (value target assigned after game ends)
        obs = board_to_obs(board, turn)
        examples.append((obs, action_probs, turn))

        # Select action with temperature
        temperature = 1.0 if move_count < temperature_threshold else 0.1
        action = mcts.select_action(action_probs, temperature=temperature)

        # Apply move and advance tree (reuse existing subtree)
        board = apply_move(board, action, turn)
        mcts.advance_tree(action)
        turn = not turn
        move_count += 1

        # Hard safety limit (should rarely trigger after repetition detection)
        if move_count > 200:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            truncated = True
            break

    # Assign value targets: +1 if current player wins, -1 if loses
    training_examples = []
    for obs, policy_target, example_turn in examples:
        # winner is from Blue's perspective
        # Convert to this example's player's perspective
        value_target = winner if example_turn else -winner
        training_examples.append((obs, policy_target, value_target))

    elapsed = time.time() - game_start
    return training_examples, winner, move_count, elapsed, truncated, legal_move_counts


# ============================================================
# Worker functions for parallel self-play (must be module-level
# so they are picklable under Windows' 'spawn' start method)
# ============================================================

# Per-process state: each worker holds its own local network copy
_local_network = None
_base_network = None        # uncompiled — in-place updates preserve CUDA graph validity
_gate_opponent_network = None
_weight_queue = None        # receives weight broadcasts from the main process


def _worker_init(state_dict, weight_queue=None):
    """Load a local network copy (optionally compiled) in each worker process."""
    global _local_network, _base_network, _weight_queue
    torch.set_num_threads(1)  # prevent thread over-subscription across workers
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml  # type: ignore[import-untyped]
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")
    net = DualHeadNetwork(num_actions=1225)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    _base_network = net  # keep uncompiled reference for in-place weight updates
    try:
        net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]
    except Exception:
        pass
    _local_network = net
    _weight_queue = weight_queue


def _gate_worker_init(state_dict_new, state_dict_best):
    """Load candidate and champion networks for gate evaluation."""
    global _local_network, _gate_opponent_network
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml  # type: ignore[import-untyped]
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")

    def _load(sd):
        net = DualHeadNetwork(num_actions=1225)
        net.load_state_dict(sd)
        net.to(device)
        net.eval()
        try:
            net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            pass
        return net

    _local_network = _load(state_dict_new)
    _gate_opponent_network = _load(state_dict_best)


def _worker_play_game(args):
    """Entry point for each worker task."""
    # Apply any pending weight update before starting the game.  load_state_dict
    # with assign=False (default) copies values in-place, so the compiled model's
    # CUDA graph tensor addresses remain valid.
    try:
        new_sd = _weight_queue.get(block=False)  # type: ignore[union-attr]
        _base_network.load_state_dict(new_sd)    # type: ignore[union-attr]
    except Exception:
        pass

    num_simulations, temperature_threshold, c_puct, dirichlet_alpha = args
    mcts = MCTS(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
    )
    return self_play_game(mcts, temperature_threshold=temperature_threshold)


def generate_self_play_data(pool: multiprocessing.Pool,
                            min_non_truncated: int = 50,
                            num_simulations: int = 100,
                            temperature_threshold: int = TEMPERATURE_MIN,
                            num_workers: int = 2):
    """Generate training data from self-play games using a persistent worker pool.

    Workers already hold up-to-date weights (broadcast by main after each
    training step). Truncated games are played but their examples are discarded.
    """
    task_args_single = (num_simulations, temperature_threshold, C_PUCT, DIRICHLET_ALPHA)

    all_examples = []
    blue_wins = 0
    green_wins = 0
    draws = 0
    truncations = 0
    non_truncated_count = 0
    game_moves: list = []
    game_times: list = []
    all_legal_counts: list = []

    pbar = tqdm(desc="Self-play", unit="game", total=min_non_truncated)
    try:
        pending = deque()
        # Seed the pipeline: keep num_workers games in flight at all times
        for _ in range(num_workers):
            pending.append(pool.apply_async(_worker_play_game, (task_args_single,)))

        while non_truncated_count < min_non_truncated:
            examples, winner, moves, gtime, trunc, legal_counts = pending.popleft().get()
            all_legal_counts.extend(legal_counts)
            game_moves.append(moves)
            game_times.append(gtime)
            if trunc:
                truncations += 1
            else:
                all_examples.extend(examples)
                non_truncated_count += 1
                pbar.update(1)
            if winner > 0:
                blue_wins += 1
            elif winner < 0:
                green_wins += 1
            else:
                draws += 1
            pbar.set_postfix(
                examples=len(all_examples),
            )
            # Keep the pipeline full until we have enough
            if non_truncated_count < min_non_truncated:
                pending.append(pool.apply_async(_worker_play_game, (task_args_single,)))
        # Cancel any remaining in-flight tasks by discarding their futures
        for fut in pending:
            fut.cancel() if hasattr(fut, 'cancel') else None
    finally:
        pbar.close()

    avg_branching = float(np.mean(all_legal_counts)) if all_legal_counts else 0.0
    return (all_examples, (blue_wins, green_wins, draws),
            game_moves, game_times, truncations, avg_branching)


# ============================================================
# Network training
# ============================================================

def train_network(network, replay_buffer, optimizer, batch_size=256,
                  epochs=5, device: str | torch.device = 'cpu'):
    """
    Train network on replay buffer data.

    Returns:
        dict with average losses
    """
    if len(replay_buffer) < batch_size:
        return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # Pre-convert to numpy in RAM (fast), then move only one batch at a time to
    # the device to avoid OOMing GPU VRAM with the full 500+ MB policy tensor.
    buffer_list = list(replay_buffer)
    obs_np     = np.array([ex[0] for ex in buffer_list])
    policy_np  = np.array([ex[1] for ex in buffer_list])
    value_np   = np.array([ex[2] for ex in buffer_list], dtype=np.float32)
    n = len(buffer_list)

    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start:start + batch_size]
            k = int(np.random.randint(0, 8))
            batch_obs    = torch.from_numpy(
                np.ascontiguousarray(apply_obs_symmetry(obs_np[idx], k))
            ).to(device)
            batch_policy = torch.from_numpy(policy_np[idx][:, SYMMETRY_INV_PERMS[k]]).to(device)
            batch_value  = torch.from_numpy(value_np[idx]).to(device).unsqueeze(-1)

            optimizer.zero_grad()

            # Forward pass
            pred_logits, pred_value = network(batch_obs)

            # Policy loss: cross-entropy with MCTS policy targets
            # MCTS targets are probability distributions, use KL divergence
            log_probs = F.log_softmax(pred_logits, dim=-1)
            policy_loss = -torch.sum(batch_policy * log_probs, dim=-1).mean()

            # Value loss: MSE between predicted and actual game outcome
            value_loss = F.mse_loss(pred_value, batch_value)

            # Total loss
            loss = policy_loss + value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

    if num_batches == 0:
        return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

    return {
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "total_loss": (total_policy_loss + total_value_loss) / num_batches,
    }


# ============================================================
# Behavioral cloning pre-training
# ============================================================

def _worker_bc_game(args: tuple) -> list:
    """
    Play one BC game in a worker process and return its augmented examples.
    Pure CPU — no network, no InferenceServer needed.
    """
    minimax_depth, noise = args
    board = new_board()
    turn = True
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

        played = int(np.random.choice(legal)) if np.random.random() < noise else expert_action
        board = apply_move(board, played, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            winner = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    # Apply all 8 D4 symmetries (8x augmentation per position)
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


def generate_bc_data(num_games: int = BC_GAMES, minimax_depth: int = BC_DEPTH,
                     noise: float = 0.15) -> list:
    """
    Generate supervised examples by playing minimax against itself (parallel).

    Games are distributed across a process pool (pure CPU — no GPU needed).
    Each position is augmented with all 8 D4 symmetries.

    Returns list of (obs, policy_one_hot, value_target) tuples.
    """
    num_workers = min(8, num_games)
    task_args = [(minimax_depth, noise)] * num_games

    examples = []
    pbar = tqdm(total=num_games, desc="BC data", unit="game")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for game_examples in pool.imap_unordered(_worker_bc_game, task_args):
            examples.extend(game_examples)
            pbar.update(1)
            pbar.set_postfix(examples=len(examples))
    pbar.close()

    return examples


def pretrain_bc(network, bc_data: list, epochs: int = BC_EPOCHS,
                batch_size: int = 256, device: str | torch.device = 'cpu',
                writer: SummaryWriter | None = None) -> None:
    """
    Train network on minimax demonstrations before MCTS self-play begins.

    Uses a fresh Adam optimizer so the main training scheduler is unaffected.
    """
    if not bc_data:
        return

    obs_np    = np.array([ex[0] for ex in bc_data])
    policy_np = np.array([ex[1] for ex in bc_data])
    value_np  = np.array([ex[2] for ex in bc_data], dtype=np.float32)
    n = len(bc_data)

    opt = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            if writer:
                writer.add_scalar("bc/policy_loss", avg_pol, epoch)
                writer.add_scalar("bc/value_loss",  avg_val, epoch)


# ============================================================
# Evaluation (parallel, mirrors the self-play worker pattern)
# ============================================================

def _play_eval_game(mcts: MCTS, minimax_depth: int, noise: float,
                    engine: str, vary_depth: bool, mcts_is_blue: bool) -> float:
    """Play one eval game; return +1 win / -1 loss / 0 draw from MCTS perspective."""
    board = new_board()
    mcts.root = None        # fresh tree each game
    turn = True             # Blue moves first
    board_history: dict = {}
    move_count = 0

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        mcts_turn = (turn == mcts_is_blue)

        if mcts_turn:
            if not np.any(action_masks(board, turn)):
                mcts.advance_tree(1225)
                turn = not turn
                continue
            action_probs = mcts.search(board, turn)
            action = mcts.select_action(action_probs, temperature=0)
            mcts.advance_tree(action)
        else:
            legal = np.where(action_masks(board, turn))[0]
            if len(legal) == 0:
                turn = not turn
                continue
            if np.random.random() < noise:
                action = int(np.random.choice(legal))
            else:
                depth = int(np.random.choice([4, minimax_depth])) if vary_depth else minimax_depth
                # Randomise stauf's internal move_count so the depths[] lookup
                # table cycles through all three %3 slots rather than always
                # landing on slot 1 (the fresh-instance default).
                stauf_mc = int(np.random.randint(0, 3)) if engine == 'stauf' else -1
                action = find_best_move(board.tobytes(), depth, turn, engine, stauf_mc)
                if action in (-1, 1225):
                    turn = not turn
                    continue

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    return blue_result if mcts_is_blue else -blue_result


def _worker_eval_game(args):
    """Entry point for parallel eval workers."""
    num_simulations, minimax_depth, noise, engine, vary_depth, mcts_is_blue = args
    mcts = MCTS(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        dirichlet_epsilon=0.0,
    )
    return _play_eval_game(mcts, minimax_depth, noise, engine, vary_depth, mcts_is_blue)


def evaluate_vs_noisy_minimax(network, minimax_depth=2, noise=0.3,
                              num_games=20, num_simulations=100,
                              engine: str = 'minimax',
                              vary_depth: bool = False):
    """
    Evaluate MCTS agent against an epsilon-greedy minimax opponent in parallel.

    Half the games are played as Blue, half as Green. Each worker holds its
    own local network copy so minimax calls run on multiple CPU cores simultaneously.

    Returns:
        win_rate: fraction of games won by MCTS agent
        results: dict with wins/losses/draws
    """
    num_workers = min(2, num_games)
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

    task_args = [
        (num_simulations, minimax_depth, noise, engine, vary_depth, game_idx % 2 == 0)
        for game_idx in range(num_games)
    ]

    engine_label = "Stauf" if engine == 'stauf' else f"MM-{minimax_depth}"
    label = f"Eval vs {engine_label} (noise={noise:.0%})"
    wins = losses = draws = 0
    pbar = tqdm(total=num_games, desc=label, unit="game", leave=False)

    try:
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(state_dict,),
        ) as pool:
            for mcts_result in pool.imap_unordered(_worker_eval_game, task_args):
                if mcts_result > 0:
                    wins += 1
                elif mcts_result < 0:
                    losses += 1
                else:
                    draws += 1
                pbar.update(1)
                pbar.set_postfix(win_rate=f"{wins / (wins + losses + draws):.0%}")
    finally:
        pbar.close()

    return wins / num_games, {"wins": wins, "losses": losses, "draws": draws}


def _play_net_vs_net_game(mcts_new: MCTS, mcts_best: MCTS, new_is_blue: bool) -> float:
    """Play one game between two MCTS agents. Returns +1 if new wins, -1 if new loses."""
    board = new_board()
    mcts_new.root = None
    mcts_best.root = None
    # Randomise who moves first so gate scores aren't purely determined by
    # first-mover advantage (which locks gate_wr to exactly 0.5 otherwise).
    turn = bool(np.random.randint(2))
    board_history: dict = {}
    move_count = 0

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        new_turn = (turn == new_is_blue)
        mcts_active  = mcts_new if new_turn else mcts_best
        mcts_passive = mcts_best if new_turn else mcts_new

        if not np.any(action_masks(board, turn)):
            mcts_active.advance_tree(1225)
            mcts_passive.advance_tree(1225)
            turn = not turn
            continue

        action_probs = mcts_active.search(board, turn)
        action = mcts_active.select_action(action_probs, temperature=0)
        mcts_active.advance_tree(action)
        mcts_passive.advance_tree(action)

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            break

    return blue_result if new_is_blue else -blue_result


def _worker_gate_game(args):
    """Entry point for gate evaluation workers (new vs best)."""
    num_simulations, new_is_blue = args
    mcts_new = MCTS(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        dirichlet_epsilon=0.0,
    )
    mcts_best = MCTS(
        _gate_opponent_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        dirichlet_epsilon=0.0,
    )
    return _play_net_vs_net_game(mcts_new, mcts_best, new_is_blue)


def evaluate_new_vs_best(network, best_state_dict: dict,
                         num_games: int = GATE_GAMES,
                         num_simulations: int = GATE_SIMULATIONS) -> tuple[float, dict]:
    """
    Gate evaluation: play new_network vs best accepted network.
    Half games as Blue, half as Green.
    Returns (win_rate_for_new, results_dict).
    """
    num_workers = min(2, num_games)
    state_dict_new = {k: v.cpu() for k, v in network.state_dict().items()}
    task_args = [(num_simulations, game_idx % 2 == 0) for game_idx in range(num_games)]

    wins = losses = draws = 0
    pbar = tqdm(total=num_games, desc="Gate (new vs best)", unit="game", leave=False)
    try:
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_gate_worker_init,
            initargs=(state_dict_new, best_state_dict),
        ) as pool:
            for result in pool.imap_unordered(_worker_gate_game, task_args):
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
                pbar.update(1)
                pbar.set_postfix(win_rate=f"{wins / (wins + losses + draws):.0%}")
    finally:
        pbar.close()

    # Score draws as ½ point so they don't count the same as losses.
    score = (wins + 0.5 * draws) / num_games
    return score, {"wins": wins, "losses": losses, "draws": draws}


# ============================================================
# Main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AlphaZero MCTS Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS,
                        help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=GAMES_PER_ITERATION,
                        help="Self-play games per iteration")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS,
                        help="Total training iterations")
    parser.add_argument("--logdir", type=str, default="tblog/mcts",
                        help="TensorBoard log root directory")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (default: timestamp). Logs go to logdir/run-name")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 matmul on Ampere+
    print(f"Device: {device}")

    # Create network
    network = DualHeadNetwork(num_actions=1225).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Resume from checkpoint (weights only — fresh optimizer, full iteration budget)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        network.load_state_dict(checkpoint['network'])
        saved_iter = checkpoint.get('iteration', 0) + 1
        print(f"Loaded weights from iteration {saved_iter}; "
              f"training for {args.iterations} fresh iterations")

    # Create checkpoint directory and TensorBoard writer
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.logdir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Behavioral cloning pre-training (fresh runs only — skipped when resuming)
    if not args.checkpoint:
        print("\nBehavioral cloning pre-training on minimax demonstrations...")
        print(f"  {BC_GAMES} games at minimax depth {BC_DEPTH}, {BC_EPOCHS} epochs")
        bc_data = generate_bc_data(num_games=BC_GAMES, minimax_depth=BC_DEPTH)
        pretrain_bc(network, bc_data, epochs=BC_EPOCHS,
                    batch_size=BATCH_SIZE, device=device, writer=writer)
        print("BC pre-training complete.\n")

    print("=" * 60)
    print("AlphaZero MCTS Training for Microscope")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Games/iteration: {args.games}")
    print(f"Simulations/move: {args.simulations}")
    print(f"Replay buffer: {REPLAY_BUFFER_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs/iteration: {EPOCHS_PER_ITERATION}")
    print("=" * 60)

    # Persistent self-play worker pool — workers are never torn down between
    # iterations. After each training step, fresh weights are broadcast via
    # weight_queue so workers update in-place before their next game.
    NUM_SELF_PLAY_WORKERS = 3
    weight_queue: multiprocessing.Queue = multiprocessing.Queue()
    init_sd = {k: v.clone().cpu() for k, v in network.state_dict().items()}
    self_play_pool = multiprocessing.Pool(
        processes=NUM_SELF_PLAY_WORKERS,
        initializer=_worker_init,
        initargs=(init_sd, weight_queue),
    )

    avg_game_length: float = 60.0  # seed estimate; updated each iteration
    best_network_state: dict = init_sd  # already a cpu clone

    iter_pbar = tqdm(range(args.iterations),
                     desc="Training", unit="iter",
                     total=args.iterations)
    for iteration in iter_pbar:
        iter_start = time.time()

        iter_pbar.set_description(f"Iter {iteration + 1}/{args.iterations}")

        # 1. Generate self-play data
        temp_threshold = max(TEMPERATURE_MIN, int(TEMPERATURE_FRACTION * avg_game_length))
        print(f"\nGenerating until {args.games} non-truncated games "
              f"({args.simulations} sims/move, temp threshold={temp_threshold})...")
        gen_start = time.time()
        sp_result = generate_self_play_data(
            self_play_pool, min_non_truncated=args.games, num_simulations=args.simulations,
            temperature_threshold=temp_threshold, num_workers=NUM_SELF_PLAY_WORKERS
        )
        examples, (bw, gw, dr), game_moves, game_times, truncations, avg_branching = sp_result
        gen_time = time.time() - gen_start
        moves_arr = np.array(game_moves)
        avg_moves = float(moves_arr.mean())
        avg_game_length = avg_moves  # update for next iteration's threshold
        med_moves = float(np.median(moves_arr))
        std_moves = float(moves_arr.std())
        avg_gtime = sum(game_times) / len(game_times)
        trunc_pct = 100.0 * truncations / len(game_moves)
        print(f"  Generated {len(examples)} examples in {gen_time:.1f}s")
        print(f"  Results: Blue {bw} / Green {gw} / Draw {dr}")
        print(f"  Game length: avg {avg_moves:.1f}  median {med_moves:.1f}  "
              f"std {std_moves:.1f}  (min {moves_arr.min()}, max {moves_arr.max()})")
        print(f"  Truncated:   {truncations}/{len(game_moves)} ({trunc_pct:.1f}%)")
        print(f"  Game time:   avg {avg_gtime:.1f}s "
              f"(min {min(game_times):.1f}s, max {max(game_times):.1f}s)")

        step = iteration + 1
        writer.add_scalar("self_play/examples_generated", len(examples), step)
        writer.add_scalar("timing/gen_seconds", gen_time, step)
        writer.add_scalar("self_play/avg_game_moves", avg_moves, step)
        writer.add_scalar("self_play/median_game_moves", med_moves, step)
        writer.add_scalar("self_play/std_game_moves", std_moves, step)
        writer.add_scalar("self_play/min_game_moves", int(moves_arr.min()), step)
        writer.add_scalar("self_play/max_game_moves", int(moves_arr.max()), step)
        writer.add_scalar("self_play/truncation_pct", trunc_pct, step)
        writer.add_scalar("self_play/temp_threshold", temp_threshold, step)
        writer.add_scalar("self_play/avg_branching_factor", avg_branching, step)
        print(f"  Avg branching:  {avg_branching:.1f} legal moves/position")
        total_sims = int(moves_arr.sum()) * args.simulations
        sims_per_sec = total_sims / gen_time if gen_time > 0 else 0.0
        print(f"  Throughput:  {sims_per_sec:.0f} sims/sec")
        writer.add_scalar("timing/avg_game_seconds", avg_gtime, step)
        writer.add_scalar("timing/min_game_seconds", min(game_times), step)
        writer.add_scalar("timing/max_game_seconds", max(game_times), step)
        writer.add_scalar("timing/sims_per_sec", sims_per_sec, step)

        is_gate_iter = (iteration + 1) % EVAL_INTERVAL == 0

        # 2. Add to replay buffer — always, so training uses fresh data.
        # On gate failure we revert the network weights but keep the examples;
        # one iteration of sub-optimal data in a 25k buffer is negligible.
        replay_buffer.extend(examples)
        print(f"  Replay buffer: {len(replay_buffer)} examples")
        writer.add_scalar("self_play/buffer_size", len(replay_buffer), step)

        # 3. Train network
        print(f"\nTraining ({EPOCHS_PER_ITERATION} epochs, "
              f"batch size {BATCH_SIZE})...")
        train_start = time.time()
        losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITERATION,
            device=device
        )
        train_time = time.time() - train_start
        print(f"  Policy loss: {losses['policy_loss']:.4f}")
        print(f"  Value loss:  {losses['value_loss']:.4f}")
        print(f"  Total loss:  {losses['total_loss']:.4f}")
        print(f"  Train time:  {train_time:.1f}s")

        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar("train/policy_loss", losses['policy_loss'], step)
        writer.add_scalar("train/value_loss",  losses['value_loss'],  step)
        writer.add_scalar("train/total_loss",  losses['total_loss'],  step)
        writer.add_scalar("train/lr",          current_lr,            step)
        writer.add_scalar("timing/train_seconds", train_time, step)

        # Broadcast updated weights so workers pick them up before the next game
        new_sd = {k: v.cpu() for k, v in network.state_dict().items()}
        for _ in range(NUM_SELF_PLAY_WORKERS):
            weight_queue.put(new_sd)

        # 4. Gate + evaluate (every EVAL_INTERVAL)
        if is_gate_iter:
            # Gate: compare trained candidate vs last accepted network
            gate_wr, gate_results = evaluate_new_vs_best(network, best_network_state)
            print(f"\n  Gate ({gate_wr:.0%} vs best, threshold {GATE_THRESHOLD:.0%}): "
                  f"W:{gate_results['wins']} L:{gate_results['losses']} "
                  f"D:{gate_results['draws']}")
            writer.add_scalar("gate/win_rate", gate_wr, step)

            if gate_wr >= GATE_THRESHOLD:
                print("  Gate PASSED — accepting network")
                best_network_state = {k: v.clone().cpu()
                                      for k, v in network.state_dict().items()}
                writer.add_scalar("gate/accepted", 1, step)
            else:
                print("  Gate FAILED — reverting to best network")
                network.load_state_dict(best_network_state)
                # Drain the rejected weights already in the queue (pushed after
                # training above) before pushing the reverted weights, so workers
                # don't consume the rejected candidate first (queue is FIFO).
                while not weight_queue.empty():
                    try:
                        weight_queue.get_nowait()
                    except Exception:
                        break
                for _ in range(NUM_SELF_PLAY_WORKERS):
                    weight_queue.put(best_network_state)
                writer.add_scalar("gate/accepted", 0, step)

            print("\nEvaluating vs minimax...")

            # vs noisy minimax depth-3 (10% blunder rate) — every EVAL_INTERVAL
            wr_mm3n, results_mm3n = evaluate_vs_noisy_minimax(
                network, minimax_depth=3, noise=0.1, num_games=EVAL_GAMES,
                num_simulations=EVAL_SIMULATIONS
            )
            print(f"  vs MM-3 (10%):  {wr_mm3n:.0%} "
                  f"(W:{results_mm3n['wins']} L:{results_mm3n['losses']} "
                  f"D:{results_mm3n['draws']})")
            writer.add_scalar("eval/win_rate_mm3_noise10", wr_mm3n, step)

            # vs Stauf (10 games, just to watch) — every EVAL_INTERVAL * 2
            if (iteration + 1) % (EVAL_INTERVAL * 2) == 0:
                wr_stauf, results_stauf = evaluate_vs_noisy_minimax(
                    network, minimax_depth=5, noise=0.0, num_games=10,
                    num_simulations=EVAL_SIMULATIONS, engine='stauf', vary_depth=True
                )
                print(f"  vs Stauf (d=5): {wr_stauf:.0%} "
                      f"(W:{results_stauf['wins']} L:{results_stauf['losses']} "
                      f"D:{results_stauf['draws']})")
                writer.add_scalar("eval/win_rate_stauf_d5", wr_stauf, step)

            # vs pure minimax depth-3 — every EVAL_INTERVAL * 2
            if (iteration + 1) % (EVAL_INTERVAL * 2) == 0:
                wr_mm3, results_mm3 = evaluate_vs_noisy_minimax(
                    network, minimax_depth=3, noise=0.0, num_games=EVAL_GAMES,
                    num_simulations=EVAL_SIMULATIONS
                )
                print(f"  vs MM-3 (pure): {wr_mm3:.0%} "
                      f"(W:{results_mm3['wins']} L:{results_mm3['losses']} "
                      f"D:{results_mm3['draws']})")
                writer.add_scalar("eval/win_rate_mm3_pure", wr_mm3, step)

            # vs pure minimax depth-5 — every EVAL_INTERVAL * 4
            if (iteration + 1) % (EVAL_INTERVAL * 4) == 0:
                wr_mm5, results_mm5 = evaluate_vs_noisy_minimax(
                    network, minimax_depth=5, noise=0.0, num_games=EVAL_GAMES,
                    num_simulations=EVAL_SIMULATIONS
                )
                print(f"  vs MM-5 (pure): {wr_mm5:.0%} "
                      f"(W:{results_mm5['wins']} L:{results_mm5['losses']} "
                      f"D:{results_mm5['draws']})")
                writer.add_scalar("eval/win_rate_mm5_pure", wr_mm5, step)

        # 5. Checkpoint
        if (iteration + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"iter_{iteration + 1:04d}.pt"
            )
            torch.save({
                'iteration': iteration,
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"\n  Checkpoint saved: {ckpt_path}")

        iter_time = time.time() - iter_start
        iter_pbar.set_postfix(
            loss=f"{losses['total_loss']:.3f}",
            buf=len(replay_buffer),
            time=f"{iter_time:.0f}s"
        )

    self_play_pool.terminate()
    self_play_pool.join()
    writer.close()

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    torch.save({
        'iteration': args.iterations - 1,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == "__main__":
    main()
