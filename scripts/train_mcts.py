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
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dual_network import DualHeadNetwork
from lib.mcts import MCTS
from lib.t7g import new_board, apply_move, check_terminal, board_to_obs
from lib.t7g import find_best_move, count_cells, show_board, action_masks


# ============================================================
# Configuration
# ============================================================

NUM_ITERATIONS = 200
GAMES_PER_ITERATION = 50
MCTS_SIMULATIONS = 100
BATCH_SIZE = 256
EPOCHS_PER_ITERATION = 5
REPLAY_BUFFER_SIZE = 100_000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE_THRESHOLD = 15   # moves before switching to greedy
C_PUCT = 2.0                 # PUCT exploration constant (lower = more exploitation)
DIRICHLET_ALPHA = 0.1        # root noise concentration (~10/avg_legal_moves)
EVAL_INTERVAL = 5            # evaluate every N iterations
EVAL_GAMES = 20
CHECKPOINT_INTERVAL = 10
CHECKPOINT_DIR = "models/mcts"


# ============================================================
# Self-play game generation
# ============================================================

def self_play_game(network, num_simulations=100, temperature_threshold=15,
                   c_puct=2.0, dirichlet_alpha=0.1):
    """
    Play one game via MCTS self-play, collecting training examples.

    Returns:
        examples: list of (obs, policy_target, turn) tuples
        winner: +1.0 if Blue wins, -1.0 if Green wins, 0.0 for draw
    """
    mcts = MCTS(network, num_simulations=num_simulations,
                c_puct=c_puct, dirichlet_alpha=dirichlet_alpha)
    board = new_board()
    turn = True  # Blue starts
    examples = []
    move_count = 0
    board_history = {}  # state_key -> visit count for repetition detection

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
            winner = terminal_value if turn else -terminal_value
            break

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
            break

    # Assign value targets: +1 if current player wins, -1 if loses
    training_examples = []
    for obs, policy_target, example_turn in examples:
        # winner is from Blue's perspective
        # Convert to this example's player's perspective
        value_target = winner if example_turn else -winner
        training_examples.append((obs, policy_target, value_target))

    return training_examples, winner


# ============================================================
# Worker functions for parallel self-play (must be module-level
# so they are picklable under Windows' 'spawn' start method)
# ============================================================

# Per-process network instance, initialised by _worker_init
_worker_network = None


def _worker_init(state_dict):
    """Initialise each worker process with its own copy of the network."""
    global _worker_network
    torch.set_num_threads(1)  # prevent thread over-subscription across workers
    if torch.cuda.is_available():          # covers both NVIDIA CUDA and AMD ROCm
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device("cpu")
    _worker_network = DualHeadNetwork(num_actions=1225)
    _worker_network.load_state_dict(state_dict)
    _worker_network.to(device)
    _worker_network.eval()


def _worker_play_game(args):
    """Entry point for each worker task."""
    num_simulations, temperature_threshold, c_puct, dirichlet_alpha = args
    return self_play_game(
        _worker_network,
        num_simulations=num_simulations,
        temperature_threshold=temperature_threshold,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
    )


def generate_self_play_data(network, num_games, num_simulations=100,
                            num_workers=None):
    """Generate training data from multiple self-play games in parallel."""
    if num_workers is None:
        # Cap at 3 workers — each torch worker commits ~6 GB of virtual address
        # space on Windows. we're bottlenecked by inference anyway. 
        num_workers = min(3, num_games)

    # CPU state dict is picklable; CUDA tensors are not
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    task_args = [(num_simulations, TEMPERATURE_THRESHOLD, C_PUCT, DIRICHLET_ALPHA)] * num_games

    all_examples = []
    blue_wins = 0
    green_wins = 0
    draws = 0

    pbar = tqdm(total=num_games, desc="Self-play", unit="game")
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(state_dict,),
    ) as pool:
        for examples, winner in pool.imap_unordered(_worker_play_game, task_args):
            all_examples.extend(examples)
            if winner > 0:
                blue_wins += 1
            elif winner < 0:
                green_wins += 1
            else:
                draws += 1
            pbar.update(1)
            pbar.set_postfix(
                examples=len(all_examples),
                W=blue_wins, L=green_wins, D=draws,
            )
    pbar.close()

    return all_examples, (blue_wins, green_wins, draws)


# ============================================================
# Network training
# ============================================================

def train_network(network, replay_buffer, optimizer, batch_size=256,
                  epochs=5, device='cpu'):
    """
    Train network on replay buffer data.

    Returns:
        dict with average losses
    """
    if len(replay_buffer) < batch_size:
        return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

    network.train()
    total_policy_loss = 0
    total_value_loss = 0
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
            batch_obs    = torch.from_numpy(obs_np[idx]).to(device)
            batch_policy = torch.from_numpy(policy_np[idx]).to(device)
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
# Evaluation
# ============================================================

def evaluate_vs_minimax(network, minimax_depth=2, num_games=20,
                        num_simulations=100):
    """
    Evaluate MCTS agent (as Blue) against minimax opponent (as Green).

    Returns:
        win_rate: fraction of games won by MCTS agent
        results: dict with wins/losses/draws
    """
    mcts = MCTS(network, num_simulations=num_simulations,
                dirichlet_epsilon=0.0)  # No exploration noise for eval
    wins = 0
    losses = 0
    draws = 0

    pbar = tqdm(range(num_games), desc=f"Eval vs MM-{minimax_depth}", unit="game", leave=False)
    for _ in pbar:
        board = new_board()
        turn = True  # Blue = MCTS agent
        move_count = 0
        board_history = {}

        while True:
            state_key = board.tobytes() + bytes([turn])
            board_history[state_key] = board_history.get(state_key, 0) + 1
            if board_history[state_key] >= 3:
                blue, green = count_cells(board)
                blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
                break

            is_terminal, terminal_value = check_terminal(board, turn)
            if is_terminal:
                # terminal_value from current player perspective
                blue_result = terminal_value if turn else -terminal_value
                break

            if turn:
                # MCTS agent (Blue)
                action_probs = mcts.search(board, turn)
                action = mcts.select_action(action_probs, temperature=0)
            else:
                # Minimax opponent (Green)
                board_bytes = board.tobytes()
                action = find_best_move(board_bytes, minimax_depth, False)
                if action in [-1, 1225]:
                    # Minimax has no moves - MCTS wins
                    blue_result = 1.0
                    break

            board = apply_move(board, action, turn)
            turn = not turn
            move_count += 1

            if move_count > 200:
                blue, green = count_cells(board)
                blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
                break

        if blue_result > 0:
            wins += 1
        elif blue_result < 0:
            losses += 1
        else:
            draws += 1

        pbar.set_postfix(W=wins, L=losses, D=draws)

    win_rate = wins / num_games
    return win_rate, {"wins": wins, "losses": losses, "draws": draws}


def evaluate_vs_random(network, num_games=20, num_simulations=50):
    """Quick evaluation against random opponent."""
    mcts = MCTS(network, num_simulations=num_simulations,
                dirichlet_epsilon=0.0)
    wins = 0

    pbar = tqdm(range(num_games), desc="Eval vs Random", unit="game", leave=False)
    for i in pbar:
        board = new_board()
        turn = True
        move_count = 0
        board_history = {}

        while True:
            state_key = board.tobytes() + bytes([turn])
            board_history[state_key] = board_history.get(state_key, 0) + 1
            if board_history[state_key] >= 3:
                blue, green = count_cells(board)
                blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
                break

            is_terminal, terminal_value = check_terminal(board, turn)
            if is_terminal:
                blue_result = terminal_value if turn else -terminal_value
                break

            if turn:
                # MCTS agent
                action_probs = mcts.search(board, turn)
                action = mcts.select_action(action_probs, temperature=0)
            else:
                # Random opponent
                masks = action_masks(board, turn)
                legal = np.where(masks)[0]
                if len(legal) == 0:
                    blue_result = 1.0
                    break
                action = int(np.random.choice(legal))

            board = apply_move(board, action, turn)
            turn = not turn
            move_count += 1

            if move_count > 200:
                blue, green = count_cells(board)
                blue_result = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
                break

        if blue_result > 0:
            wins += 1
        pbar.set_postfix(win_rate=f"{wins / (i + 1):.0%}")

    return wins / num_games


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
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create network
    network = DualHeadNetwork(num_actions=1225).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Resume from checkpoint
    start_iteration = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        network.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = checkpoint.get('iteration', 0) + 1
        print(f"Resuming from iteration {start_iteration}")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

    # Quick baseline: random network vs random opponent
    print("\nBaseline: untrained network vs random...")
    baseline_wr = evaluate_vs_random(network, num_games=10, num_simulations=20)
    print(f"  Win rate vs random: {baseline_wr:.0%}")

    iter_pbar = tqdm(range(start_iteration, args.iterations),
                     desc="Training", unit="iter",
                     initial=start_iteration, total=args.iterations)
    for iteration in iter_pbar:
        iter_start = time.time()

        iter_pbar.set_description(f"Iter {iteration + 1}/{args.iterations}")

        # 1. Generate self-play data
        print(f"\nGenerating {args.games} self-play games "
              f"({args.simulations} sims/move)...")
        gen_start = time.time()
        examples, (bw, gw, dr) = generate_self_play_data(
            network, args.games, num_simulations=args.simulations
        )
        gen_time = time.time() - gen_start
        print(f"  Generated {len(examples)} examples in {gen_time:.1f}s")
        print(f"  Results: Blue {bw} / Green {gw} / Draw {dr}")

        # 2. Add to replay buffer
        replay_buffer.extend(examples)
        print(f"  Replay buffer: {len(replay_buffer)} examples")

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

        # 4. Evaluate
        if (iteration + 1) % EVAL_INTERVAL == 0:
            print("\nEvaluating...")

            # vs random
            wr_random = evaluate_vs_random(
                network, num_games=EVAL_GAMES, num_simulations=args.simulations
            )
            print(f"  vs Random: {wr_random:.0%}")

            # vs minimax depth-1
            wr_mm1, results_mm1 = evaluate_vs_minimax(
                network, minimax_depth=1, num_games=EVAL_GAMES,
                num_simulations=args.simulations
            )
            print(f"  vs Minimax-1: {wr_mm1:.0%} "
                  f"(W:{results_mm1['wins']} L:{results_mm1['losses']} "
                  f"D:{results_mm1['draws']})")

            # vs minimax depth-2 (less frequent, slower)
            if (iteration + 1) % (EVAL_INTERVAL * 2) == 0:
                wr_mm2, results_mm2 = evaluate_vs_minimax(
                    network, minimax_depth=2, num_games=10,
                    num_simulations=args.simulations
                )
                print(f"  vs Minimax-2: {wr_mm2:.0%} "
                      f"(W:{results_mm2['wins']} L:{results_mm2['losses']} "
                      f"D:{results_mm2['draws']})")

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
