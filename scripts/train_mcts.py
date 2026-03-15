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
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.bc import generate_bc_data, pretrain_bc          # noqa: E402
from lib.curriculum import generate_curriculum_data       # noqa: E402
from lib.device_utils import load_compiled_network  # noqa: E402
from lib.dual_network import DualHeadNetwork               # noqa: E402
from lib.mcgs import MCGS                                  # noqa: E402
from lib.train_workers import self_play_game, play_eval_game  # noqa: E402
from lib.training import train_network                     # noqa: E402


# ============================================================
# Configuration
# ============================================================

NUM_ITERATIONS       = 200
# Iteration time budget: GAMES_PER_ITERATION × avg_moves(≈100) × MCTS_SIMULATIONS
# ÷ effective_sims_per_sec(≈8600 at 100 sims / 3 workers) ≈ target gen time.
# 250 × 100 × 100 / 8600 ≈ 290s — matches previous 50 × 100 × 750 / 12500 ≈ 300s.
GAMES_PER_ITERATION  = 250
MCTS_SIMULATIONS     = 100   # prior-dominated regime: fewer sims = more concentrated targets
EVAL_SIMULATIONS     = 100
BATCH_SIZE           = 256
EPOCHS_PER_ITERATION = 5
REPLAY_BUFFER_SIZE   = 100_000  # ≈ 4 iterations at 250 games × 100 moves
LEARNING_RATE        = 3e-4
WEIGHT_DECAY         = 1e-4
C_PUCT               = 0.75  # reduced from 1.5: Q-dominated at this sim count
GUMBEL_K             = 8      # root candidates for Gumbel top-K sampling
EVAL_INTERVAL        = 10
EVAL_GAMES           = 50
CHECKPOINT_INTERVAL  = 10
CHECKPOINT_DIR       = "models/mcts"
BC_GAMES             = 500
BC_DEPTH             = 1     # teach legal moves only — avoids overpowering MCTS prior
BC_EPOCHS            = 5     # light fitting: policy must stay plastic; value bootstraps via curriculum
BC_NOISE             = 0.10  # total non-expert rate (blunders + random)
BC_BLUNDER_DEPTH     = 1     # blunder depth matches BC depth
BC_RANDOM_RATE       = 0.02  # true-random floor for positional coverage
# Value curriculum ladder: (mm_depth, npz_cache_path)
# Network advances to the next level once it beats the current level's pure
# win rate >= CURRICULUM_ADVANCE_THRESHOLD for CURRICULUM_ADVANCE_CONSECUTIVE
# consecutive gate evals.  .npz files are generated lazily and cached on disk;
# MM-3 takes ~5 min, MM-5 ~50 min to generate.
# Ladder starts at MM-3: MM-1/MM-2 games are too shallow to provide meaningful
# value signal (position quality barely correlates with outcome at depth 1-2).
CURRICULUM_LADDER = [
    (3, "models/mcts/curriculum_mm3.npz"),
    (5, "models/mcts/curriculum_mm5.npz"),
]
CURRICULUM_GAMES               = 500   # games generated per level
CURRICULUM_RATIO               = 0.50  # enough to hold value signal against noisy early self-play;
                                        # curriculum policy is masked so this doesn't affect policy learning
CURRICULUM_ADVANCE_THRESHOLD   = 0.65  # combined win rate vs current level to advance
                                        # (50% reachable via first-mover advantage alone)
CURRICULUM_ADVANCE_CONSECUTIVE = 2     # consecutive gate evals required


# ============================================================
# Per-process globals (multiprocessing spawn requires module-level state)
# ============================================================

_local_network        = None   # compiled model used for inference in workers
_base_network         = None   # uncompiled reference for in-place weight updates
_weight_queue         = None   # receives weight broadcasts from the main process


def _worker_init(state_dict, weight_queue=None):
    """Initialise a self-play or evaluation worker process."""
    global _local_network, _base_network, _weight_queue
    torch.set_num_threads(1)
    # Workers use CPU for inference: 3 independent CPU processes outrun
    # 3 processes sharing one GPU at small MCTS batch sizes (bs~K=8).
    # GPU is reserved exclusively for the training step in the main process.
    device = torch.device("cpu")
    _local_network, _base_network = load_compiled_network(state_dict, device, compile_net=False)
    _weight_queue = weight_queue


# ============================================================
# Worker entry points (must be module-level for spawn pickling)
# ============================================================

def _worker_play_game(args):
    """Self-play worker: apply pending weight update, then play one game."""
    try:
        new_sd = _weight_queue.get(block=False)   # type: ignore[union-attr]
        _base_network.load_state_dict(new_sd)     # type: ignore[union-attr]
    except Exception:
        pass
    num_simulations, c_puct, gumbel_k = args
    mcts = MCGS(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        c_puct=c_puct,
        gumbel_k=gumbel_k,
    )
    return self_play_game(mcts)


def _worker_eval_game(args):
    """Evaluation worker: play one game vs minimax/stauf."""
    num_simulations, minimax_depth, noise, engine, vary_depth, mcts_is_blue = args
    mcts = MCGS(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
    )
    result = play_eval_game(mcts, minimax_depth, noise, engine, vary_depth, mcts_is_blue)
    return result, mcts_is_blue


# ============================================================
# Self-play data generation
# ============================================================

def generate_self_play_data(
    pool,
    min_non_truncated: int = 50,
    num_simulations: int = 100,
    num_workers: int = 2,
) -> tuple:
    """
    Generate training examples from a persistent worker pool.

    Workers already hold up-to-date weights (broadcast by main after each
    training step).  Truncated games are played but their examples are discarded.

    Returns
    -------
    (examples, (blue_wins, green_wins, draws), game_moves, game_times,
     truncations, avg_branching)
    """
    task_args = (num_simulations, C_PUCT, GUMBEL_K)

    all_examples: list = []
    blue_wins = green_wins = draws = truncations = non_truncated_count = 0
    game_moves: list = []
    game_times: list = []
    all_legal_counts: list = []

    pbar = tqdm(desc="Self-play", unit="game", total=min_non_truncated)
    try:
        pending = deque()
        for _ in range(num_workers):
            pending.append(pool.apply_async(_worker_play_game, (task_args,)))

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
            pbar.set_postfix(examples=len(all_examples))
            if non_truncated_count < min_non_truncated:
                pending.append(pool.apply_async(_worker_play_game, (task_args,)))

        for fut in pending:
            fut.cancel() if hasattr(fut, 'cancel') else None  # type: ignore[attr-defined]
    finally:
        pbar.close()

    avg_branching = float(np.mean(all_legal_counts)) if all_legal_counts else 0.0
    return (all_examples, (blue_wins, green_wins, draws),
            game_moves, game_times, truncations, avg_branching)


# ============================================================
# Evaluation
# ============================================================

def _run_pool_wld(
    worker_fn,
    task_args: list,
    initializer,
    initargs: tuple,
    desc: str,
    num_workers: int,
) -> tuple[int, int, int]:
    """
    Run a multiprocessing pool collecting +1/−1/0 results into W/L/D counts.
    """
    wins = losses = draws = 0
    pbar = tqdm(total=len(task_args), desc=desc, unit="game", leave=False)
    try:
        with multiprocessing.Pool(
            processes=num_workers, initializer=initializer, initargs=initargs
        ) as pool:
            for result in pool.imap_unordered(worker_fn, task_args):
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
    return wins, losses, draws


def evaluate_vs_noisy_minimax(
    network,
    minimax_depth: int = 2,
    noise: float = 0.3,
    num_games: int = 20,
    num_simulations: int = 100,
    engine: str = 'minimax',
    vary_depth: bool = False,
) -> tuple[float, dict]:
    """
    Evaluate MCTS agent against an epsilon-greedy minimax opponent.

    Half the games are played as Blue, half as Green.
    Returns (win_rate, {wins, losses, draws}).
    """
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    task_args = [
        (num_simulations, minimax_depth, noise, engine, vary_depth, game_idx % 2 == 0)
        for game_idx in range(num_games)
    ]
    engine_label = "Stauf" if engine == 'stauf' else f"MM-{minimax_depth}"
    wins = losses = draws = 0
    wins_b = losses_b = wins_g = losses_g = 0
    pbar = tqdm(total=num_games, desc=f"Eval vs {engine_label} (noise={noise:.0%})",
                unit="game", leave=False)
    try:
        with multiprocessing.Pool(
            processes=min(2, num_games), initializer=_worker_init, initargs=(state_dict,)
        ) as pool:
            for result, is_blue in pool.imap_unordered(_worker_eval_game, task_args):
                if result > 0:
                    wins += 1
                    if is_blue:
                        wins_b += 1
                    else:
                        wins_g += 1
                elif result < 0:
                    losses += 1
                    if is_blue:
                        losses_b += 1
                    else:
                        losses_g += 1
                else:
                    draws += 1
                pbar.update(1)
                pbar.set_postfix(win_rate=f"{wins / (wins + losses + draws):.0%}")
    finally:
        pbar.close()
    games_b = num_games // 2
    games_g = num_games - games_b
    return wins / num_games, {
        "wins": wins, "losses": losses, "draws": draws,
        "wr_as_blue":  wins_b / games_b if games_b else 0.0,
        "wr_as_green": wins_g / games_g if games_g else 0.0,
    }


# ============================================================
# Curriculum helpers
# ============================================================

def _load_or_generate_curriculum(depth: int, path: str, n_games: int) -> tuple:
    """
    Return a (obs, policy, value) numpy-array tuple for MM-*depth* curriculum.

    Loads from *path* if the .npz cache already exists; otherwise generates
    *n_games* MM-depth vs MM-depth games, saves to *path*, then returns.
    """
    if os.path.exists(path):
        print(f"  Loading cached MM-{depth} curriculum: {path}")
    else:
        print(f"  Generating MM-{depth} curriculum ({n_games} games) ...")
        examples = generate_curriculum_data(depth, n_games)
        obs_np = np.array([e[0] for e in examples], dtype=np.float32)
        pol_np = np.array([e[1] for e in examples], dtype=np.float32)
        val_np = np.array([e[2] for e in examples], dtype=np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, obs=obs_np, policy=pol_np, value=val_np)
        print(f"  Saved: {path}  ({len(examples)} examples)")
    cur = np.load(path)
    n = len(cur['obs'])
    print(f"  MM-{depth} curriculum ready: {n} examples")
    return (
        cur['obs'].astype(np.float32),
        cur['policy'].astype(np.float32),
        cur['value'].astype(np.float32),
    )


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
                        help="Name for this run (default: timestamp)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    network   = DualHeadNetwork(num_actions=1225).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        network.load_state_dict(checkpoint['network'])
        saved_iter = checkpoint.get('iteration', 0) + 1
        print(f"Loaded weights from iteration {saved_iter}; "
              f"training for {args.iterations} fresh iterations")

        # Re-seed the replay buffer from the saved BC examples so the network
        # has the same BC anchor as a fresh run, without re-running minimax.
        bc_seed_path = os.path.join(CHECKPOINT_DIR, "bc_seed.npz")
        if os.path.exists(bc_seed_path):
            print(f"Loading BC seed: {bc_seed_path}")
            seed_data = np.load(bc_seed_path)
            for obs, pol, val in zip(seed_data['obs'], seed_data['policy'], seed_data['value']):
                replay_buffer.append((obs, pol, float(val)))
            print(f"Replay buffer seeded with {len(replay_buffer)} BC examples")
        else:
            # Seed file missing (e.g. first resume after upgrading the training script).
            # Generate a quick low-depth seed instead of re-running full BC generation.
            print("bc_seed.npz not found — generating quick seed (depth-3, 200 games)...")
            quick_seed = generate_bc_data(
                num_games=200, minimax_depth=3,
                noise=BC_NOISE, blunder_depth=BC_BLUNDER_DEPTH, random_rate=BC_RANDOM_RATE,
            )
            n_seed = min(len(quick_seed), REPLAY_BUFFER_SIZE)
            seed_idx = np.random.choice(len(quick_seed), n_seed, replace=False)
            replay_buffer.extend(quick_seed[i] for i in seed_idx)
            print(f"Replay buffer seeded with {len(replay_buffer)} quick-seed examples")
            # Save for next resume
            s_obs = np.array([quick_seed[i][0] for i in seed_idx])
            s_pol = np.array([quick_seed[i][1] for i in seed_idx])
            s_val = np.array([quick_seed[i][2] for i in seed_idx], dtype=np.float32)
            np.savez_compressed(bc_seed_path, obs=s_obs, policy=s_pol, value=s_val)
            print(f"Quick seed saved for future resumes: {bc_seed_path}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(args.logdir, run_name)
    writer   = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    writer.add_custom_scalars({
        "Game Stats": {
            "Game Length": ["Margin", [
                "self_play/avg_game_moves",
                "self_play/min_game_moves",
                "self_play/max_game_moves",
            ]],
            "Policy Entropy": ["Multiline", [
                "self_play/avg_policy_entropy",
            ]],
        },
        "Eval": {
            "Ladder Progress": ["Multiline", [
                "eval/ladder_progress",
                "eval/win_rate_current",
                "eval/win_rate_next",
            ]],
            "Win Rate by Colour": ["Multiline", [
                "eval/win_rate_current",
                "eval/win_rate_as_blue",
                "eval/win_rate_as_green",
            ]],
            "Curriculum": ["Multiline", [
                "curriculum/win_rate",
                "curriculum/level",
            ]],
        },
        "Training": {
            "Per-Epoch Loss": ["Multiline", [
                "train/epoch_policy_loss",
                "train/epoch_value_loss",
            ]],
            "Iteration Loss": ["Multiline", [
                "train/policy_loss",
                "train/value_loss",
            ]],
        },
    })

    if not args.checkpoint:
        print(f"\nBC pre-training: {BC_GAMES} games at depth {BC_DEPTH}, {BC_EPOCHS} epochs")
        bc_data = generate_bc_data(
            num_games=BC_GAMES, minimax_depth=BC_DEPTH,
            noise=BC_NOISE, blunder_depth=BC_BLUNDER_DEPTH, random_rate=BC_RANDOM_RATE,
        )
        pretrain_bc(network, bc_data, epochs=BC_EPOCHS, batch_size=BATCH_SIZE,
                    device=device, learning_rate=1e-3, weight_decay=WEIGHT_DECAY,
                    writer=writer)

        # Save BC baseline so it can be loaded for comparison later
        bc_ckpt = os.path.join(CHECKPOINT_DIR, "bc_baseline.pt")
        torch.save({'network': network.state_dict()}, bc_ckpt)
        print(f"BC baseline saved: {bc_ckpt}")

        # Seed the replay buffer with a random sample of BC examples.
        # This prevents the first self-play iterations from training on an
        # empty buffer (cold-start) and gives the network a stable BC anchor
        # while self-play data gradually takes over.
        n_seed = min(len(bc_data), REPLAY_BUFFER_SIZE)
        seed_idx = np.random.choice(len(bc_data), n_seed, replace=False)
        replay_buffer.extend(bc_data[i] for i in seed_idx)
        print(f"Replay buffer seeded with {len(replay_buffer)} BC examples "
              f"(evicts over ~{REPLAY_BUFFER_SIZE // (GAMES_PER_ITERATION * 60)} iters)\n")

        # Persist the seeded examples so resuming from bc_baseline.pt can reload
        # them without re-running the 50-minute minimax data generation.
        # Policy targets are one-hot → compress to a small file.
        bc_seed_path = os.path.join(CHECKPOINT_DIR, "bc_seed.npz")
        seed_obs = np.array([bc_data[i][0] for i in seed_idx])
        seed_pol = np.array([bc_data[i][1] for i in seed_idx])
        seed_val = np.array([bc_data[i][2] for i in seed_idx], dtype=np.float32)
        np.savez_compressed(bc_seed_path, obs=seed_obs, policy=seed_pol, value=seed_val)
        print(f"BC seed saved: {bc_seed_path}")

    # ── Value curriculum ladder ───────────────────────────────────────────
    # Start at MM-1, advance when the network beats each level >= threshold
    # for CURRICULUM_ADVANCE_CONSECUTIVE consecutive gate evals.  Levels are
    # generated lazily and cached as .npz files.
    curriculum_level: int = 0         # current index into CURRICULUM_LADDER
    curriculum_consecutive: int = 0   # consecutive gate evals beating current level
    cur_depth, cur_path = CURRICULUM_LADDER[curriculum_level]
    print(f"Initialising value curriculum ladder at MM-{cur_depth} ...")
    value_curriculum: tuple | None = _load_or_generate_curriculum(
        cur_depth, cur_path, CURRICULUM_GAMES,
    )

    print("=" * 60)
    print("AlphaZero MCTS Training for Microscope")
    print("=" * 60)
    print(f"Iterations:       {args.iterations}")
    print(f"Games/iteration:  {args.games}")
    print(f"Sims/move:        {args.simulations}")
    print(f"Replay buffer:    {REPLAY_BUFFER_SIZE}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs/iteration: {EPOCHS_PER_ITERATION}")
    print(f"Value curriculum: MM-{CURRICULUM_LADDER[curriculum_level][0]} "
          f"(ladder: {' → '.join(str(d) for d, _ in CURRICULUM_LADDER)} → retire)")
    print("=" * 60)

    NUM_SELF_PLAY_WORKERS = 3
    weight_queue: multiprocessing.Queue = multiprocessing.Queue()
    init_sd = {k: v.clone().cpu() for k, v in network.state_dict().items()}
    self_play_pool = multiprocessing.Pool(
        processes=NUM_SELF_PLAY_WORKERS,
        initializer=_worker_init,
        initargs=(init_sd, weight_queue),
    )


    iter_pbar = tqdm(range(args.iterations), desc="Training", unit="iter")
    for iteration in iter_pbar:
        iter_start = time.time()
        iter_pbar.set_description(f"Iter {iteration + 1}/{args.iterations}")
        step = iteration + 1

        # ── 1. Self-play ──────────────────────────────────────────────────
        print(f"\nGenerating {args.games} non-truncated games "
              f"({args.simulations} sims/move)...")
        gen_start = time.time()
        examples, (bw, gw, dr), game_moves, game_times, truncations, avg_branching = (
            generate_self_play_data(
                self_play_pool,
                min_non_truncated=args.games,
                num_simulations=args.simulations,
                num_workers=NUM_SELF_PLAY_WORKERS,
            )
        )
        gen_time = time.time() - gen_start
        moves_arr = np.array(game_moves)
        avg_moves = float(moves_arr.mean())
        med_moves = float(np.median(moves_arr))
        std_moves = float(moves_arr.std())
        avg_gtime = sum(game_times) / len(game_times)
        trunc_pct = 100.0 * truncations / len(game_moves)
        total_sims = int(moves_arr.sum()) * args.simulations
        sims_per_sec = total_sims / gen_time if gen_time > 0 else 0.0

        print(f"  {len(examples)} examples in {gen_time:.1f}s  "
              f"| Blue {bw} / Green {gw} / Draw {dr}")
        print(f"  Game length: avg {avg_moves:.1f}  med {med_moves:.1f}  "
              f"std {std_moves:.1f}  (min {moves_arr.min()}, max {moves_arr.max()})")
        print(f"  Truncated: {truncations}/{len(game_moves)} ({trunc_pct:.1f}%)"
              f"  |  Branching: {avg_branching:.1f}"
              f"  |  {sims_per_sec:.0f} sims/sec")

        writer.add_scalar("self_play/examples_generated",   len(examples),         step)
        writer.add_scalar("self_play/avg_game_moves",       avg_moves,             step)
        writer.add_scalar("self_play/median_game_moves",    med_moves,             step)
        writer.add_scalar("self_play/std_game_moves",       std_moves,             step)
        writer.add_scalar("self_play/min_game_moves",       int(moves_arr.min()),  step)
        writer.add_scalar("self_play/max_game_moves",       int(moves_arr.max()),  step)
        writer.add_scalar("self_play/truncation_pct",       trunc_pct,             step)
        writer.add_scalar("self_play/avg_branching_factor", avg_branching,         step)

        # Shannon entropy of MCTS visit distributions used as policy targets.
        # Ceiling is log(K) ≈ 2.08 nats (uniform over K Gumbel candidates),
        # not log(branching). Decreases toward 0 as Q sharpens and one action dominates.
        entropies = []
        for _, policy_target, _ in examples:
            p = policy_target[policy_target > 0]
            if p.size > 0:
                entropies.append(float(-np.sum(p * np.log(p))))
        avg_policy_entropy = float(np.mean(entropies)) if entropies else 0.0
        print(f"  Policy entropy: {avg_policy_entropy:.3f} nats  "
              f"(max≈{np.log(GUMBEL_K):.3f} uniform over K={GUMBEL_K}, concentrated→0)")
        writer.add_scalar("self_play/avg_policy_entropy", avg_policy_entropy, step)
        if entropies:
            writer.add_histogram("self_play/policy_entropy_dist", np.array(entropies), step)

        writer.add_scalar("timing/gen_seconds",             gen_time,              step)
        writer.add_scalar("timing/avg_game_seconds",        avg_gtime,             step)
        writer.add_scalar("timing/sims_per_sec",            sims_per_sec,          step)

        is_eval_iter = step % EVAL_INTERVAL == 0

        # ── 2. Replay buffer ──────────────────────────────────────────────
        replay_buffer.extend(examples)
        print(f"  Replay buffer: {len(replay_buffer)} examples")
        writer.add_scalar("self_play/buffer_size", len(replay_buffer), step)

        # ── 3. Train ──────────────────────────────────────────────────────
        print(f"\nTraining ({EPOCHS_PER_ITERATION} epochs, batch {BATCH_SIZE})...")
        train_start = time.time()
        losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITERATION, device=device,
            curriculum=value_curriculum, curriculum_ratio=CURRICULUM_RATIO,
        )
        train_time = time.time() - train_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  policy={losses['policy_loss']:.4f}  value={losses['value_loss']:.4f}"
              f"  total={losses['total_loss']:.4f}  ({train_time:.1f}s)")

        writer.add_scalar("train/policy_loss",  losses['policy_loss'], step)
        writer.add_scalar("train/value_loss",   losses['value_loss'],  step)
        writer.add_scalar("train/total_loss",   losses['total_loss'],  step)
        writer.add_scalar("train/lr",           current_lr,            step)
        for ep_idx, ep in enumerate(losses.get('epoch_losses', [])):
            ep_step = (iteration * EPOCHS_PER_ITERATION) + ep_idx + 1
            writer.add_scalar("train/epoch_policy_loss", ep['policy_loss'], ep_step)
            writer.add_scalar("train/epoch_value_loss",  ep['value_loss'],  ep_step)
        writer.add_scalar("timing/train_seconds", train_time, step)

        # Value output distribution: sample buffer, forward pass, histogram.
        # Bimodal (mass near ±1) = value head learning; flat = still guessing.
        _sample_n = min(1024, len(replay_buffer))
        _buf = list(replay_buffer)
        _idx = np.random.choice(len(_buf), _sample_n, replace=False)
        _obs = torch.from_numpy(np.array([_buf[i][0] for i in _idx])).to(device)
        network.eval()
        with torch.no_grad():
            _, _val_preds = network(_obs)
        network.train()
        writer.add_histogram("train/value_output_dist", _val_preds.squeeze().cpu(), step)

        # Broadcast updated weights to self-play workers
        new_sd = {k: v.cpu() for k, v in network.state_dict().items()}
        for _ in range(NUM_SELF_PLAY_WORKERS):
            weight_queue.put(new_sd)

        # ── 4. Evaluate ───────────────────────────────────────────────────
        if is_eval_iter:
            # Evaluate vs current ladder depth, plus the next depth if one exists.
            # ladder_progress = curriculum_level + win_rate_vs_current (0–4 while
            # curriculum is active; 4 + wr_vs_final once retired).
            if value_curriculum is not None:
                eval_cur_depth  = CURRICULUM_LADDER[curriculum_level][0]
                eval_next_entry = (CURRICULUM_LADDER[curriculum_level + 1]
                                   if curriculum_level + 1 < len(CURRICULUM_LADDER)
                                   else None)
                level_base      = curriculum_level
            else:
                eval_cur_depth  = CURRICULUM_LADDER[-1][0]  # keep tracking vs MM-5
                eval_next_entry = None
                level_base      = len(CURRICULUM_LADDER)    # = 4

            print(f"\nEvaluating vs ladder (MM-{eval_cur_depth}"
                  + (f" + MM-{eval_next_entry[0]}" if eval_next_entry else "")
                  + ")...")

            wr_cur, res_cur = evaluate_vs_noisy_minimax(
                network, minimax_depth=eval_cur_depth, noise=0.0,
                num_games=EVAL_GAMES, num_simulations=EVAL_SIMULATIONS,
            )
            label = f"lvl {curriculum_level}" if value_curriculum is not None else "post-ladder"
            print(f"  vs MM-{eval_cur_depth} ({label}): {wr_cur:.0%} "
                  f"W:{res_cur['wins']} L:{res_cur['losses']} D:{res_cur['draws']}  "
                  f"(B:{res_cur['wr_as_blue']:.0%} G:{res_cur['wr_as_green']:.0%})")
            writer.add_scalar("eval/win_rate_current",  wr_cur,                   step)
            writer.add_scalar("eval/win_rate_as_blue",  res_cur['wr_as_blue'],    step)
            writer.add_scalar("eval/win_rate_as_green", res_cur['wr_as_green'],   step)
            writer.add_scalar("eval/ladder_progress",   level_base + wr_cur,      step)
            writer.add_scalar("curriculum/level",       curriculum_level,         step)

            if eval_next_entry is not None:
                next_depth, _ = eval_next_entry
                wr_next, res_next = evaluate_vs_noisy_minimax(
                    network, minimax_depth=next_depth, noise=0.0,
                    num_games=EVAL_GAMES, num_simulations=EVAL_SIMULATIONS,
                )
                print(f"  vs MM-{next_depth} (next):   {wr_next:.0%} "
                      f"W:{res_next['wins']} L:{res_next['losses']} D:{res_next['draws']}")
                writer.add_scalar("eval/win_rate_next", wr_next, step)

            # ── Curriculum ladder advancement ──────────────────────────────
            if value_curriculum is not None:
                writer.add_scalar("curriculum/win_rate", wr_cur, step)

                if wr_cur >= CURRICULUM_ADVANCE_THRESHOLD:
                    curriculum_consecutive += 1
                    print(f"  Curriculum beat "
                          f"({curriculum_consecutive}/{CURRICULUM_ADVANCE_CONSECUTIVE})")
                else:
                    curriculum_consecutive = 0

                if curriculum_consecutive >= CURRICULUM_ADVANCE_CONSECUTIVE:
                    curriculum_consecutive = 0
                    curriculum_level += 1
                    if curriculum_level >= len(CURRICULUM_LADDER):
                        print("  Curriculum ladder complete — retiring value curriculum")
                        value_curriculum = None
                        writer.add_scalar("curriculum/level", curriculum_level, step)
                    else:
                        next_depth, next_path = CURRICULUM_LADDER[curriculum_level]
                        print(f"  Advancing curriculum to MM-{next_depth}")
                        value_curriculum = _load_or_generate_curriculum(
                            next_depth, next_path, CURRICULUM_GAMES,
                        )

        # ── 5. Checkpoint ─────────────────────────────────────────────────
        if step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"iter_{step:04d}.pt")
            torch.save({
                'iteration': iteration,
                'network':   network.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"\n  Checkpoint saved: {ckpt_path}")

        iter_time = time.time() - iter_start
        iter_pbar.set_postfix(
            loss=f"{losses['total_loss']:.3f}",
            buf=len(replay_buffer),
            time=f"{iter_time:.0f}s",
        )

    self_play_pool.terminate()
    self_play_pool.join()
    writer.close()

    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    torch.save({
        'iteration': args.iterations - 1,
        'network':   network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == "__main__":
    main()
