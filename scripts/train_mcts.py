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
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.device_utils import load_compiled_network                   # noqa: E402
from lib.dual_network import DualHeadNetwork                         # noqa: E402
from lib.mcgs import MCGS                                            # noqa: E402
from lib.t7g import (                                                 # noqa: E402
    soft_policy_from_mm, new_board, apply_move, check_terminal,
    board_to_obs, find_best_move, action_masks, count_cells,
)
from lib.train_workers import (                                      # noqa: E402
    self_play_game_pool, play_eval_game,
)
from lib.training import train_network                               # noqa: E402


# ============================================================
# Configuration
# ============================================================

NUM_ITERATIONS       = 500
GAMES_PER_ITERATION  = 250
MCTS_SIMULATIONS     = 250
EVAL_SIMULATIONS     = 250
BATCH_SIZE           = 256
EPOCHS_PER_ITERATION = 5
REPLAY_BUFFER_ITERS  = 7        # keep this many iterations of self-play data
TARGET_EXAMPLES_ITER = 6_000    # adaptive games/iter targets this example count
GAMES_MIN            = 50       # clamp range for adaptive scheduling
GAMES_MAX            = 500
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
C_PUCT               = 0.75  # reduced from 1.5: Q-dominated at this sim count
GUMBEL_K             = 8      # root candidates; bench confirms 4x better Q estimates vs K=20

POOL_SIZE            = 32     # concurrent games per worker (batches bs~K*POOL_SIZE leaves/pass)
MM_MIX_GAMES         = 100    # MM3 vs MCTS games added to buffer each iteration
MM_MIX_DEPTH         = 3
EVAL_INTERVAL        = 5
EVAL_GAMES           = 50
EVAL_WORKERS         = 4
CHECKPOINT_INTERVAL  = 10
CHECKPOINT_DIR       = "models/mcts"
# Value curriculum ladder: (mm_depth, npz_cache_path)
# Network advances to the next level once it beats the current level's pure
# Eval ladder: tracks playing-strength progression (MM-3 → MM-5).
# Advancement requires beating each level >= threshold for consecutive evals.
# .npz files are generated lazily and cached on disk (MM-3 ~5 min, MM-5 ~50 min).
# Each entry: (mm_depth, noise, label)
# noise=1.0 → pure random opponent; noise=0.0 → pure minimax
EVAL_LADDER = [
    (1, 0.60, "MM1-semi"),    # first rung: 60% random softens the cold-start cliff
    (1, 0.20, "MM1-noisy"),
    (1, 0.00, "MM1"),
    (2, 0.00, "MM2"),
    (3, 0.00, "MM3"),
    (4, 0.00, "MM4"),
    (5, 0.00, "MM5"),
]
EVAL_ADVANCE_THRESHOLD   = 0.65  # combined win rate required to advance ladder
EVAL_ADVANCE_CONSECUTIVE = 2     # consecutive gate evals required
# Policy distillation: replace MCTS visit-count policy targets with soft MM distributions.
# score_root_moves(depth=2) scores all legal moves; softmax at low temperature
# concentrates mass on the best move while maintaining a small exploration signal.
# Applied to both self-play and MM-mix positions (wherever the network was on the board).
POLICY_DISTILL_DEPTH = 3    # odd-depth avoids even-depth horizon artefact (inflated score gaps)
# micro4 BFS scores wider than micro3; T=0.075 => ~2.08 nats (MCTS K=8 ceiling)
POLICY_DISTILL_TEMP  = 0.075


# ============================================================
# Replay buffer
# ============================================================

class _IterBuffer:
    """
    Rolling window replay buffer sized by iteration count, not example count.

    Stores the last REPLAY_BUFFER_ITERS iterations as separate batches so the
    window stays proportional to current game length.  Early training produces
    long branchy games (~17k examples/iter); late training produces short
    focused games (~1-2k examples/iter).  A fixed example-count deque would
    hold 60+ stale iterations in the latter case; this stays at exactly N.
    """

    def __init__(self, maxiters: int) -> None:
        self._batches: deque = deque(maxlen=maxiters)

    def append_batch(self, batch: list) -> None:
        self._batches.append(batch)

    def __len__(self) -> int:
        return sum(len(b) for b in self._batches)

    def __iter__(self):
        return chain.from_iterable(self._batches)


# ============================================================
# Per-process globals (multiprocessing spawn requires module-level state)
# ============================================================

_local_network        = None   # compiled model used for inference in workers
_base_network         = None   # uncompiled reference for in-place weight updates
_weight_queue         = None   # receives weight broadcasts from the main process
_worker_mcts_cls      = None   # MCGS - set in _worker_init
_worker_mcts_kwargs   = {}     # num_simulations, c_puct, gumbel_k, heuristic weights


def _worker_init(state_dict, weight_queue=None, num_actions=1225, mcts_cls=None,
                 mcts_kwargs=None):
    """Initialise a self-play or evaluation worker process."""
    global _local_network, _base_network, _weight_queue, _worker_mcts_cls, _worker_mcts_kwargs
    import torch as _torch
    from lib.device_utils import get_device as _get_device  # noqa: E402
    # Pool workers batch bs~K*POOL_SIZE leaves per forward pass — GPU beneficial.
    # CPU fallback if no GPU available (also used for eval workers).
    try:
        device = _get_device()
    except Exception:
        device = _torch.device("cpu")
    if device.type != "cpu":
        # One worker per GPU: no thread tuning needed
        pass
    else:
        _torch.set_num_threads(1)
    _local_network, _base_network = load_compiled_network(
        state_dict, device, num_actions=num_actions, compile_net=False,
    )
    _weight_queue = weight_queue
    _worker_mcts_cls    = mcts_cls if mcts_cls is not None else MCGS
    _worker_mcts_kwargs = mcts_kwargs or {}


# ============================================================
# Worker entry points (must be module-level for spawn pickling)
# ============================================================

def _worker_eval_game(args):
    """Evaluation worker: play one game vs minimax/stauf."""
    num_simulations, minimax_depth, noise, engine, vary_depth, mcts_is_blue = args
    _local_network.eval()  # type: ignore[union-attr]
    mcts = _worker_mcts_cls(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        **{k: v for k, v in _worker_mcts_kwargs.items() if k != 'num_simulations'},
    )
    result = play_eval_game(mcts, minimax_depth, noise, engine, vary_depth, mcts_is_blue)
    return result, mcts_is_blue


# ============================================================
# Self-play data generation
# ============================================================

def generate_self_play_data(
    mcts: MCGS,
    min_non_truncated: int = 50,
    mcts_pool: 'list | None' = None,
    policy_relabel_fn=None,
) -> tuple:
    """
    Generate training examples in-process using the main network directly.

    Plays games one at a time until min_non_truncated non-truncated games have
    been collected.  Truncated games are counted but their examples are discarded.

    Returns
    -------
    (examples, (blue_wins, green_wins, draws), game_moves, game_times,
     truncations, avg_branching)
    """
    all_examples: list = []
    blue_wins = green_wins = draws = truncations = non_truncated_count = 0
    game_moves: list = []
    game_times: list = []
    all_legal_counts: list = []
    # Raw 5-tuples (obs, raw_policy, value, board, turn) — relabeled after pool drains
    # so policy distillation (minimax calls) never blocks the GPU-critical pool loop.
    raw_examples: list = []

    pbar = tqdm(desc="Self-play", unit="game", total=min_non_truncated)
    try:
        while non_truncated_count < min_non_truncated:
            remaining = min_non_truncated - non_truncated_count
            # No fixed multiplier: target exactly what's needed.  The generator
            # streams results so we break as soon as the count is met, giving
            # at most POOL_SIZE-1 extra games instead of 25%+POOL_SIZE.
            # The while loop handles any shortfall if truncations reappear.
            target = remaining
            for game_examples, winner, moves, gtime, trunc, legal_counts in (
                self_play_game_pool(mcts, POOL_SIZE, target, mcts_pool)
            ):
                all_legal_counts.extend(legal_counts)
                game_moves.append(moves)
                game_times.append(gtime)
                if trunc:
                    truncations += 1
                else:
                    raw_examples.extend(game_examples)
                    non_truncated_count += 1
                    pbar.update(1)
                if winner > 0:
                    blue_wins += 1
                elif winner < 0:
                    green_wins += 1
                else:
                    draws += 1
                if non_truncated_count >= min_non_truncated:
                    break
            pbar.set_postfix(raw=len(raw_examples))
    finally:
        pbar.close()

    # Policy relabeling — outside the GPU loop, parallelised across CPU cores.
    # ctypes releases the GIL so threads give true parallelism against micro4.dll.
    # Chunked to amortise per-task dispatch overhead (per-example dispatch is too fine).
    if policy_relabel_fn:
        _fn = policy_relabel_fn
        n_workers = max(1, (os.cpu_count() or 4) // 2)
        chunk_size = max(32, len(raw_examples) // (n_workers * 4))
        chunks = [raw_examples[i:i + chunk_size]
                  for i in range(0, len(raw_examples), chunk_size)]

        def _relabel_chunk(chunk):
            return [_fn(e[3], e[4]) for e in chunk]

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            policy_chunks = list(ex.map(_relabel_chunk, chunks))
        policies = [p for pc in policy_chunks for p in pc]
        all_examples = [
            (obs, pol, val)
            for (obs, _, val, _, _), pol in zip(raw_examples, policies)
        ]
    else:
        all_examples = [(obs, p, v) for obs, p, v, _, _ in raw_examples]

    avg_branching = float(np.mean(all_legal_counts)) if all_legal_counts else 0.0
    return (all_examples, (blue_wins, green_wins, draws),
            game_moves, game_times, truncations, avg_branching)


# ============================================================
# MM-mix data generation
# ============================================================

def generate_mm_mix_data(
    mcts: MCGS,
    num_games: int,
    mm_depth: int = MM_MIX_DEPTH,
) -> list:
    """
    Generate training examples from MCTS vs MM games.

    Half the games are played as Blue, half as Green.  Only positions where
    MCTS was the active player are recorded — MM moves provide the opponent
    context but are not used as training targets.

    These games are decisive (MM3 nearly always forces a winner), giving the
    value head concrete ±1 targets that self-play alone cannot provide during
    the cold-start period.
    """
    examples: list = []
    blue_wins = green_wins = draws = 0

    for game_idx in range(num_games):
        mcts_is_blue = game_idx % 2 == 0
        board  = new_board()
        turn   = True
        mcts.clear()
        board_history: dict = {}
        move_count = 0
        history: list[tuple] = []   # (obs, policy, was_blue)
        outcome = 0.0

        while True:
            state_key = board.tobytes() + bytes([turn])
            board_history[state_key] = board_history.get(state_key, 0) + 1
            if board_history[state_key] >= 3:
                blue, green = count_cells(board)
                outcome = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
                break

            is_t, val = check_terminal(board, turn)
            if is_t:
                outcome = float(val) if val is not None else 0.0
                break

            if turn == mcts_is_blue:
                action_probs = mcts.search(board, turn)
                action = mcts.select_action(action_probs, board=board, turn=turn, temperature=1)
                mcts.advance_tree(action)
                history.append((board_to_obs(board, turn), action_probs, turn))
            else:
                legal = np.where(action_masks(board, turn))[0]
                if len(legal) == 0:
                    turn = not turn
                    continue
                action = find_best_move(board.tobytes(), mm_depth, turn)
                if action in (-1, 1225):
                    turn = not turn
                    continue

            board = apply_move(board, action, turn)
            turn  = not turn
            move_count += 1
            if move_count > 200:
                blue, green = count_cells(board)
                outcome = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
                break

        if outcome > 0:
            blue_wins += 1
        elif outcome < 0:
            green_wins += 1
        else:
            draws += 1

        for obs, policy, was_blue in history:
            value = outcome if was_blue else -outcome
            examples.append((obs, policy, value))

    print(f"  MM-mix ({mm_depth}): {num_games} games → {len(examples)} examples"
          f"  | Blue {blue_wins} / Green {green_wins} / Draw {draws}")
    return examples


# ============================================================
# BC warmup
# ============================================================

BC_WARMUP_DEPTH   = MM_MIX_DEPTH
BC_WARMUP_TEMP    = 0.075   # same as POLICY_DISTILL_TEMP — concentrate on best moves
BC_RANDOM_OPENS   = 2       # number of random moves at the start of each game per side
BC_BLUNDER_RATE   = 0.02    # probability of playing a random legal move instead of MM best

MAX_MOVES = 500             # truncation guard


def _bc_play_game(args: tuple) -> list:
    """Play one MM vs MM game with random openings and occasional blunders."""
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
            outcome = float(val) if val is not None else 0.0
            break

        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            outcome = -1.0
            break

        # Random opening: first BC_RANDOM_OPENS moves for each side are random
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


def generate_bc_warmup_data(num_games: int, depth: int = BC_WARMUP_DEPTH,
                            cache_path: str | None = None) -> list:
    """
    Generate behaviour-cloning examples from MM vs MM self-play.

    Each position seen by the active player becomes one training example:
      obs    = board_to_obs(board, turn)
      policy = soft_policy_from_mm(board, depth, as_blue=turn, temp=BC_WARMUP_TEMP)
      value  = game outcome from the active player's perspective (+1 win / -1 loss / 0 draw)

    Games are played in parallel threads — ctypes releases the GIL so MM calls
    give true CPU parallelism.

    If cache_path is given, attempts to load from file first; saves on generation.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading BC data from cache: {cache_path}")
        data = np.load(cache_path)
        obs_arr    = data['obs']      # (N, 7, 7, 4)
        policy_arr = data['policy']   # (N, 1225)
        value_arr  = data['value']    # (N,)
        examples = list(zip(obs_arr, policy_arr, value_arr.tolist()))
        print(f"  Loaded {len(examples)} BC examples")
        return examples

    n_workers = max(1, (os.cpu_count() or 4) - 1)
    examples: list = []  # type: ignore[assignment]
    pbar = tqdm(total=num_games, desc=f"BC warmup (MM{depth})", unit="game")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for game_examples in ex.map(_bc_play_game, [(i, depth) for i in range(num_games)]):
            examples.extend(game_examples)
            pbar.update(1)
            pbar.set_postfix(examples=len(examples))
    pbar.close()
    print(f"  BC warmup: {num_games} games → {len(examples)} examples")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        obs_arr    = np.stack([e[0] for e in examples])
        policy_arr = np.stack([e[1] for e in examples])
        value_arr  = np.array([e[2] for e in examples], dtype=np.float32)
        np.savez_compressed(cache_path, obs=obs_arr, policy=policy_arr, value=value_arr)
        print(f"  Saved BC cache: {cache_path}")

    return examples


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
    Run a multiprocessing pool collecting +1/-1/0 results into W/L/D counts.
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
    num_actions: int = 1225,
    mcts_cls=None,
    mcts_kwargs: dict | None = None,
) -> tuple[float, dict]:
    """
    Evaluate MCTS agent against an epsilon-greedy minimax opponent.

    Half the games are played as Blue, half as Green.
    Returns (win_rate, {wins, losses, draws}).
    """
    _mcts_cls    = mcts_cls or MCGS
    _mcts_kw     = mcts_kwargs or {}
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
            processes=min(EVAL_WORKERS, num_games), initializer=_worker_init,
            initargs=(state_dict, None, num_actions, _mcts_cls, _mcts_kw),
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
    parser.add_argument("--memtest", action="store_true",
                        help="Minimal config to verify memory stabilises")
    parser.add_argument("--no-relabel", action="store_true",
                        help="Use raw MCTS visit-count targets instead of MM policy distillation")
    parser.add_argument("--bc-warmup", type=int, default=0, metavar="N",
                        help="Pre-fill replay buffer with N BC games before iteration 1")
    parser.add_argument("--bc-depth", type=int, default=BC_WARMUP_DEPTH, metavar="D",
                        help=f"Minimax depth for BC data generation (default: {BC_WARMUP_DEPTH})")
    parser.add_argument("--bc-epochs", type=int, default=100, metavar="N",
                        help="Training epochs on BC data before self-play begins (default: 100)")
    parser.add_argument("--bc-cache", type=str, default=None, metavar="PATH",
                        help="Path to save/load BC data (.npz). Use 'auto' to derive from params.")
    args = parser.parse_args()

    if args.memtest:
        if args.games == GAMES_PER_ITERATION:   # not explicitly overridden
            args.games = 8
        args.simulations = 32
        args.iterations  = 20
        global POOL_SIZE
        POOL_SIZE = args.games

    # Policy distillation: replace MCTS visit-count policy targets with soft MM-2 distributions.
    # Win-rate analysis shows median move gap is 0.067 in win-probability space vs 0.006 in
    # tanh material space — 10x more signal.  MM-2 scores all legal moves; softmax at
    # temperature=0.02 concentrates mass on the best move while maintaining a small
    # exploration signal.  Applied to all positions (self-play + MM-mix).
    policy_relabel_fn = None if args.no_relabel else (
        lambda board, turn: soft_policy_from_mm(board, POLICY_DISTILL_DEPTH, turn,
                                                POLICY_DISTILL_TEMP)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    num_actions = 1225
    network   = DualHeadNetwork(num_actions=num_actions).to(device)
    # Compiled wrapper shares weights with network; used for MCGS inference only.
    # network itself is kept uncompiled for optimizer, state_dict, train/eval, etc.
    inference_network = (  # type: ignore[assignment]
        torch.compile(network) if device.type == "cuda" else network
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-4,
    )

    replay_buffer = _IterBuffer(maxiters=2 if args.memtest else REPLAY_BUFFER_ITERS)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        network.load_state_dict(checkpoint['network'])
        saved_iter = checkpoint.get('iteration', 0) + 1
        print(f"Loaded weights from iteration {saved_iter}; "
              f"training for {args.iterations} fresh iterations")

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

    # ── Eval ladder (tracks playing-strength progression) ─────────────────
    eval_level: int = 0               # current index into EVAL_LADDER
    eval_consecutive: int = 0         # consecutive gate evals beating current level

    print("=" * 60)
    print("AlphaZero MCTS Training for Microscope")
    print("=" * 60)
    print(f"Iterations:       {args.iterations}")
    print(f"Games/iteration:  {args.games}")
    print(f"Sims/move:        {args.simulations}")
    print(f"Replay buffer:    last {REPLAY_BUFFER_ITERS} iterations")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs/iteration: {EPOCHS_PER_ITERATION}")
    print(f"Eval ladder:      {' > '.join(lbl for _, _, lbl in EVAL_LADDER)} > retire")
    print("=" * 60)

    # Self-play runs in-process: network lives on GPU in the main thread,
    # no weight serialisation, no inter-process memory overhead.
    _mcts_kwargs = dict(num_simulations=args.simulations, c_puct=C_PUCT, gumbel_k=GUMBEL_K)
    self_play_mcts = MCGS(inference_network, **_mcts_kwargs)
    mcts_pool = [
        MCGS(inference_network, **_mcts_kwargs)
        for _ in range(POOL_SIZE)
    ]
    # Trigger torch.compile JIT before the first iteration so the warmup cost
    # doesn't skew iteration-1 timing.
    if device.type == "cuda":
        print("Warming up torch.compile ...", end=" ", flush=True)
        network.eval()
        with torch.no_grad():
            _w = torch.zeros(1, 7, 7, 4, device=device)
            inference_network(_w)  # type: ignore[operator]
        network.train()
        print("done")

    # ── BC warmup: pre-fill buffer with MM3 examples and train before self-play ──
    if args.bc_warmup > 0:
        print(f"\nBC warmup: generating {args.bc_warmup} MM{args.bc_depth} games...")
        bc_cache = args.bc_cache
        if bc_cache == "auto":
            bc_cache = (
                f"{CHECKPOINT_DIR}/bc_N{args.bc_warmup}"
                f"_D{args.bc_depth}_T{BC_WARMUP_TEMP}"
                f"_O{BC_RANDOM_OPENS}_B{BC_BLUNDER_RATE}.npz"
            )
        bc_examples = generate_bc_warmup_data(args.bc_warmup, depth=args.bc_depth,
                                              cache_path=bc_cache)
        replay_buffer.append_batch(bc_examples)
        print(f"  Replay buffer: {len(replay_buffer)} examples")
        print(f"  Pre-training on BC data ({args.bc_epochs} epochs)...")
        network.train()
        bc_losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=args.bc_epochs, device=device,
            desc=f"BC pre-train (MM{args.bc_depth})",
        )
        print(f"  BC pre-train: policy={bc_losses['policy_loss']:.4f}"
              f"  value={bc_losses['value_loss']:.4f}")

    # Adaptive games/iter: EMA of examples-per-game keeps training workload stable
    # as games shorten with improving policy.  Initialised to None so the first
    # iteration uses args.games unchanged; EMA kicks in from iteration 2 onward.
    _epg_ema: float | None = None   # examples-per-game exponential moving average
    games_this_iter = args.games

    iter_pbar = tqdm(range(args.iterations), desc="Training", unit="iter")
    for iteration in iter_pbar:
        iter_start = time.time()
        iter_pbar.set_description(f"Iter {iteration + 1}/{args.iterations}")
        step = iteration + 1

        # ── 1. Self-play ──────────────────────────────────────────────────
        print(f"\nGenerating {games_this_iter} non-truncated games "
              f"({args.simulations} sims/move)...")
        gen_start = time.time()
        network.eval()
        examples, (bw, gw, dr), game_moves, game_times, truncations, avg_branching = (
            generate_self_play_data(
                self_play_mcts,
                min_non_truncated=games_this_iter,
                mcts_pool=mcts_pool,
                policy_relabel_fn=policy_relabel_fn,
            )
        )
        network.train()
        gen_time = time.time() - gen_start
        moves_arr = np.array(game_moves)
        avg_moves = float(moves_arr.mean())
        med_moves = float(np.median(moves_arr))
        std_moves = float(moves_arr.std())
        avg_gtime = gen_time / len(game_times)
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

        # Adaptive games/iter — update EMA and compute next iteration's game count.
        _epg = len(examples) / max(1, games_this_iter)
        _epg_ema = _epg if _epg_ema is None else 0.7 * _epg_ema + 0.3 * _epg
        games_this_iter = max(GAMES_MIN, min(GAMES_MAX,
                                             round(TARGET_EXAMPLES_ITER / _epg_ema)))
        print(f"  examples/game: {_epg:.1f}  (EMA {_epg_ema:.1f})"
              f"  → next iter: {games_this_iter} games")
        writer.add_scalar("self_play/examples_per_game",    _epg,                  step)
        writer.add_scalar("self_play/games_this_iter",      games_this_iter,       step)

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
              f"(max~{np.log(GUMBEL_K):.3f} uniform over K={GUMBEL_K}, concentrated->0)")
        writer.add_scalar("self_play/avg_policy_entropy", avg_policy_entropy, step)
        if entropies:
            writer.add_histogram("self_play/policy_entropy_dist", np.array(entropies), step)

        writer.add_scalar("timing/gen_seconds",             gen_time,              step)
        writer.add_scalar("timing/avg_game_seconds",        avg_gtime,             step)
        writer.add_scalar("timing/sims_per_sec",            sims_per_sec,          step)

        is_eval_iter = (step % EVAL_INTERVAL == 0) and not args.memtest

        # ── 1b. MM-mix games ──────────────────────────────────────────────
        network.eval()
        mm_mix_examples = generate_mm_mix_data(self_play_mcts, MM_MIX_GAMES)
        network.train()
        writer.add_scalar("self_play/mm_mix_examples", len(mm_mix_examples), step)

        # ── 2. Replay buffer ──────────────────────────────────────────────
        replay_buffer.append_batch(examples + mm_mix_examples)
        print(f"  Replay buffer: {len(replay_buffer)} examples")
        writer.add_scalar("self_play/buffer_size", len(replay_buffer), step)

        # Value target distribution diagnostics — tracks whether climbing value
        # loss is caused by distributional shift (targets moving towards ±1 as
        # games become more decisive) vs a genuine prediction failure.
        _vt = np.array([e[2] for e in examples + mm_mix_examples], dtype=np.float32)
        _vt_mean   = float(_vt.mean())
        _vt_std    = float(_vt.std())
        _vt_absm   = float(np.abs(_vt).mean())
        _vt_near1  = float(np.mean(np.abs(_vt) > 0.9))   # fraction of decisive outcomes
        print(f"  Value targets: mean={_vt_mean:+.3f}  std={_vt_std:.3f}"
              f"  |v|={_vt_absm:.3f}  decisive(|v|>0.9)={_vt_near1:.1%}")
        writer.add_scalar("value_targets/mean",        _vt_mean,  step)
        writer.add_scalar("value_targets/std",         _vt_std,   step)
        writer.add_scalar("value_targets/abs_mean",    _vt_absm,  step)
        writer.add_scalar("value_targets/frac_decisive", _vt_near1, step)
        writer.add_histogram("value_targets/dist", _vt, step)

        # ── 3. Train ──────────────────────────────────────────────────────
        print(f"\nTraining ({EPOCHS_PER_ITERATION} epochs, batch {BATCH_SIZE})...")
        train_start = time.time()
        losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITERATION, device=device,
        )
        train_time = time.time() - train_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  policy={losses['policy_loss']:.4f}  value={losses['value_loss']:.4f}"
              f"  total={losses['total_loss']:.4f}  ({train_time:.1f}s)")

        scheduler.step(losses['total_loss'])

        writer.add_scalar("train/policy_loss",        losses['policy_loss'], step)
        writer.add_scalar("train/value_loss",         losses['value_loss'],  step)
        writer.add_scalar("train/total_loss",         losses['total_loss'],  step)
        writer.add_scalar("train/lr",                 current_lr,            step)
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

        # ── 4. Evaluate ───────────────────────────────────────────────────
        if is_eval_iter:
            # Eval ladder tracks playing-strength progress independently of value curriculum.
            # ladder_progress = eval_level + wr_vs_current (continuous 0→len(EVAL_LADDER)).
            if eval_level < len(EVAL_LADDER):
                eval_cur_depth, eval_cur_noise, eval_cur_label = EVAL_LADDER[eval_level]
                level_base = eval_level
            else:
                eval_cur_depth, eval_cur_noise, eval_cur_label = EVAL_LADDER[-1]
                level_base = len(EVAL_LADDER)

            print(f"\nEvaluating vs {eval_cur_label} ...")

            wr_cur, res_cur = evaluate_vs_noisy_minimax(
                network, minimax_depth=eval_cur_depth, noise=eval_cur_noise,
                num_games=EVAL_GAMES, num_simulations=EVAL_SIMULATIONS,
                num_actions=num_actions, mcts_cls=MCGS, mcts_kwargs=_mcts_kwargs,
            )
            label = f"lvl {eval_level}" if eval_level < len(EVAL_LADDER) else "post-ladder"
            print(f"  vs {eval_cur_label} ({label}): {wr_cur:.0%} "
                  f"W:{res_cur['wins']} L:{res_cur['losses']} D:{res_cur['draws']}  "
                  f"(B:{res_cur['wr_as_blue']:.0%} G:{res_cur['wr_as_green']:.0%})")
            writer.add_scalar("eval/win_rate_current",  wr_cur,                   step)
            writer.add_scalar("eval/win_rate_as_blue",  res_cur['wr_as_blue'],    step)
            writer.add_scalar("eval/win_rate_as_green", res_cur['wr_as_green'],   step)
            writer.add_scalar("eval/ladder_progress",   level_base + wr_cur,      step)
            writer.add_scalar("eval/ladder_level",      eval_level,               step)

            writer.add_scalar("curriculum/win_rate", wr_cur, step)

            # ── Eval ladder advancement ────────────────────────────────────
            if eval_level < len(EVAL_LADDER):
                if wr_cur >= EVAL_ADVANCE_THRESHOLD:
                    eval_consecutive += 1
                    print(f"  Ladder beat ({eval_consecutive}/{EVAL_ADVANCE_CONSECUTIVE})")
                else:
                    eval_consecutive = 0

                if eval_consecutive >= EVAL_ADVANCE_CONSECUTIVE:
                    eval_consecutive = 0
                    eval_level += 1
                    if eval_level >= len(EVAL_LADDER):
                        print("  Eval ladder complete!")
                        writer.add_scalar("eval/ladder_level", eval_level, step)
                    else:
                        next_depth = EVAL_LADDER[eval_level][0]
                        print(f"  Advancing eval ladder to MM-{next_depth}")

        # ── 5. Memory accounting ──────────────────────────────────────────
        try:
            import psutil as _psutil
            rss = _psutil.Process().memory_info().rss / 1024**2
            writer.add_scalar("system/rss_mb", rss, step)
        except Exception:
            pass

        # ── 6. Checkpoint ─────────────────────────────────────────────────
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
