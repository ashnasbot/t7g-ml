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
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.bc import (                                                  # noqa: E402
    generate_bc_warmup_data,
    BC_WARMUP_DEPTH, BC_WARMUP_TEMP, BC_RANDOM_OPENS, BC_BLUNDER_RATE,
)
from lib.device_utils import load_compiled_network                   # noqa: E402
from lib.dual_network import DualHeadNetwork                         # noqa: E402
from lib.evaluation import (                                         # noqa: E402
    evaluate_vs_noisy_minimax, rate_vs_pool, _calibrate_ladder,
)
from lib.mcgs import MCGS                                            # noqa: E402
from lib.t7g import (                                                # noqa: E402
    soft_policy_from_mm, new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells,
)
from lib.train_workers import (                                      # noqa: E402
    self_play_game_pool, generate_mm_mix_data,
)
from lib.training import train_network, _IterBuffer                  # noqa: E402


# ============================================================
# Configuration
# ============================================================

# Base
NUM_ITERATIONS       = 500
GAMES_PER_ITERATION  = 250
MCTS_SIMULATIONS     = 500
EVAL_SIMULATIONS     = 500
BATCH_SIZE           = 256
EPOCHS_PER_ITERATION = 1        # 1 pass/iter; with an 8-iter buffer each example
                                # still gets ~8 gradient passes over its lifetime
                                # (5 epochs = ~40 passes -> value-head memorization)
REPLAY_BUFFER_ITERS  = 8        # keep this many iterations of self-play data
TARGET_EXAMPLES_ITER = 54_000   # adaptive games/iter targets this example count
                                # (35k policy from full games + ~19k value-only
                                # from fast games at FAST_PLAY_FRACTION=0.35)
POOL_SIZE            = 64       # concurrent games per worker

# Model parameters
VALUE_BLEND_ALPHA    = 1.0      # α=1 → pure terminal target (blending off).
                                # α<1 mixes gated root-Q into the value target
                                # AT THE LOSS as soft WDL class probabilities
                                # (1-w)*onehot(z) + w*[(1+q)/2, 0, (1-q)/2] -
                                # pre-blending the scalar would be quantized
                                # away by the hard ±0.33 class thresholds.
                                # Disabled 2026-04 when the value head was
                                # broken (memory/project_value_head_broken.md);
                                # that predates the WDL head + search fixes.
VALUE_COEF           = 1.0      # value-loss weight; bumped to rebalance against
                                # policy CE's 1225-class gradient magnitude
MARGIN_COEF          = 0.4      # auxiliary margin-head loss weight (final
                                # material margin / 49) - dense KataGo-style
                                # signal that trains the trunk from every game
WDL_VALUE            = True     # 3-way W/D/L cross-entropy value head (fixes
                                # the tanh+MSE saturation pathology: 2026-07-10
                                # audit found 10% of positions confidently wrong
                                # with ~zero gradient to unlearn them)
OWNERSHIP_AUX        = True     # per-cell final-ownership aux head (KataGo);
OWNERSHIP_COEF       = 0.15     # 49 dense spatial targets/position teach the
                                # trunk to count material.  Aux TRAINING signal
                                # only - never enters search utility or move
                                # selection (margin-maximizing play is a known
                                # failure state)
DIRICHLET_ALPHA      = 0.0      # OFF: Gumbel top-K sampling already explores;
DIRICHLET_EPS        = 0.0      # root noise leaked into the completed-Q policy
                                # targets and permanently corrupted TT priors
GAMES_MIN            = 50
GAMES_MAX            = 650
LEARNING_RATE        = 1.0e-4  # was 1.5e-4 for the ladder climb; lowered 2026-07-09
                               # for the post-peak refinement regime (run-1 declined
                               # monotonically after iter 50 - churn, not signal)
WEIGHT_DECAY         = 1e-4
C_PUCT               = 1.3
GUMBEL_K             = 16
SIGMA_SCALE          = 1.0      # multiplier on the Gumbel sigma(q) transform;
                                # <1 makes completed-Q targets stickier to the
                                # prior (probe knob for value-noise amplification)
COMPLETION_N0        = 50.0     # visit-shrinkage prior strength in the completed-Q
                                # target: q~(a)=(n_a*q_a+n0*v_root)/(n_a+n0).  Caps
                                # low-visit Q noise before sigma amplifies it into
                                # target logits (2026-07-11 target-noise fix; the
                                # temp-0 played move is the SH winner, also part of
                                # that fix)
SELF_PLAY_TEMP_MOVES = 16       # was 30 (~half a 73-move game adding value noise);
                                # sigma-scaled Gumbel targets need less forcing
ENTROPY_COEFF        = 0.00
FAST_PLAY_FRACTION   = 0.35  # fraction of games run fast (value-only); 0=disabled
FAST_PLAY_SIMS       = 100   # simulations for fast games

# MM Mix
MM_MIX_GAMES         = 100
MM_MIX_DEPTH         = 5
MM_MIX_POOL_SIZE     = 32
MM_MIX_WORKERS       = 16
MM_MIX_RETIRE_LEVEL  = 0        # set > 0 to enable MM-mix up to that ladder level

# Eval
EVAL_INTERVAL        = 5
EVAL_GAMES           = 30       # 10 games/rung made the ladder oscillate 0->100%
EVAL_WORKERS         = 4
CHECKPOINT_INTERVAL  = 10
CHECKPOINT_DIR       = "models/mcts"
EVAL_LADDER = [
    (1, 0.60, "MM1-semi"),
    (1, 0.20, "MM1-noisy"),
    (1, 0.00, "MM1"),
    (2, 0.00, "MM2"),
    (3, 0.00, "MM3"),
    (4, 0.00, "MM4"),
    (5, 0.00, "MM5"),
]
EVAL_ADVANCE_THRESHOLD   = 0.90
EVAL_ADVANCE_CONSECUTIVE = 2
# Elo anchor pool (see models/elo_pool/pool.json for how anchors were rated)
ELO_POOL_PATH          = "models/elo_pool/pool.json"
ELO_GAMES_PER_OPPONENT = 8
ELO_PROMOTE_MARGIN     = 35     # candidate elo must beat the best-net bar by this
                                # much to take over self-play (24-game rating is
                                # ~+/-70 noisy; the margin filters false promotions
                                # while a stall just leaves data at the proven best)
ELO_ROLLING_WINDOW     = 2      # self-anchored rating: on each promotion the new
                                # best is appended to the pool as an opponent, and
                                # only the newest N such self-anchors are kept.  A
                                # fixed set of anchors always saturates once the net
                                # sweeps them; measuring against recent selves keeps
                                # an opponent near current strength so the Elo (and
                                # promotion bar) never ceilings.  The fixed engine +
                                # seed-net anchors (pool.json "fixed": true) stay to
                                # pin the absolute scale / flag regressions.  Can be
                                # overridden by "rolling_window" in pool.json.

# Policy distillation
POLICY_DISTILL_DEPTH = 3
POLICY_DISTILL_TEMP  = 0.075


# ============================================================
# Self-play data generation
# ============================================================

def generate_self_play_data(
    mcts: MCGS,
    min_non_truncated: int = 50,
    mcts_pool: 'list | None' = None,
    policy_relabel_fn=None,
    temp_moves: int = 0,
) -> tuple:
    """
    Generate training examples in-process using the main network directly.

    Plays games until min_non_truncated non-truncated games have been collected.
    Truncated games are counted but their examples are discarded.

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
    raw_examples: list = []

    pbar = tqdm(desc="Self-play", unit="game", total=min_non_truncated)
    try:
        while non_truncated_count < min_non_truncated:
            target = min_non_truncated - non_truncated_count
            for game_examples, winner, moves, gtime, trunc, legal_counts in (
                self_play_game_pool(mcts, POOL_SIZE, target, mcts_pool,
                                    temp_moves=temp_moves,
                                    blend_alpha=VALUE_BLEND_ALPHA)
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

    if policy_relabel_fn:
        _fn = policy_relabel_fn
        n_workers = max(1, (os.cpu_count() or 4) // 2)
        chunk_size = max(32, len(raw_examples) // (n_workers * 4))
        chunks = [raw_examples[i:i + chunk_size]
                  for i in range(0, len(raw_examples), chunk_size)]

        def _relabel_chunk(chunk):
            return [_fn(e[5], e[6]) for e in chunk]

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            policy_chunks = list(ex.map(_relabel_chunk, chunks))
        policies = [p for pc in policy_chunks for p in pc]
        all_examples = [
            (obs, pol, val, margin, own, rq, qw)
            for (obs, _, val, margin, own, _, _, rq, qw), pol in zip(raw_examples, policies)
        ]
    else:
        all_examples = [(obs, p, v, m, o, rq, qw)
                        for obs, p, v, m, o, _, _, rq, qw in raw_examples]

    avg_branching = float(np.mean(all_legal_counts)) if all_legal_counts else 0.0
    return (all_examples, (blue_wins, green_wins, draws),
            game_moves, game_times, truncations, avg_branching)


# ============================================================
# Main training loop
# ============================================================

def main():
    global VALUE_BLEND_ALPHA
    parser = argparse.ArgumentParser(description="AlphaZero MCTS Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS,
                        help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=GAMES_PER_ITERATION,
                        help="Self-play games per iteration")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS,
                        help="Total training iterations")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate, constant for the whole run "
                             f"(default: {LEARNING_RATE}); lower it between "
                             f"runs when eval plateaus")
    parser.add_argument("--logdir", type=str, default="tblog/mcts",
                        help="TensorBoard log root directory")
    parser.add_argument("--relabel", action="store_true",
                        help="Use MM policy distillation to relabel MCTS visit-count targets")
    parser.add_argument("--bc-warmup", type=int, default=0, metavar="N",
                        help="Pre-fill replay buffer with N BC games before iteration 1")
    parser.add_argument("--bc-depth", type=int, default=BC_WARMUP_DEPTH, metavar="D",
                        help=f"Minimax depth for BC data generation (default: {BC_WARMUP_DEPTH})")
    parser.add_argument("--bc-epochs", type=int, default=100, metavar="N",
                        help="Training epochs on BC data before self-play begins (default: 100)")
    parser.add_argument("--bc-cache", type=str, default=None, metavar="PATH",
                        help="Path to save/load BC data (.npz). Use 'auto' to derive from params.")
    parser.add_argument("--blend-alpha", type=float, default=VALUE_BLEND_ALPHA,
                        help=f"Value-target blend: 1.0 = pure game outcome, "
                             f"<1 mixes gated root-Q into soft WDL targets, "
                             f"max Q weight = 1-alpha (default: {VALUE_BLEND_ALPHA})")
    args = parser.parse_args()

    VALUE_BLEND_ALPHA = args.blend_alpha

    policy_relabel_fn = (
        lambda board, turn: soft_policy_from_mm(board, POLICY_DISTILL_DEPTH, turn,
                                                POLICY_DISTILL_TEMP)
    ) if args.relabel else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    num_actions = 1225
    _net_kwargs = dict(wdl=WDL_VALUE, ownership=OWNERSHIP_AUX)
    network   = DualHeadNetwork(num_actions=num_actions, **_net_kwargs).to(device)
    inference_network = (  # type: ignore[assignment]
        torch.compile(network) if device.type == "cuda" else network
    )
    # Constant LR: the pipeline is run in chunks (resume via --checkpoint), so
    # a per-run schedule would sawtooth on every restart.  Lower manually with
    # --lr between runs if eval goes flat while losses oscillate.
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = None

    replay_buffer = _IterBuffer(maxiters=REPLAY_BUFFER_ITERS)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        _ckpt_sd = checkpoint['network']
        if DualHeadNetwork.infer_arch(_ckpt_sd) == _net_kwargs:
            network.load_state_dict(_ckpt_sd)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr
                print(f"  Restored optimizer state (LR overridden to {args.lr})")
        else:
            # Cross-architecture warm start (e.g. legacy tanh-value checkpoint
            # into a wdl/ownership net): transfer every shape-compatible
            # tensor, leave the rest at fresh init.  Optimizer state is NOT
            # restored - its moments refer to the old parameterization.
            _ref = network.state_dict()
            _ok = {k: v for k, v in _ckpt_sd.items()
                   if k in _ref and _ref[k].shape == v.shape}
            network.load_state_dict(_ok, strict=False)
            _fresh = sorted(set(_ref) - set(_ok))
            print(f"  Cross-arch warm start: {len(_ok)}/{len(_ref)} tensors "
                  f"transferred; fresh init: {', '.join(_fresh)}")
        saved_iter = checkpoint.get('iteration', 0) + 1
        print(f"Loaded weights from iteration {saved_iter}; "
              f"training for {args.iterations} fresh iterations")

    # Best-so-far net: generates ALL self-play data (the ratchet).  The training
    # net only takes over data generation when its pool Elo clears the bar -
    # run 1 showed ungated self-play walks downhill from the peak (270 Elo lost
    # over iters 50-180) because nothing enforces monotonic strength.
    best_network = DualHeadNetwork(num_actions=num_actions, **_net_kwargs).to(device)
    best_network.load_state_dict(network.state_dict())
    best_network.eval()
    best_inference_network = (  # type: ignore[assignment]
        torch.compile(best_network) if device.type == "cuda" else best_network
    )
    best_elo: 'float | None' = None   # promotion bar; set at the first pool rating

    # Elo pool: a fixed part (engine + seed-net anchors that pin the absolute
    # scale) plus a rolling part of recent promoted selves appended below, so the
    # net always has an opponent near its own strength and the rating can't
    # saturate.  The current net's Elo is solved against the whole pool each eval.
    # Members carry "fixed": True (never evicted) or False (rolling self-anchor).
    elo_pool: list = []
    rolling_window = ELO_ROLLING_WINDOW
    if os.path.exists(ELO_POOL_PATH):
        with open(ELO_POOL_PATH) as _f:
            _pool_cfg = json.load(_f)
        rolling_window = _pool_cfg.get("rolling_window", ELO_ROLLING_WINDOW)
        for _m in _pool_cfg["members"]:
            if _m["kind"] == "net":
                _blob = torch.load(_m["path"], map_location="cpu", weights_only=True)
                _payload = _blob["network"] if "network" in _blob else _blob
            else:
                _payload = _m["depth"]
            elo_pool.append({"name": _m["name"], "kind": _m["kind"],
                             "payload": _payload, "elo": _m["elo"],
                             "fixed": _m.get("fixed", True)})
        print(f"Elo pool: {', '.join(m['name'] for m in elo_pool)} "
              f"(rolling window {rolling_window} self-anchors)")
    else:
        print(f"Elo pool: none ({ELO_POOL_PATH} not found; eval/elo disabled)")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(args.logdir, run_name)
    writer   = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    writer.add_custom_scalars({
        "Game Stats": {
            "Policy Entropy": ["Multiline", ["self_play/avg_policy_entropy"]],
        },
        "Eval": {
            "Ladder Progress": ["Multiline", ["eval/ladder_progress"]],
            "Win Rate by Colour": ["Multiline", [
                "eval/win_rate_as_blue",
                "eval/win_rate_as_green",
            ]],
            "Elo": ["Multiline", ["eval/elo", "eval/best_elo"]],
        },
        "Training": {
            "Iteration Loss": ["Multiline", [
                "train/policy_loss",
                "train/value_loss",
            ]],
        },
    })

    eval_level: int = 0
    eval_consecutive: int = 0

    print("=" * 60)
    print("AlphaZero MCTS Training for Microscope")
    print("=" * 60)
    print(f"Iterations:       {args.iterations}")
    print(f"Games/iteration:  {args.games}")
    print(f"Sims/move:        {args.simulations}")
    print(f"Replay buffer:    last {REPLAY_BUFFER_ITERS} iterations")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs/iteration: {EPOCHS_PER_ITERATION}")
    print(f"Value blend:      alpha={VALUE_BLEND_ALPHA}"
          + ("  (pure terminal)" if VALUE_BLEND_ALPHA >= 1.0
             else f"  (max Q weight {1.0 - VALUE_BLEND_ALPHA:.2f}, gated)"))
    print(f"Eval ladder:      {' > '.join(lbl for _, _, lbl in EVAL_LADDER)} > retire")
    print("=" * 60)

    _mcts_kwargs = dict(num_simulations=args.simulations, c_puct=C_PUCT, gumbel_k=GUMBEL_K,
                        sigma_scale=SIGMA_SCALE, completion_n0=COMPLETION_N0)
    _sp_kwargs   = dict(**_mcts_kwargs,
                        dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_eps=DIRICHLET_EPS)
    # AlphaGo-Zero-style ratchet: self-play uses the BEST net, not the current
    # one.  The search targets are still one improvement step ahead of the data
    # generator, so the training net can (and must) surpass it to get promoted;
    # eval/elo vs the anchor pool is both the progress metric and the gate.
    self_play_mcts = MCGS(best_inference_network, **_sp_kwargs)
    mcts_pool = [
        MCGS(best_inference_network, **_sp_kwargs)
        for _ in range(POOL_SIZE)
    ]
    try:
        term_width, term_height = os.get_terminal_size()
    except OSError:  # no tty (piped/headless run)
        term_width, term_height = 80, 24

    if device.type == "cuda":
        print("Warming up torch.compile ...", end=" ", flush=True)
        network.eval()
        with torch.no_grad():
            _w = torch.zeros(1, 7, 7, 4, device=device)
            inference_network(_w)       # type: ignore[operator]
            best_inference_network(_w)  # type: ignore[operator]
        network.train()
        print("done")

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
            value_coef=VALUE_COEF,
        )
        print(f"  BC pre-train: policy={bc_losses['policy_loss']:.4f}"
              f"  value={bc_losses['value_loss']:.4f}")

    if args.checkpoint:
        eval_level, _ = _calibrate_ladder(
            network, num_actions, MCGS, _mcts_kwargs,
            eval_ladder=EVAL_LADDER, eval_simulations=EVAL_SIMULATIONS,
            num_workers=EVAL_WORKERS,
        )
        eval_consecutive = 0
        print(f"  Starting at ladder level {eval_level}"
              + (f" ({EVAL_LADDER[eval_level][2]})" if eval_level < len(EVAL_LADDER) else " (complete)"))

    _epg_ema: float | None = None
    games_this_iter = args.games

    iter_pbar = tqdm(range(args.iterations), desc="Training", unit="iter")
    for iteration in iter_pbar:
        iter_start = time.time()
        iter_pbar.set_description(f"Iter {iteration + 1}/{args.iterations}")
        step = iteration + 1

        #  Self-play
        print("\n")
        gen_start = time.time()
        network.eval()

        full_games = (max(1, round(games_this_iter * (1 - FAST_PLAY_FRACTION)))
                      if FAST_PLAY_FRACTION > 0 else games_this_iter)
        fast_games = games_this_iter - full_games

        examples, (bw, gw, dr), game_moves, game_times, truncations, avg_branching = (
            generate_self_play_data(
                self_play_mcts,
                min_non_truncated=full_games,
                mcts_pool=mcts_pool,
                policy_relabel_fn=policy_relabel_fn,
                temp_moves=SELF_PLAY_TEMP_MOVES,
            )
        )

        if fast_games > 0:
            fast_mcts = MCGS(inference_network,
                             num_simulations=FAST_PLAY_SIMS, c_puct=C_PUCT, gumbel_k=GUMBEL_K,
                             dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_eps=DIRICHLET_EPS,
                             completion_n0=COMPLETION_N0)
            fast_raw, (fbw, fgw, fdr), fmoves, ftimes, ftrunc, _ = generate_self_play_data(
                fast_mcts,
                min_non_truncated=fast_games,
                temp_moves=999,
            )
            # Zero policy targets — these examples are value-only
            fast_examples = [(obs, np.zeros_like(pol), val, m, o, rq, qw)
                             for obs, pol, val, m, o, rq, qw in fast_raw]
            bw += fbw; gw += fgw; dr += fdr
            game_moves += fmoves; game_times += ftimes; truncations += ftrunc
            examples += fast_examples

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

        print(" " * term_width + "\r", end='')  # Clear the tqdm bar
        fast_tag = f" ({fast_games} fast)" if fast_games > 0 else ""
        print(f"  Self-play  {len(examples):>6} ex  {gen_time:.0f}s  {sims_per_sec:.0f} sim/s"
              f"  B:{bw} G:{gw} D:{dr}{fast_tag}")
        print(f"  Games      avg {avg_moves:.1f}  med {med_moves:.1f}  std {std_moves:.1f}"
              f"  [{int(moves_arr.min())}-{int(moves_arr.max())}]"
              f"  trunc {trunc_pct:.1f}%  branch {avg_branching:.1f}")

        _epg = len(examples) / max(1, games_this_iter)
        _epg_ema = _epg if _epg_ema is None else 0.7 * _epg_ema + 0.3 * _epg
        assert _epg_ema is not None
        games_this_iter = max(GAMES_MIN, min(GAMES_MAX,
                                             round(TARGET_EXAMPLES_ITER / _epg_ema)))
        writer.add_scalar("self_play/examples_per_game",    _epg,                  step)
        writer.add_scalar("self_play/games_this_iter",      games_this_iter,       step)
        writer.add_scalar("self_play/examples_generated",   len(examples),         step)
        writer.add_scalar("self_play/avg_game_moves",        avg_moves,             step)
        writer.add_histogram("self_play/game_moves_dist",   moves_arr,             step)
        writer.add_scalar("self_play/truncation_pct",       trunc_pct,             step)
        writer.add_scalar("self_play/avg_branching_factor", avg_branching,         step)

        entropies = []
        for _, policy_target, *_ in examples:
            p = policy_target[policy_target > 0]
            if p.size > 0:
                entropies.append(float(-np.sum(p * np.log(p))))
        avg_policy_entropy = float(np.mean(entropies)) if entropies else 0.0
        print(f"  Policy     entropy {avg_policy_entropy:.3f}"
              f"  epg {_epg:.1f} (ema {_epg_ema:.1f})  -> {games_this_iter} next")
        writer.add_scalar("self_play/avg_policy_entropy", avg_policy_entropy, step)
        if entropies:
            writer.add_histogram("self_play/policy_entropy_dist", np.array(entropies), step)

        writer.add_scalar("timing/gen_seconds",      gen_time,     step)
        writer.add_scalar("timing/avg_game_seconds", avg_gtime,    step)
        writer.add_scalar("timing/sims_per_sec",     sims_per_sec, step)

        is_eval_iter = (step % EVAL_INTERVAL == 0)

        #  MM-mix 
        if eval_level < MM_MIX_RETIRE_LEVEL:
            network.eval()
            mm_mix_examples = generate_mm_mix_data(
                self_play_mcts, MM_MIX_GAMES,
                mm_depth=MM_MIX_DEPTH, pool_size=MM_MIX_POOL_SIZE, n_workers=MM_MIX_WORKERS,
            )
            network.train()
        else:
            mm_mix_examples = []
        writer.add_scalar("self_play/mm_mix_examples", len(mm_mix_examples), step)

        replay_buffer.append_batch(examples + mm_mix_examples)
        writer.add_scalar("self_play/buffer_size", len(replay_buffer), step)

        _iter_hashes  = {e[0].tobytes() for e in examples}
        _buf_hashes   = {e[0].tobytes() for e in replay_buffer}
        _uniq_iter    = len(_iter_hashes)
        _uniq_buf     = len(_buf_hashes)
        _uniq_iter_pct = _uniq_iter / max(1, len(examples))
        _uniq_buf_pct  = _uniq_buf  / max(1, len(replay_buffer))
        print(f"  Buffer     {len(replay_buffer):>6} ex"
              f"  uniq {_uniq_iter_pct:.0%} iter / {_uniq_buf_pct:.0%} buf")
        writer.add_scalar("self_play/unique_positions_iter",   _uniq_iter,     step)
        writer.add_scalar("self_play/unique_positions_buffer", _uniq_buf,      step)
        writer.add_scalar("self_play/unique_pct_iter",         _uniq_iter_pct, step)
        writer.add_scalar("self_play/unique_pct_buffer",       _uniq_buf_pct,  step)

        #  Train 
        train_start = time.time()
        losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITERATION, device=device,
            entropy_coeff=ENTROPY_COEFF,
            value_coef=VALUE_COEF,
            margin_coef=MARGIN_COEF,
            ownership_coef=OWNERSHIP_COEF,
        )
        train_time = time.time() - train_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train      pol {losses['policy_loss']:.4f}  val {losses['value_loss']:.4f}"
              f"  marg {losses['margin_loss']:.4f}  own {losses['ownership_loss']:.4f}"
              f"  tot {losses['total_loss']:.4f}"
              f"  sign {losses['sign_acc']:.1%}  {train_time:.0f}s")

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("train/policy_loss",    losses['policy_loss'], step)
        writer.add_scalar("train/value_loss",     losses['value_loss'],  step)
        writer.add_scalar("train/margin_loss",    losses['margin_loss'], step)
        writer.add_scalar("train/ownership_loss", losses['ownership_loss'], step)
        writer.add_scalar("train/total_loss",     losses['total_loss'],  step)
        writer.add_scalar("train/value_sign_acc", losses['sign_acc'],    step)
        writer.add_scalar("train/lr",             current_lr,            step)
        writer.add_scalar("timing/train_seconds", train_time,            step)

        _sample_n = min(1024, len(replay_buffer))
        _buf = list(replay_buffer)
        _idx = np.random.choice(len(_buf), _sample_n, replace=False)
        _obs = torch.from_numpy(np.array([_buf[i][0] for i in _idx])).to(device)
        network.eval()
        with torch.no_grad():
            _, _val_preds, _marg_preds = network(_obs)
        network.train()
        writer.add_histogram("train/value_output_dist",  _val_preds.squeeze().cpu(),  step)
        writer.add_histogram("train/margin_output_dist", _marg_preds.squeeze().cpu(), step)

        #  Eval
        if is_eval_iter:
            # MM ladder: only while incomplete.  Once beaten it stops running -
            # a saturated rung is pure eval overhead (every rung beaten ~100%),
            # and the Elo pool's MM5 anchor still ties ratings to the engines.
            if eval_level < len(EVAL_LADDER):
                eval_cur_depth, eval_cur_noise, eval_cur_label = EVAL_LADDER[eval_level]

                wr_cur, res_cur = evaluate_vs_noisy_minimax(
                    network, minimax_depth=eval_cur_depth, noise=eval_cur_noise,
                    num_games=EVAL_GAMES, num_simulations=EVAL_SIMULATIONS,
                    num_actions=num_actions, mcts_cls=MCGS, mcts_kwargs=_mcts_kwargs,
                    engine='micro3', num_workers=EVAL_WORKERS,
                )
                print(f"  Eval       {eval_cur_label}  {wr_cur:.0%}"
                      f"  W:{res_cur['wins']} L:{res_cur['losses']} D:{res_cur['draws']}"
                      f"  B:{res_cur['wr_as_blue']:.0%} G:{res_cur['wr_as_green']:.0%}"
                      f"  t:{res_cur['n_terminal']} r:{res_cur['n_repetition']}"
                      f" x:{res_cur['n_truncated']}")
                writer.add_scalar("eval/win_rate_as_blue",  res_cur['wr_as_blue'],    step)
                writer.add_scalar("eval/win_rate_as_green", res_cur['wr_as_green'],   step)
                writer.add_scalar("eval/ladder_progress",   eval_level + wr_cur,      step)
                writer.add_scalar("eval/n_terminal",        res_cur['n_terminal'],    step)
                writer.add_scalar("eval/n_repetition",      res_cur['n_repetition'],  step)
                writer.add_scalar("eval/n_truncated",       res_cur['n_truncated'],   step)

            # Elo vs the fixed anchor pool - the primary progress metric once
            # the MM ladder saturates (every net past ~1100 beats MM5 ~100%).
            if elo_pool:
                elo, elo_res = rate_vs_pool(
                    network, elo_pool,
                    games_per_opponent=ELO_GAMES_PER_OPPONENT,
                    num_actions=num_actions,
                    mcts_kwargs=dict(_mcts_kwargs, num_simulations=EVAL_SIMULATIONS),
                    num_workers=EVAL_WORKERS,
                )
                writer.add_scalar("eval/elo", elo, step)
                detail = "  ".join(f"{n}:{w}-{d}-{ls}"
                                   for n, (w, d, ls) in elo_res.items())
                print(f"  Elo        {elo:.0f}  ({detail})")

                # Ratchet: promote the training net to data generator only when
                # it provably outrates the incumbent best.
                if best_elo is None:
                    best_elo = elo
                    print(f"  Ratchet    bar set at {elo:.0f}")
                elif elo >= best_elo + ELO_PROMOTE_MARGIN:
                    best_network.load_state_dict(network.state_dict())
                    best_network.eval()
                    best_elo = elo
                    # Self-anchor: the just-promoted net joins the pool as a rolling
                    # opponent so future ratings are measured against current-strength
                    # play (never saturates).  Keep only the newest ELO_ROLLING_WINDOW.
                    elo_pool.append({
                        "name": f"self_iter{step:04d}", "kind": "net",
                        "payload": {k: v.detach().cpu()
                                    for k, v in network.state_dict().items()},
                        "elo": elo, "fixed": False})
                    _rolling = [m for m in elo_pool if not m["fixed"]]
                    for _old in _rolling[:-rolling_window]:
                        elo_pool.remove(_old)
                    # Persist the promoted net so it can be rated offline
                    # (scripts/eval_db.py).  The in-memory self-anchor above is
                    # ephemeral - lost on restart - and every-N checkpoints miss
                    # the exact promotion step; this is the rateable frontier net.
                    promoted_path = os.path.join(CHECKPOINT_DIR, f"promoted_iter{step:04d}.pt")
                    torch.save({'iteration': iteration, 'network': network.state_dict(),
                                'elo': float(elo)}, promoted_path)
                    print(f"  Ratchet    PROMOTED - self-play net now {elo:.0f} "
                          f"(pool: {', '.join(m['name'] for m in elo_pool)}) "
                          f"-> {promoted_path}")
                else:
                    print(f"  Ratchet    retained (needs {best_elo + ELO_PROMOTE_MARGIN:.0f})")
                writer.add_scalar("eval/best_elo", best_elo, step)

            # Ladder advancement
            if eval_level < len(EVAL_LADDER):
                if wr_cur >= EVAL_ADVANCE_THRESHOLD:
                    eval_consecutive += 1
                    print(f"  Ladder     beat {eval_consecutive}/{EVAL_ADVANCE_CONSECUTIVE}")
                else:
                    eval_consecutive = 0

                if eval_consecutive >= EVAL_ADVANCE_CONSECUTIVE:
                    eval_consecutive = 0
                    eval_level += 1
                    if eval_level < len(EVAL_LADDER):
                        print(f"  Ladder     -> {EVAL_LADDER[eval_level][2]}")
                    else:
                        print("  Ladder     complete!")

                    # Fast promotion: immediately test each higher rung.
                    while eval_level < len(EVAL_LADDER):
                        fp_depth, fp_noise, fp_label = EVAL_LADDER[eval_level]
                        fp_wr, fp_res = evaluate_vs_noisy_minimax(
                            network, minimax_depth=fp_depth, noise=fp_noise,
                            num_games=EVAL_GAMES, num_simulations=EVAL_SIMULATIONS,
                            num_actions=num_actions, mcts_cls=MCGS, mcts_kwargs=_mcts_kwargs,
                            engine='micro3', num_workers=EVAL_WORKERS,
                        )
                        print(f"  Fast-promo {fp_label}  {fp_wr:.0%}"
                              f"  W:{fp_res['wins']} L:{fp_res['losses']} D:{fp_res['draws']}"
                              f"  B:{fp_res['wr_as_blue']:.0%} G:{fp_res['wr_as_green']:.0%}")
                        writer.add_scalar("eval/ladder_progress", eval_level + fp_wr, step)
                        if fp_wr >= EVAL_ADVANCE_THRESHOLD:
                            eval_level += 1
                            if eval_level < len(EVAL_LADDER):
                                print(f"  Ladder     -> {EVAL_LADDER[eval_level][2]}!")
                            else:
                                print("  Ladder     complete!")
                        else:
                            break

        #  Housekeeping 
        try:
            import psutil as _psutil
            rss = _psutil.Process().memory_info().rss / 1024**2
            writer.add_scalar("system/rss_mb", rss, step)
        except Exception:
            pass

        if step % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"iter_{step:04d}.pt")
            torch.save({
                'iteration': iteration,
                'network':   network.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint {ckpt_path}")

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
    multiprocessing.set_start_method("spawn", force=True)
    main()
