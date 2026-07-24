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

# Cap CPU math-lib threads to 1 BEFORE numpy/torch import.  This project's hot
# path is a single-threaded batched game pool + GPU; left at the default (all
# 32 cores on framework) the OMP/BLAS pools only spin-wait between tiny per-step
# ops -- ~20 cores burned + clocks throttled for zero throughput.  setdefault so
# an explicit launch-env override still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.device_utils import get_gpu_stats                          # noqa: E402
from lib.dual_network import DualHeadNetwork                         # noqa: E402
from lib.net2 import Net2                                            # noqa: E402
from lib.evaluation import (                                         # noqa: E402
    evaluate_vs_noisy_minimax, rate_vs_pool, h2h_gate, _calibrate_ladder,
)
from lib.mcgs import MCGS                                            # noqa: E402
from lib.t7g import action_to_move                                   # noqa: E402

# Boolean mask over the 1225-action space: True where the move is a jump
# (source vacated) rather than a clone (source retained).  Ataxx strategy
# hinges on preferring clones; the search's jump-mass is its style fingerprint.
JUMP_MASK = np.array([action_to_move(a)[4] for a in range(1225)], dtype=bool)
from lib.train_workers import self_play_game_pool                    # noqa: E402
from lib.training import train_network, _IterBuffer                  # noqa: E402


# ============================================================
# Configuration
# ============================================================

def _env_int(name: str, default: int) -> int:
    """Platform-tuning override from the environment (T7G_*).

    Machine-dependent performance knobs (pool size, workers, batch size,
    sim budgets) read their default from the environment so each box can
    bank its own sweep results (e.g. `export T7G_POOL_SIZE=256` on the
    desktop vs 512 on the framework) without editing this file.  Science
    hyperparameters stay static here on purpose - they are experiment
    state, not machine state.  Precedence: cmdline > T7G_* env > default.
    """
    v = os.environ.get(name)
    return int(v) if v else default


# Base
NUM_ITERATIONS       = 500
GAMES_PER_ITERATION  = 1200     # iter-1 seed for the adaptive games/iter loop
                                # (PCR steady state is ~1250/iter)
MCTS_SIMULATIONS     = _env_int("T7G_SIMULATIONS", 500)
EVAL_SIMULATIONS     = _env_int("T7G_EVAL_SIMULATIONS", 500)
BATCH_SIZE           = _env_int("T7G_BATCH_SIZE", 256)
EPOCHS_PER_ITERATION = 1        # 1 pass/iter; over a 3-iter buffer each example
                                # gets ~3 gradient passes (more limits value-head
                                # memorization)
REPLAY_BUFFER_ITERS  = 3        # iterations of self-play data to keep - a
                                # ~4.2k-game window at PCR game rates, ~3 gradient
                                # passes per example over its lifetime
TARGET_EXAMPLES_ITER = 120_000  # adaptive games/iter targets this example count;
                                # ~1250 games/iter for the diversity the value
                                # head needs.  ~25% of rows carry policy targets
POOL_SIZE            = _env_int("T7G_POOL_SIZE", 512)
                                # concurrent self-play games; each half of the
                                # double-buffered pool is one inference batch, so
                                # this IS the self-play batch-size knob.  Powers
                                # of two only (ROCm pads batches to pow2).  Tune
                                # per machine via T7G_POOL_SIZE (512≈14 GB slabs).

# Model parameters
NET_ARCH             = "net2"   # "net2" = t7g-net2 (lib/net2.py, KataGo-family);
                                # "old" = legacy DualHeadNetwork
SOFT_POLICY_COEF     = 0.0      # net2 aux soft policy head: OFF - ablation
                                # showed KataGo's nominal 8.0 bought zero policy
                                # CE (the attention head owns the win) at ~0.01
                                # holdout value CE.  No-op for the old arch.
ST_VALUE_COEF        = 0.25     # net2 short-term value heads: MSE toward
                                # lambda-averaged future root Q (horizons
                                # ~6/16/40 plies) - the search-derived value
                                # signal, distinct from the --blend-alpha path.
VALUE_BLEND_ALPHA    = 0.25     # α=1 → pure terminal target (blending off).
                                # α<1 mixes gated root-Q into the value target
                                # AT THE LOSS as soft WDL class probabilities
                                # (1-w)*onehot(z) + w*[(1+q)/2, 0, (1-q)/2] -
                                # pre-blending the scalar would be quantized
                                # away by the hard ±0.33 class thresholds.
                                # α=0.25 (max Q weight 0.75) is where the
                                # measured target SNR peaks while split-half
                                # fidelity to the true outcome is still rising;
                                # it collapses at α=0 (pure self-distillation).
VALUE_COEF           = 1.0      # value-loss weight; bumped to rebalance against
                                # policy CE's 1225-class gradient magnitude
MARGIN_COEF          = 0.4      # auxiliary margin-head loss weight (final
                                # material margin / 49) - dense KataGo-style
                                # signal that trains the trunk from every game
WDL_VALUE            = True     # 3-way W/D/L cross-entropy value head - avoids
                                # the tanh+MSE saturation pathology (confidently
                                # wrong positions with ~zero gradient to unlearn)
OWNERSHIP_AUX        = True     # per-cell final-ownership aux head (KataGo);
OWNERSHIP_COEF       = 0.15     # 49 dense spatial targets/position teach the
                                # trunk to count material.  Aux TRAINING signal
                                # only - never enters search utility or move
                                # selection.
GAMES_MIN            = 50
GAMES_MAX            = 2000     # playout-cap randomization targets ~1250
                                # games/iter (120k examples / ~96 per game)
LEARNING_RATE        = 1.0e-4
WEIGHT_DECAY         = 1e-4
C_PUCT               = 1.3
GUMBEL_K             = 16
SIGMA_SCALE          = 1.0      # multiplier on the Gumbel sigma(q) transform;
                                # <1 makes completed-Q targets stickier to the prior
COMPLETION_N0        = 50.0     # visit-shrinkage prior strength in the completed-Q
                                # target: q~(a)=(n_a*q_a+n0*v_root)/(n_a+n0).  Caps
                                # low-visit Q noise before sigma amplifies it into
                                # target logits.  The temp-0 played move is the SH
                                # winner.
SELF_PLAY_TEMP_MOVES = 16       # sigma-scaled Gumbel targets need less temperature
                                # forcing than a longer temp window would add
ENTROPY_COEFF        = 0.00

# Playout-cap randomization (KataGo): per MOVE, with prob PCR_P_FULL run the
# full --simulations budget and keep the policy target; otherwise run a cheap
# PCR_FAST_SIMS search whose example trains value/margin/ownership only
# (policy target zeroed -> masked out of the policy CE; root-Q blend weight
# dropped too - a 100-sim root is too noisy to teach).  Decouples the two data
# appetites: policy needs deep searches, value needs MANY GAMES.  At p=0.25
# a game costs ~0.4x the sims, so the example budget above buys ~2.5x the
# distinct games/outcomes per iteration.  PCR_P_FULL = 1.0 disables.
PCR_P_FULL           = 0.25
PCR_FAST_SIMS        = 100

# Eval
EVAL_INTERVAL        = 5
EVAL_GAMES           = 30       # 10 games/rung made the ladder oscillate 0->100%
EVAL_WORKERS         = _env_int("T7G_EVAL_WORKERS", 4)
CHECKPOINT_INTERVAL  = 10
CHECKPOINT_DIR       = "models/mcts"   # override per run with --checkpoint-dir
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
ELO_GAMES_PER_OPPONENT = 16     # an 8-game gate is too noisy - it can both miss
                                # a real ~100-Elo climb and promote a regressed
                                # net on one lucky 8-0-0 block
GATE_MARGIN_ELO        = 25     # candidate must clear the incumbent by this
                                # (as a score margin) head-to-head.  The gate is
                                # RELATIVE - candidate vs best_network in the same
                                # games - because an absolute bar (best-elo point
                                # estimate + margin) gets set on upward promotion
                                # noise and then stalls the ratchet as headroom
                                # shrinks
GATE_BLOCK_GAMES       = 16     # h2h games per block (colour-balanced)
GATE_MAX_GAMES         = 96     # adaptive cap: clear results stop at one block;
                                # only genuine close calls extend (96 games
                                # resolves ~+/-35 Elo; promotion needs ~57% there)
GATE_MM7_FLOOR         = 12     # catastrophe guard: never promote a net whose
                                # gauntlet MM7 wins drop below this (out of
                                # ELO_GAMES_PER_OPPONENT), whatever the h2h says
ELO_ROTATE_AFTER       = 6      # evals without promotion before the current net
                                # is injected as a rolling anchor anyway (no
                                # generator change).  Keeps the pool near current
                                # strength so ratings don't saturation-cap against
                                # long-outgrown anchors (the mechanism that hid
                                # run_fastblend's climb)
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


# ============================================================
# Self-play data generation
# ============================================================

def generate_self_play_data(
    mcts: MCGS,
    min_non_truncated: int = 50,
    mcts_pool: 'list | None' = None,
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
                                    blend_alpha=VALUE_BLEND_ALPHA,
                                    pcr_p_full=PCR_P_FULL,
                                    pcr_fast_sims=PCR_FAST_SIMS)
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

    all_examples = [(obs, p, v, m, o, rq, qw, st)
                    for obs, p, v, m, o, _, _, rq, qw, st in raw_examples]

    avg_branching = float(np.mean(all_legal_counts)) if all_legal_counts else 0.0
    return (all_examples, (blue_wins, green_wins, draws),
            game_moves, game_times, truncations, avg_branching)


# ============================================================
# Main training loop
# ============================================================

def main():
    global VALUE_BLEND_ALPHA, POOL_SIZE, PCR_FAST_SIMS, TARGET_EXAMPLES_ITER
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser(description="AlphaZero MCTS Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS,
                        help="MCTS simulations per move")
    parser.add_argument("--pool", type=int, default=POOL_SIZE,
                        help=f"concurrent self-play games / inference batch knob "
                             f"(default: {POOL_SIZE}, from T7G_POOL_SIZE if set)")
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
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help=f"where iter_*.pt / promoted_*.pt / final.pt go "
                             f"(default: {CHECKPOINT_DIR}); set this per run or "
                             f"a new run overwrites the previous run's history")
    parser.add_argument("--blend-alpha", type=float, default=VALUE_BLEND_ALPHA,
                        help=f"Value-target blend: 1.0 = pure game outcome, "
                             f"<1 mixes gated root-Q into soft WDL targets, "
                             f"max Q weight = 1-alpha (default: {VALUE_BLEND_ALPHA})")
    parser.add_argument("--pcr-fast-sims", type=int, default=PCR_FAST_SIMS,
                        help=f"PCR fast-move sim budget: the cheap search that "
                             f"plays most moves and trains value/margin/ownership "
                             f"only (default: {PCR_FAST_SIMS})")
    parser.add_argument("--target-examples", type=int, default=TARGET_EXAMPLES_ITER,
                        help=f"adaptive games/iter targets this many examples; "
                             f"buffer = this x REPLAY_BUFFER_ITERS "
                             f"(default: {TARGET_EXAMPLES_ITER})")
    parser.add_argument("--arch", choices=["net2", "old"], default=NET_ARCH,
                        help=f"network architecture (default: {NET_ARCH})")
    parser.add_argument("--cudagraphs", action="store_true",
                        help="compile inference nets with mode='reduce-overhead' "
                             "(CUDA/hip graphs) and force pow2 batch padding. "
                             "Cuts per-forward Python dispatch ~9x (1.4ms -> "
                             "0.16ms measured on the 3060 Ti, 2026-07-19); "
                             "needs a go/no-go validation on ROCm/gfx1151 "
                             "before use there.")
    args = parser.parse_args()

    VALUE_BLEND_ALPHA = args.blend_alpha
    CHECKPOINT_DIR    = args.checkpoint_dir
    POOL_SIZE = args.pool
    PCR_FAST_SIMS = args.pcr_fast_sims
    TARGET_EXAMPLES_ITER = args.target_examples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    num_actions = 1225
    if args.arch == "net2":
        def _make_net():
            return Net2(num_actions=num_actions)
    else:
        def _make_net():
            return DualHeadNetwork(num_actions=num_actions,
                                   wdl=WDL_VALUE, ownership=OWNERSHIP_AUX)
    _compile_kwargs = {"mode": "reduce-overhead"} if args.cudagraphs else {}
    if args.cudagraphs:
        import lib.mcgs as _mcgs_mod
        _mcgs_mod.PAD_BATCH_POW2 = True  # bound the CUDA-graph shape set

    network   = _make_net().to(device)
    inference_network = (  # type: ignore[assignment]
        torch.compile(network, **_compile_kwargs)
        if device.type == "cuda" else network
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
        try:
            network.load_state_dict(_ckpt_sd)
            _same_arch = True
        except RuntimeError:
            _same_arch = False
        if _same_arch:
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
    best_network = _make_net().to(device)
    best_network.load_state_dict(network.state_dict())
    best_network.eval()
    best_inference_network = (  # type: ignore[assignment]
        torch.compile(best_network, **_compile_kwargs)
        if device.type == "cuda" else best_network
    )
    best_elo: 'float | None' = None   # telemetry: pool elo at the last promotion
    incumbent_name: 'str | None' = None  # the incumbent's pool entry (pinned:
                                      # never evicted by rolling trims) - its 16
                                      # gauntlet games double as the gate's free
                                      # first block, so a clear gate costs zero
                                      # fresh games
    retained_streak = 0               # consecutive "retained" evals; triggers a
                                      # scheduled anchor rotation at ELO_ROTATE_AFTER

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
            "Elo (95% CI)": ["Margin", ["eval/elo", "eval/elo_lo", "eval/elo_hi"]],
            "Best Elo": ["Multiline", ["eval/best_elo"]],
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
    print(f"Pool size:        {POOL_SIZE}"
          + ("  (T7G_POOL_SIZE)" if os.environ.get("T7G_POOL_SIZE") else ""))
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Epochs/iteration: {EPOCHS_PER_ITERATION}")
    print(f"Value blend:      alpha={VALUE_BLEND_ALPHA}"
          + ("  (pure terminal)" if VALUE_BLEND_ALPHA >= 1.0
             else f"  (max Q weight {1.0 - VALUE_BLEND_ALPHA:.2f}, gated)"))
    print(f"Eval ladder:      {' > '.join(lbl for _, _, lbl in EVAL_LADDER)} > retire")
    print("=" * 60)

    _mcts_kwargs = dict(num_simulations=args.simulations, c_puct=C_PUCT, gumbel_k=GUMBEL_K,
                        sigma_scale=SIGMA_SCALE, completion_n0=COMPLETION_N0)
    _sp_kwargs   = _mcts_kwargs
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
        term_width = os.get_terminal_size().columns
    except OSError:  # no tty (piped/headless run)
        term_width = 80

    if device.type == "cuda":
        print("Warming up torch.compile ...", end=" ", flush=True)
        network.eval()
        with torch.no_grad():
            _w = torch.zeros(1, 7, 7, 4, device=device)
            inference_network(_w)       # type: ignore[operator]
            best_inference_network(_w)  # type: ignore[operator]
        network.train()
        print("done")

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

        examples, (bw, gw, dr), game_moves, game_times, truncations, avg_branching = (
            generate_self_play_data(
                self_play_mcts,
                min_non_truncated=games_this_iter,
                mcts_pool=mcts_pool,
                temp_moves=SELF_PLAY_TEMP_MOVES,
            )
        )

        # Playout-cap split: PCR fast rows carry a zeroed policy target.
        pcr_fast_rows = (sum(1 for e in examples if not np.any(e[1]))
                         if PCR_P_FULL < 1.0 else 0)
        pcr_full_rows = len(examples) - pcr_fast_rows

        network.train()
        gen_time = time.time() - gen_start
        moves_arr = np.array(game_moves)
        avg_moves = float(moves_arr.mean())
        med_moves = float(np.median(moves_arr))
        std_moves = float(moves_arr.std())
        avg_gtime = gen_time / len(game_times)
        trunc_pct = 100.0 * truncations / len(game_moves)
        # Bill each move class at its own rate -- a single-rate formula
        # inflated run_fastblend's sim/s ~35% (109k logged, ~78k true).  With
        # PCR, example rows are billed exactly by their recorded cap; the few
        # non-example moves (truncated games, spurious-zero recoveries) at the
        # expected mixed cost.
        _avg_move_sims = (PCR_P_FULL * args.simulations
                          + (1.0 - PCR_P_FULL) * PCR_FAST_SIMS)
        _non_example_moves = max(0, int(moves_arr.sum())
                                 - pcr_full_rows - pcr_fast_rows)
        total_sims = int(pcr_full_rows * args.simulations
                         + pcr_fast_rows * PCR_FAST_SIMS
                         + _non_example_moves * _avg_move_sims)
        sims_per_sec = total_sims / gen_time if gen_time > 0 else 0.0

        print(" " * term_width + "\r", end='')  # Clear the tqdm bar
        pcr_tag = (f"  pol-rows {pcr_full_rows} ({pcr_full_rows / max(1, len(examples)):.0%})"
                   if PCR_P_FULL < 1.0 else "")
        print(f"  Self-play  {len(examples):>6} ex  {gen_time:.0f}s  {sims_per_sec:.0f} sim/s"
              f"  B:{bw} G:{gw} D:{dr}{pcr_tag}")
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
        writer.add_scalar("self_play/policy_rows",          pcr_full_rows,         step)
        writer.add_scalar("self_play/avg_game_moves",        avg_moves,             step)
        writer.add_histogram("self_play/game_moves_dist",   moves_arr,             step)
        writer.add_scalar("self_play/truncation_pct",       trunc_pct,             step)
        writer.add_scalar("self_play/avg_branching_factor", avg_branching,         step)

        entropies = []
        jump_masses = []
        for _, policy_target, *_ in examples:
            nz = policy_target[policy_target > 0]
            if nz.size > 0:
                entropies.append(float(-np.sum(nz * np.log(nz))))
                # Fraction of search visits placed on jump (vs clone) moves --
                # the strategy fingerprint; a shift here is play-style drift even
                # while the net keeps sweeping the minimax ladder.
                jump_masses.append(float(policy_target[JUMP_MASK].sum()))
        avg_policy_entropy = float(np.mean(entropies)) if entropies else 0.0
        avg_jump_frac = float(np.mean(jump_masses)) if jump_masses else 0.0
        print(f"  Policy     entropy {avg_policy_entropy:.3f}  jump {avg_jump_frac:.1%}"
              f"  epg {_epg:.1f} (ema {_epg_ema:.1f})  -> {games_this_iter} next")
        writer.add_scalar("self_play/avg_policy_entropy", avg_policy_entropy, step)
        writer.add_scalar("self_play/jump_move_frac",     avg_jump_frac,      step)
        if entropies:
            writer.add_histogram("self_play/policy_entropy_dist", np.array(entropies), step)

        writer.add_scalar("timing/gen_seconds",      gen_time,     step)
        writer.add_scalar("timing/avg_game_seconds", avg_gtime,    step)
        writer.add_scalar("timing/sims_per_sec",     sims_per_sec, step)

        is_eval_iter = (step % EVAL_INTERVAL == 0)

        replay_buffer.append_batch(examples)
        writer.add_scalar("self_play/buffer_size", len(replay_buffer), step)

        # Per-iteration uniqueness only.  The buffer-level variant was a rolling
        # window of this same signal (it just lags by REPLAY_BUFFER_ITERS) and
        # cost an O(buffer) rehash every iteration for no extra information.
        _uniq_iter     = len({e[0].tobytes() for e in examples})
        _uniq_iter_pct = _uniq_iter / max(1, len(examples))
        print(f"  Buffer     {len(replay_buffer):>6} ex  uniq {_uniq_iter_pct:.0%} iter")
        writer.add_scalar("self_play/unique_positions_iter", _uniq_iter,     step)
        writer.add_scalar("self_play/unique_pct_iter",       _uniq_iter_pct, step)

        #  Train 
        train_start = time.time()
        losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITERATION, device=device,
            entropy_coeff=ENTROPY_COEFF,
            value_coef=VALUE_COEF,
            margin_coef=MARGIN_COEF,
            ownership_coef=OWNERSHIP_COEF,
            soft_policy_coef=SOFT_POLICY_COEF,
            st_value_coef=ST_VALUE_COEF,
        )
        train_time = time.time() - train_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train      pol {losses['policy_loss']:.4f}  val {losses['value_loss']:.4f}"
              f"  marg {losses['margin_loss']:.4f}  own {losses['ownership_loss']:.4f}"
              f"  soft {losses['soft_policy_loss']:.4f}  st {losses['st_value_loss']:.4f}"
              f"  tot {losses['total_loss']:.4f}"
              f"  sign {losses['sign_acc']:.1%}  draw {losses['draw_frac']:.1%}"
              f"  vdec {losses['value_ce_decisive']:.4f}"
              f"  vdrw {losses['value_ce_draw']:.4f}  {train_time:.0f}s")

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("train/policy_loss",    losses['policy_loss'], step)
        writer.add_scalar("train/value_loss",     losses['value_loss'],  step)
        writer.add_scalar("train/margin_loss",    losses['margin_loss'], step)
        writer.add_scalar("train/ownership_loss", losses['ownership_loss'], step)
        writer.add_scalar("train/soft_policy_loss", losses['soft_policy_loss'], step)
        writer.add_scalar("train/st_value_loss",  losses['st_value_loss'], step)
        writer.add_scalar("train/total_loss",     losses['total_loss'],  step)
        writer.add_scalar("train/value_sign_acc", losses['sign_acc'],    step)
        writer.add_scalar("train/draw_frac",      losses['draw_frac'],   step)
        writer.add_scalar("train/value_ce_decisive", losses['value_ce_decisive'], step)
        writer.add_scalar("train/value_ce_draw",  losses['value_ce_draw'], step)
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
                      f"  t:{res_cur['n_terminal']} c:{res_cur['n_clock']}"
                      f" x:{res_cur['n_truncated']}")
                writer.add_scalar("eval/ladder_progress",   eval_level + wr_cur,      step)

            # Elo vs the fixed anchor pool - the primary progress metric once
            # the MM ladder saturates (every net past ~1100 beats MM5 ~100%).
            if elo_pool:
                elo, elo_ci95, elo_res, elo_shape = rate_vs_pool(
                    network, elo_pool,
                    games_per_opponent=ELO_GAMES_PER_OPPONENT,
                    num_actions=num_actions,
                    mcts_kwargs=dict(_mcts_kwargs, num_simulations=EVAL_SIMULATIONS),
                    num_workers=EVAL_WORKERS,
                )
                writer.add_scalar("eval/elo",    elo,               step)
                writer.add_scalar("eval/elo_lo", elo - elo_ci95,    step)
                writer.add_scalar("eval/elo_hi", elo + elo_ci95,    step)
                detail = "  ".join(f"{n}:{w}-{d}-{ls}"
                                   for n, (w, d, ls) in elo_res.items())
                print(f"  Elo        {elo:.0f} +/- {elo_ci95:.0f}  ({detail})")
                # How the rating was earned: win margin (pieces) + game length
                # separate a dominant sweep from a marginal one at the same Elo.
                print(f"  Gauntlet   win+{elo_shape['win_margin_med']:.0f} /"
                      f" loss{elo_shape['loss_margin_med']:.0f} pcs (med)"
                      f"  len {elo_shape['moves_med']:.0f}"
                      f"  W/D/L {elo_shape['n_win']}/{elo_shape['n_draw']}/{elo_shape['n_loss']}")
                writer.add_scalar("eval/gauntlet_win_margin",  elo_shape['win_margin_med'],     step)
                writer.add_scalar("eval/gauntlet_loss_margin", elo_shape['loss_margin_med'],    step)
                writer.add_scalar("eval/gauntlet_moves_med",   elo_shape['moves_med'],          step)
                writer.add_scalar("eval/gauntlet_draw_margin", elo_shape['draw_margin_absmed'], step)

                # Ratchet: promote the training net to data generator only when
                # it beats the incumbent head-to-head, measured fresh in the
                # same games (relative gate - immune to the winner's-curse /
                # stale-bar stall of the old absolute-Elo gate) and extended
                # adaptively when the record is too close to call.
                if best_elo is None:
                    best_elo = elo          # telemetry baseline only
                s_margin = 1.0 / (1.0 + 10.0 ** (-GATE_MARGIN_ELO / 400.0))
                # The pinned incumbent's gauntlet games seed the gate for free.
                seed = (tuple(elo_res[incumbent_name])
                        if incumbent_name in elo_res else None)
                gate, (gw, gd, gl), gscore = h2h_gate(
                    network, best_network, s_margin=s_margin, seed_record=seed,
                    block_games=GATE_BLOCK_GAMES, max_games=GATE_MAX_GAMES,
                    num_actions=num_actions,
                    mcts_kwargs=dict(_mcts_kwargs,
                                     num_simulations=EVAL_SIMULATIONS),
                    num_workers=EVAL_WORKERS,
                )
                mm7_rec = elo_res.get("MM7")
                mm7_ok = mm7_rec is None or mm7_rec[0] >= GATE_MM7_FLOOR
                gate_n = gw + gd + gl
                fresh_n = gate_n - (sum(seed) if seed else 0)
                print(f"  Gate       vs-best {gw}-{gd}-{gl} ({gate_n} games,"
                      f" {fresh_n} fresh)  s={gscore:.3f}"
                      + ("" if mm7_ok else
                         f"  MM7 FLOOR VETO ({mm7_rec[0]}/{ELO_GAMES_PER_OPPONENT}"
                         f" < {GATE_MM7_FLOOR})"))
                writer.add_scalar("eval/gate_score", gscore, step)
                writer.add_scalar("eval/gate_games", gate_n, step)
                if gate == "promote" and mm7_ok:
                    best_network.load_state_dict(network.state_dict())
                    best_network.eval()
                    best_elo = elo
                    retained_streak = 0
                    # Self-anchor: the just-promoted net joins the pool as a rolling
                    # opponent so future ratings are measured against current-strength
                    # play (never saturates).  Keep only the newest ELO_ROLLING_WINDOW.
                    incumbent_name = f"self_iter{step:04d}"
                    elo_pool.append({
                        "name": incumbent_name, "kind": "net",
                        "payload": {k: v.detach().cpu()
                                    for k, v in network.state_dict().items()},
                        "elo": elo, "fixed": False})
                    _rolling = [m for m in elo_pool if not m["fixed"]]
                    _evictable = [m for m in _rolling
                                  if m["name"] != incumbent_name]
                    _excess = max(0, len(_rolling) - rolling_window)
                    for _old in _evictable[:_excess]:
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
                    retained_streak += 1
                    print("  Ratchet    retained")
                    if retained_streak >= ELO_ROTATE_AFTER:
                        # Scheduled anchor rotation: the pool must track current
                        # strength even when nothing promotes, or ratings
                        # saturation-cap against outgrown anchors and the gate
                        # goes blind (run_fastblend hid a ~100-Elo climb this
                        # way).  The training net joins as a ROLLING ANCHOR
                        # ONLY: the generator (best_network) is untouched.
                        retained_streak = 0
                        elo_pool.append({
                            "name": f"anchor_iter{step:04d}", "kind": "net",
                            "payload": {k: v.detach().cpu()
                                        for k, v in network.state_dict().items()},
                            "elo": elo, "fixed": False})
                        _rolling = [m for m in elo_pool if not m["fixed"]]
                        _evictable = [m for m in _rolling
                                      if m["name"] != incumbent_name]
                        _excess = max(0, len(_rolling) - rolling_window)
                        for _old in _evictable[:_excess]:
                            elo_pool.remove(_old)
                        print(f"  Ratchet    anchor rotation ({ELO_ROTATE_AFTER} evals "
                              f"without promotion) - pool: "
                              f"{', '.join(m['name'] for m in elo_pool)}")
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
        gpu_stats = get_gpu_stats()
        if "util_pct" in gpu_stats:
            writer.add_scalar("system/gpu_util_pct", gpu_stats["util_pct"], step)
        if "temp_c" in gpu_stats:
            writer.add_scalar("system/gpu_temp_c", gpu_stats["temp_c"], step)

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
