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

from lib.device_utils import get_gpu_stats, build_inference_network  # noqa: E402
from lib.net2 import Net2                                            # noqa: E402
from lib.evaluation import (                                         # noqa: E402
    evaluate_vs_noisy_minimax, rate_vs_pool, _calibrate_ladder,
)
from lib.mcgs import MCGS                                            # noqa: E402
from lib.t7g import action_to_move                                   # noqa: E402

# Boolean mask over the 1225-action space: True where the move is a jump
# (source vacated) rather than a clone (source retained).  Ataxx strategy
# hinges on preferring clones; the search's jump-mass is its style fingerprint.
JUMP_MASK = np.array([action_to_move(a)[4] for a in range(1225)], dtype=bool)
from lib.train_workers import self_play_game_pool, take_spurious_zero_count  # noqa: E402
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
                                # "net2c" = t7g-net2c (lib/net2c.py) - margin,
                                # soft-policy and short-term-value heads removed,
                                # ownership rebranched off the trunk.  net2c is
                                # NOT weight-compatible with net2: fresh run only.
SOFT_POLICY_COEF     = 0.0      # net2 aux soft policy head: OFF - ablation
                                # showed KataGo's nominal 8.0 bought zero policy
                                # CE (the attention head owns the win) at ~0.01
                                # holdout value CE.
ST_VALUE_COEF        = 0.0      # OFF.  The short-term value heads existed to
                                # sharpen LATE-game value, but irreducible var(z)
                                # is already 0.279 at ply 60-80 and 0.000 at 80+ -
                                # z teaches the endgame exactly; the opening
                                # (0.917) is the unlearnable part, and that is the
                                # main target's job.  They were near-redundant
                                # anyway: lambda-returns over horizons 2..200 have
                                # participation ratio 1.26 (one direction = 88%),
                                # and st_value_fc read the same 96-dim vector as
                                # value_wdl.  net2c has no such head at all; this
                                # only still applies to net2.
VALUE_TARGET_LAMBDA  = 0.9375   # main value target = lambda-return over FUTURE
                                # root Q, not the 1-step root Q.  None = pure
                                # game-outcome target z.
                                # Why: z is one label per GAME (ICC 1.000 ->
                                # n_eff 4000 in a 392k-row set); a target that
                                # varies within a game breaks that clustering.
                                # Measured on stored data, lambda=0.9375 gives
                                # n_eff 6769 AND split-half fidelity to the true
                                # outcome 0.265 vs z's 0.198.
                                # 0.9375 = ST_HORIZONS' old middle horizon (16
                                # plies); the fidelity optimum is 0.9-0.95.
VALUE_Q_WEIGHT_START = 0.1      # q_weight on ITERATION 1, ramped linearly to
                                # VALUE_Q_WEIGHT over VALUE_Q_WEIGHT_RAMP_ITERS.
                                # A fresh net's root Q is worthless, so the
                                # lambda-return degenerates to a damped z at
                                # init (measured on the net2c smoke: |vt| mean
                                # 0.178 vs ~0.26 on trained-net data).  Holding
                                # q_weight at 0.9 from iteration 1 would put 90%
                                # of the value target on an untrained net's
                                # opinion of itself, and would barely teach
                                # draws at all (q_dist has no draw mass).  The
                                # lambda-return measurement that justified 0.9
                                # was made on a TRAINED net's Q, so the ramp is
                                # what makes the premise hold.
VALUE_Q_WEIGHT_RAMP_ITERS = 10  # short on purpose.  The bar is not "the net is
                                # good", only "root Q beats a damped z as a value
                                # target", which a warm-up of ~10 iters clears -
                                # search does most of the work at 500 sims even
                                # behind a weak prior.  A longer ramp would just
                                # spend iterations training on the z target the
                                # lambda-return is meant to replace.
VALUE_Q_WEIGHT       = 0.9      # FINAL weight on that target when
                                # VALUE_TARGET_LAMBDA is set.  Mixed at the
                                # LOSS as soft WDL class probabilities
                                # (1-w)*onehot(z) + w*[(1+q)/2, 0, (1-q)/2] -
                                # pre-blending the scalar would be quantized
                                # away by the hard ±0.33 class thresholds.
                                # The residual 0.1 stays on the true outcome:
                                # it anchors against bootstrap drift and is
                                # the ONLY teacher of draw mass.
VALUE_COEF           = 1.0      # value-loss weight; bumped to rebalance against
                                # policy CE's 1225-class gradient magnitude
MARGIN_COEF          = 0.0      # OFF.  debug/aux_noise_floor.py measured the
                                # margin target as carrying NO learnable signal
                                # beyond z (beyond-z variance ~0): it restates the
                                # outcome.  At 0.4 it was the second-heaviest term
                                # in the loss.  net2c removes the head outright;
                                # this zeroes it for net2 too.
OWNERSHIP_COEF       = 2.0      # Aux TRAINING signal only - never enters search
                                # utility or move selection.  Raised from 0.15:
                                # ownership carries ~4x z's learnable signal, 75%
                                # of it independent of the outcome, over ~13
                                # effective spatial directions - yet the loss is a
                                # MEAN over 49 cells, which silently divided its
                                # aggregate by 49.  Parity anchor: value has 0.141
                                # learnable nats/position vs ownership's 0.0635
                                # per cell, so 0.141/0.0635 ~= 2.2 weights the two
                                # heads' LEARNABLE content 1:1.  2.0 is the
                                # conservative side of that.
                                # TODO sweep {0.5, 2.0, 8.0} on net2c once a run
                                # exists - judge on holdout POLICY CE and value
                                # sign-acc, never on ownership CE (which improves
                                # monotonically with the coef and proves nothing).
                                # Watch that policy CE does not degrade: raising
                                # this dilutes policy's share of the gradient, and
                                # policy is the biggest Elo contributor (~894).
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

# Playout-cap randomization (KataGo): per MOVE, with prob PCR_P_FULL run the
# full --simulations budget and keep the policy target; otherwise run a cheap
# PCR_FAST_SIMS search whose example trains value/margin/ownership only
# (policy target zeroed -> masked out of the policy CE).  Decouples the two data
# appetites: policy needs deep searches, value needs MANY GAMES.  At p=0.25
# a game costs ~0.4x the sims, so the example budget above buys ~2.5x the
# distinct games/outcomes per iteration.  PCR_P_FULL = 1.0 disables.
PCR_P_FULL           = 0.25
PCR_FAST_SIMS        = 100

# Eval
EVAL_INTERVAL        = 5
EVAL_GAMES           = 30       # 10 games/rung made the ladder oscillate 0->100%
# Concurrent games in the eval tournament pool (lib/train_workers.tournament_pool).
# Batches inference across eval games the way self-play already does; more slots
# = bigger batch but each holds an MCGS arena for a whole game, so this trades
# memory for throughput.
EVAL_POOL            = _env_int("T7G_EVAL_POOL", 32)
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
# Fallback pool, used when ELO_POOL_PATH is absent - a fresh checkout has no
# models/ directory, so the only anchors that can be assumed present are the
# engines, which are code (lib/micro3) and need no checkpoint.  MM5's Elo is on
# the canonical Stauf=1000 gauge (derived by the deterministic engine-vs-engine
# ladder in debug/eval_db/RESULTS.md, same provenance as the MM7=1392 pin), so
# a pool-less checkout still reads Elo on the same scale as pool.json - just
# with a single, easily-swept anchor and correspondingly wide error bars.
DEFAULT_ELO_POOL: list = [
    {"name": "MM5", "kind": "mm", "depth": 5, "elo": 1237, "fixed": True},
]
ELO_GAMES_PER_OPPONENT = 16     # 8 games/opponent is too noisy to read a real
                                # ~100-Elo move off; this is a measurement now,
                                # it gates nothing
ELO_ROLLING_WINDOW     = 2      # self-anchored rating: at each eval the current
                                # net is appended to the pool as an opponent, and
                                # only the newest N such self-anchors are kept.  A
                                # fixed set of anchors always saturates once the net
                                # sweeps them; measuring against recent selves keeps
                                # an opponent near current strength so the Elo (and
                                # promotion bar) never ceilings.  The fixed engine +
                                # seed-net anchors (pool.json "fixed": true) stay to
                                # pin the absolute scale / flag regressions.  Can be
                                # overridden by "rolling_window" in pool.json.


def _value_q_weight(step: int) -> float:
    """q_weight for this iteration: linear ramp START -> VALUE_Q_WEIGHT.

    Keyed off the global iteration counter, so a resumed run picks the ramp up
    where it left off rather than restarting it.
    """
    if VALUE_TARGET_LAMBDA is None:
        return VALUE_Q_WEIGHT
    if VALUE_Q_WEIGHT_RAMP_ITERS <= 0:
        return VALUE_Q_WEIGHT
    frac = min(1.0, max(0.0, (step - 1) / float(VALUE_Q_WEIGHT_RAMP_ITERS)))
    return VALUE_Q_WEIGHT_START + frac * (VALUE_Q_WEIGHT - VALUE_Q_WEIGHT_START)


# ============================================================
# Self-play data generation
# ============================================================

def generate_self_play_data(
    mcts: MCGS,
    num_games: int = 50,
    mcts_pool: 'list | None' = None,
    temp_moves: int = 0,
    value_q_weight: float = VALUE_Q_WEIGHT,
) -> tuple:
    """
    Generate training examples in-process using the main network directly.

    Plays num_games games and keeps every one of them.  Self-play has no move
    cap, so every game reaches a genuine rules outcome (board terminal or
    halfmove-clock draw) and contributes its examples; nothing is discarded.

    Returns
    -------
    (examples, (blue_wins, green_wins, draws), game_moves, game_times,
     avg_branching)
    """
    all_examples: list = []
    blue_wins = green_wins = draws = games_done = 0
    game_moves: list = []
    game_times: list = []
    all_legal_counts: list = []
    raw_examples: list = []

    pbar = tqdm(desc="Self-play", unit="game", total=num_games)
    try:
        for game_examples, winner, moves, gtime, legal_counts in (
            self_play_game_pool(mcts, POOL_SIZE, num_games, mcts_pool,
                                temp_moves=temp_moves,
                                value_lambda=VALUE_TARGET_LAMBDA,
                                value_q_weight=value_q_weight,
                                pcr_p_full=PCR_P_FULL,
                                pcr_fast_sims=PCR_FAST_SIMS)
        ):
            all_legal_counts.extend(legal_counts)
            game_moves.append(moves)
            game_times.append(gtime)
            raw_examples.extend(game_examples)
            games_done += 1
            pbar.update(1)
            if winner > 0:
                blue_wins += 1
            elif winner < 0:
                green_wins += 1
            else:
                draws += 1
            if games_done >= num_games:
                break
        pbar.set_postfix(raw=len(raw_examples))
    finally:
        pbar.close()

    all_examples = [(obs, p, v, m, o, rq, qw, st)
                    for obs, p, v, m, o, _, _, rq, qw, st in raw_examples]

    avg_branching = float(np.mean(all_legal_counts)) if all_legal_counts else 0.0
    return (all_examples, (blue_wins, green_wins, draws),
            game_moves, game_times, avg_branching)


# ============================================================
# Main training loop
# ============================================================

def main():
    global POOL_SIZE, PCR_FAST_SIMS, TARGET_EXAMPLES_ITER
    global CHECKPOINT_DIR, VALUE_Q_WEIGHT, VALUE_Q_WEIGHT_RAMP_ITERS, WEIGHT_DECAY
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
    parser.add_argument("--pcr-fast-sims", type=int, default=PCR_FAST_SIMS,
                        help=f"PCR fast-move sim budget: the cheap search that "
                             f"plays most moves and trains value/margin/ownership "
                             f"only (default: {PCR_FAST_SIMS})")
    parser.add_argument("--target-examples", type=int, default=TARGET_EXAMPLES_ITER,
                        help=f"adaptive games/iter targets this many examples; "
                             f"buffer = this x REPLAY_BUFFER_ITERS "
                             f"(default: {TARGET_EXAMPLES_ITER})")
    # Policy-target TEMPERATURE knobs.  sigma_mult = (SIGMA_C_VISIT + max_a N(a))
    # * sigma_scale scales with VISIT COUNTS, so raising --simulations silently
    # sharpens the stored policy target -- the bug that killed run_deepsearch
    # (target entropy 0.348 -> 0.238; the net fit the "better" targets WORSE).
    # Raising the sim budget therefore REQUIRES pinning both of these, so that
    # teacher DEPTH is the only variable that changes:
    #   sims 2000 -> --sigma-scale 0.27   ((50+2000)*0.27 ~= (50+500)*1.0)
    #               --completion-n0 200   (4x, so n0 shrinks Q as much as at 500)
    # Value-target mix.  q_weight is the weight on the bootstrapped lambda-return
    # vs the true outcome z; it ramps START -> VALUE_Q_WEIGHT over --q-ramp-iters
    # counted from the LOCAL iteration number, so a warm start RESTARTS the ramp.
    # NOTE the two optimizers apply this COMPLETELY differently.  Adam adds the
    # L2 term to the gradient, so it is divided by sqrt(v) and bites hardest in
    # low-gradient directions (measured: uniform 37% trunk-norm loss over 120
    # iters, ~0.4%/iter).  AdamW decays by lr*wd directly, so at lr=1e-4 a wd of
    # 1e-4 is ~1e-8/step -- effectively OFF.  To match Adam's observed decay rate
    # under AdamW you need wd ~ 0.03.  Do not port a value across optimizers.
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help=f"weight decay (default: {WEIGHT_DECAY}); read the "
                             f"comment above -- NOT comparable between adam/adamw")
    parser.add_argument("--value-q-weight", type=float, default=VALUE_Q_WEIGHT,
                        help=f"final weight on the lambda-return value target; "
                             f"1-this is the anchor on the true outcome "
                             f"(default: {VALUE_Q_WEIGHT})")
    parser.add_argument("--q-ramp-iters", type=int, default=VALUE_Q_WEIGHT_RAMP_ITERS,
                        help=f"iterations to ramp q_weight from "
                             f"{VALUE_Q_WEIGHT_START} to --value-q-weight; 0 = "
                             f"start at the final value (default: "
                             f"{VALUE_Q_WEIGHT_RAMP_ITERS})")
    parser.add_argument("--sigma-scale", type=float, default=SIGMA_SCALE,
                        help=f"Gumbel sigma(q) multiplier. MUST be scaled down "
                             f"when --simulations goes up, to hold policy-target "
                             f"temperature fixed (default: {SIGMA_SCALE})")
    parser.add_argument("--completion-n0", type=float, default=COMPLETION_N0,
                        help=f"completed-Q visit-shrinkage prior. Scale WITH the "
                             f"sim budget (default: {COMPLETION_N0})")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam",
                        help="adamw = DECOUPLED weight decay: plain Adam adds the "
                             "L2 term to the gradient, so it is amplified by 1/sqrt(v) "
                             "and bites hardest in low-gradient directions -- measured "
                             "2026-07-26 as a uniform 37%% trunk-norm loss over 120 iters")
    parser.add_argument("--arch", choices=["net2", "net2c"], default=NET_ARCH,
                        help=f"network architecture (default: {NET_ARCH})")
    parser.add_argument("--cudagraphs", action="store_true",
                        help="compile inference nets with mode='reduce-overhead' "
                             "(CUDA/hip graphs) and force pow2 batch padding. "
                             "Cuts per-forward Python dispatch ~9x (1.4ms -> "
                             "0.16ms measured on the 3060 Ti, 2026-07-19); "
                             "needs a go/no-go validation on ROCm/gfx1151 "
                             "before use there.")
    args = parser.parse_args()

    CHECKPOINT_DIR    = args.checkpoint_dir
    POOL_SIZE = args.pool
    PCR_FAST_SIMS = args.pcr_fast_sims
    TARGET_EXAMPLES_ITER = args.target_examples
    WEIGHT_DECAY = args.weight_decay
    VALUE_Q_WEIGHT = args.value_q_weight
    VALUE_Q_WEIGHT_RAMP_ITERS = args.q_ramp_iters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    num_actions = 1225
    if args.arch == "net2c":
        from lib.net2c import Net2C
        def _make_net():
            return Net2C(num_actions=num_actions)
    else:
        def _make_net():
            return Net2(num_actions=num_actions)
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
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr,
                                      weight_decay=WEIGHT_DECAY)
        print(f"Optimizer:        adamw lr={args.lr:.1e} wd={WEIGHT_DECAY:.1e} "
              f"(decoupled)")
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr,
                                     weight_decay=WEIGHT_DECAY)

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
                # load_state_dict restores param_groups WHOLESALE from the
                # checkpoint, so every hyperparameter silently reverts to
                # whatever the run that wrote the file used.  That voided the
                # 2026-07-27 overnight arm 3: --weight-decay 3e-2 was
                # discarded in favour of the checkpoint's 1e-4, and the arm
                # ran an exact duplicate of arm 2.  Re-assert everything the
                # CLI can set, not just the LR.
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr
                    pg['weight_decay'] = WEIGHT_DECAY
                print(f"  Restored optimizer state (LR overridden to "
                      f"{args.lr}, wd to {WEIGHT_DECAY:.1e})")
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

    # The self-play generator.  Refreshed from the training net at the END of
    # every iteration, so self-play always runs on the latest weights.
    #
    # This used to be an AlphaGo-Zero ratchet: the generator only advanced when
    # the training net won a head-to-head gate.  Removed 2026-07-25 - at
    # run_net2c's strength the 16-game gate was mostly noise (scores decayed
    # 1.000 -> 0.688 over four evals while the net was still climbing), so it
    # amounted to requiring luck at the top end rather than enforcing progress.
    # The historical argument for a ratchet was run 1 losing ~270 Elo over iters
    # 50-180 with ungated self-play; if that reappears it will show up as a
    # sustained fall in the pool Elo, which is now pure telemetry.  Watch for it.
    best_network = _make_net().to(device)
    best_network.load_state_dict(network.state_dict())
    best_network.eval()
    best_inference_network = (  # type: ignore[assignment]
        torch.compile(best_network, **_compile_kwargs)
        if device.type == "cuda" else best_network
    )
    best_elo: 'float | None' = None   # telemetry: pool elo at the last eval
    incumbent_name: 'str | None' = None  # newest self-anchor's pool entry, pinned
                                      # so a rolling trim never evicts it

    # Elo pool: a fixed part (engine + seed-net anchors that pin the absolute
    # scale) plus a rolling part of recent promoted selves appended below, so the
    # net always has an opponent near its own strength and the rating can't
    # saturate.  The current net's Elo is solved against the whole pool each eval.
    # Members carry "fixed": True (never evicted) or False (rolling self-anchor).
    # Every net anchor is optional: a checkout without our checkpoints (or a
    # pool file pointing at a net this code can no longer build) drops that
    # member and rates against whatever is left, rather than failing the run.
    elo_pool: list = []
    rolling_window = ELO_ROLLING_WINDOW
    if os.path.exists(ELO_POOL_PATH):
        with open(ELO_POOL_PATH) as _f:
            _pool_cfg = json.load(_f)
        rolling_window = _pool_cfg.get("rolling_window", ELO_ROLLING_WINDOW)
        _members = _pool_cfg["members"]
    else:
        print(f"Elo pool: {ELO_POOL_PATH} not found; using the built-in engine anchors")
        _members = DEFAULT_ELO_POOL
    for _m in _members:
        if _m["kind"] == "net":
            try:
                _blob = torch.load(_m["path"], map_location="cpu", weights_only=True)
                _payload = _blob["network"] if "network" in _blob else _blob
                # Buildable-for-play check now, not at the first eval hours in.
                # Anchors are only ever played, so this accepts legacy
                # dual-head nets that training itself cannot use.
                build_inference_network(_payload)   # discard the module
            except (OSError, KeyError, RuntimeError, ValueError) as _e:
                print(f"  skipping Elo anchor {_m['name']}: {type(_e).__name__}: {_e}")
                continue
        else:
            _payload = _m["depth"]
        elo_pool.append({"name": _m["name"], "kind": _m["kind"],
                         "payload": _payload, "elo": _m["elo"],
                         "fixed": _m.get("fixed", True)})
    if elo_pool:
        print(f"Elo pool: {', '.join(m['name'] for m in elo_pool)} "
              f"(rolling window {rolling_window} self-anchors)")
    else:
        print("Elo pool: no usable anchors; eval/elo disabled")

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
    if VALUE_TARGET_LAMBDA is not None:
        print(f"Value target:     lambda-return l={VALUE_TARGET_LAMBDA} "
              f"@ q_weight {VALUE_Q_WEIGHT_START} -> {VALUE_Q_WEIGHT} "
              f"over {VALUE_Q_WEIGHT_RAMP_ITERS} iters "
              f"({1.0 - VALUE_Q_WEIGHT:.2f} final anchor on the true outcome)")
    else:
        print("Value target:     pure game outcome z")
    print(f"Eval ladder:      {' > '.join(lbl for _, _, lbl in EVAL_LADDER)} > retire")
    print("=" * 60)

    _mcts_kwargs = dict(num_simulations=args.simulations, c_puct=C_PUCT, gumbel_k=GUMBEL_K,
                        sigma_scale=args.sigma_scale, completion_n0=args.completion_n0)
    _sp_kwargs   = _mcts_kwargs
    # EVAL is pinned to the rated search config (sigma 1.0 / n0 50), NOT the
    # self-play one.  The eval-DB scale (config_hash da7e1ee0) and the fixed Elo
    # anchors were both fitted at those values, so letting a deep-teacher
    # --sigma-scale leak into eval would silently move every rating off-scale.
    _eval_mcts_kwargs = dict(_mcts_kwargs, sigma_scale=SIGMA_SCALE,
                             completion_n0=COMPLETION_N0)
    # Self-play runs on best_network, which now tracks the training net one
    # iteration behind (refreshed after each train step).  The search targets are
    # still one improvement step ahead of the generator that produced them.
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
            network, _mcts_kwargs,
            eval_ladder=EVAL_LADDER, eval_simulations=EVAL_SIMULATIONS,
            pool_size=EVAL_POOL,
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

        examples, (bw, gw, dr), game_moves, game_times, avg_branching = (
            generate_self_play_data(
                self_play_mcts,
                num_games=games_this_iter,
                mcts_pool=mcts_pool,
                temp_moves=SELF_PLAY_TEMP_MOVES,
                value_q_weight=_value_q_weight(step),
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
        # Bill each move class at its own rate -- a single-rate formula
        # inflated run_fastblend's sim/s ~35% (109k logged, ~78k true).  With
        # PCR, example rows are billed exactly by their recorded cap; the few
        # non-example moves (spurious-zero recoveries) at the expected mixed
        # cost.
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
        # Games past the retired 200-move cap: these used to be discarded, so
        # this is the share of the data the cap removal recovered.
        long_pct = 100.0 * float((moves_arr > 200).mean())
        # Slab-overflow recoveries.  Expected to be 0; anything else means the
        # search is returning empty policies and those positions were dropped.
        n_spurious = take_spurious_zero_count()
        spurious_tag = f"  SLAB-OVERFLOW {n_spurious}" if n_spurious else ""
        print(f"  Games      avg {avg_moves:.1f}  med {med_moves:.1f}  std {std_moves:.1f}"
              f"  [{int(moves_arr.min())}-{int(moves_arr.max())}]"
              f"  >200 {long_pct:.1f}%  branch {avg_branching:.1f}{spurious_tag}")

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
        writer.add_scalar("self_play/long_game_pct",        long_pct,              step)
        writer.add_scalar("self_play/slab_overflow_skips",  n_spurious,            step)
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
        # Target audit.  run_deepsearch burned 35 iterations on a silently
        # corrupted target; these two numbers make that visible on iteration 1.
        # With the lambda-return live, q_w should pin to VALUE_Q_WEIGHT and
        # |vt| should sit well below 1 (a lambda-return is graded, unlike z).
        _vt = np.array([e[5] for e in examples], dtype=np.float32)
        _qw = np.array([e[6] for e in examples], dtype=np.float32)
        print(f"  Target     q_w mean {_qw.mean():.3f} (min {_qw.min():.3f} "
              f"max {_qw.max():.3f})  |vt| mean {np.abs(_vt).mean():.3f}  "
              f"var {_vt.var():.4f}  |vt|>0.33 {100*np.mean(np.abs(_vt) > 0.33):.0f}%")
        writer.add_scalar("train/target_qw_mean", float(_qw.mean()), step)
        writer.add_scalar("train/target_vt_var", float(_vt.var()), step)
        writer.add_scalar("self_play/unique_positions_iter", _uniq_iter,     step)
        writer.add_scalar("self_play/unique_pct_iter",       _uniq_iter_pct, step)

        #  Train 
        train_start = time.time()
        losses = train_network(
            network, replay_buffer, optimizer,
            batch_size=BATCH_SIZE, epochs=EPOCHS_PER_ITERATION, device=device,
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
        if _marg_preds is not None:      # net2c has no margin head
            writer.add_histogram("train/margin_output_dist", _marg_preds.squeeze().cpu(), step)

        #  Generator refresh
        # Unconditional, every iteration: next iteration's self-play runs on the
        # weights we just trained.  No gate, no floor - see the best_network
        # comment above for why the ratchet was removed.  load_state_dict copies
        # into best_network's existing tensors, so best_inference_network (the
        # torch.compile wrapper holding a reference to it) picks the new weights
        # up without recompiling.
        best_network.load_state_dict(network.state_dict())
        best_network.eval()

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
                    mcts_kwargs=_eval_mcts_kwargs, engine='micro3',
                    pool_size=EVAL_POOL,
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
                    mcts_kwargs=dict(_eval_mcts_kwargs, num_simulations=EVAL_SIMULATIONS),
                    pool_size=EVAL_POOL,
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

                # Self-anchor.  The generator is refreshed every iteration
                # (see "Generator refresh" below), so there is no promotion
                # decision to make here - this only keeps the rating pool near
                # current strength: the net joins as a rolling opponent, and a
                # fixed anchor set would ceiling the Elo the moment it is swept.
                # Keep the newest ELO_ROLLING_WINDOW selves.
                best_elo = elo
                incumbent_name = f"self_iter{step:04d}"
                elo_pool.append({
                    "name": incumbent_name, "kind": "net",
                    "payload": {k: v.detach().cpu()
                                for k, v in network.state_dict().items()},
                    "elo": elo, "fixed": False})
                _rolling = [m for m in elo_pool if not m["fixed"]]
                _evictable = [m for m in _rolling if m["name"] != incumbent_name]
                _excess = max(0, len(_rolling) - rolling_window)
                for _old in _evictable[:_excess]:
                    elo_pool.remove(_old)
                # Persist this net so it can be rated offline (scripts/eval_db.py):
                # the in-memory self-anchor is ephemeral and every-N checkpoints
                # miss the eval steps that have a measured Elo attached.
                promoted_path = os.path.join(CHECKPOINT_DIR, f"promoted_iter{step:04d}.pt")
                torch.save({'iteration': iteration, 'network': network.state_dict(),
                            'elo': float(elo)}, promoted_path)
                print(f"  Anchor     {incumbent_name} @ {elo:.0f} joins pool "
                      f"({', '.join(m['name'] for m in elo_pool)})"
                      f" -> {promoted_path}")
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
                            mcts_kwargs=_eval_mcts_kwargs, engine='micro3',
                            pool_size=EVAL_POOL,
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
