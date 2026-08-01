"""
Game-logic helpers for AlphaZero self-play training.

Stateless (no module-level globals) to allow multiprocessing.
"""
import os
import time

import numpy as np

from lib.mcgs import MCGS, MCGSSearch, step_searches
from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move,
    tick_clock, CLOCK_LIMIT,
)
from lib.training import ST_LAMBDAS

# External UAI engines (lib.uai_engine) -- imported lazily where used, since
# they require a subprocess binary under 3rd_party/ rather than a ctypes DLL.
UAI_ENGINES = {"autaxx", "autaxx-ab", "tiktaxx", "scarlettxx"}


# ---------------------------------------------------------------------------
# Value-target blending (Option A + B)
# ---------------------------------------------------------------------------
#
# See docs/value_blending.md for the full rationale and the Option D swap-in
# recipe.  Short version: the final value target is
#
#     value_target = α * terminal + (1 - α) * root_q
#
# The blend is applied in the LOSS (lib/training.py), not here: workers store
# the pure terminal outcome plus (root_q, q_weight) per example, and the WDL
# head trains on soft class targets  (1-w)*onehot(z) + w*[(1+q)/2, 0, (1-q)/2].
# Pre-blending into one scalar would be quantized away by the hard ±0.33
# class conversion (2026-07-12 audit).  The per-example weight w suppresses
# Q influence in regimes where Q is known to be unreliable:
#
#   (A) Noise ramp: Q weight follows how UNPREDICTABLE the terminal outcome
#       is at that ply.  Q earns its place by replacing a noisy label, so it
#       is worth most in the opening and worth nothing once z is already
#       exact.  Profile below is measured, not guessed.
#
#   (B) Visit concentration gate: if the root visit distribution is flat
#       (MCTS uncertain), down-weight Q further.  Concentration is
#       1 - normalised_entropy; a peaked distribution → 1, uniform → 0.
#
# The two gates combine multiplicatively: both must fire for Q to reach
# full weight.
#
# (A) weights Q by where the terminal label z is actually noisy: irreducible
# var(z) is ~0.92 in the opening (ply 0-20) and ~0.00 past ply 80, so the search
# Q is most worth blending in early and least worth it late.  Temperature
# affects which move is PLAYED, not the root's value backup, so it must not gate
# the blend.  See memory/project_blend_gate_inverted.md + debug/target_noise_floor.py.
#
# Caution: un-gated heavy blending can drive the value head toward 0 everywhere.
# ~0 IS correct in the opening on a 95%-noise label; the failure mode to watch
# is value variance going flat across ALL plies, not just early ones.
# ---------------------------------------------------------------------------

# Irreducible var(z) by ply, from duplicate-position groups (2026-07-22,
# debug/target_noise_floor.py), normalised by its own max to a weight in [0,1].
# np.interp clamps outside the knots: full weight before ply 10, zero past 90.
_NOISE_PLY = (10.0, 30.0, 50.0, 70.0, 90.0)
_NOISE_VAR = (0.92, 0.85, 0.68, 0.28, 0.00)


def _q_blend_weight(
    move_idx: int,
    policy_target: np.ndarray,
    blend_alpha: float,
) -> float:
    """
    Gated Q weight for one example's value target.

    When blend_alpha == 1.0 returns 0.0 (blending off; no-op fast path).
    Otherwise applies the noise × concentration gating described above.

    Parameters
    ----------
    move_idx       : move number when the example was recorded
    policy_target  : MCTS visit-weighted policy at the root (used for concentration)
    blend_alpha    : maximum α used for pure terminal (1 - blend_alpha = max Q weight)

    Returns
    -------
    q_weight in [0, 1 - blend_alpha]
    """
    if blend_alpha >= 1.0:
        return 0.0

    # (A) Noise ramp: how much of z is irreducible at this ply, normalised.
    noise = float(np.interp(move_idx, _NOISE_PLY, _NOISE_VAR)) / _NOISE_VAR[0]

    # (B) Concentration of MCTS visit distribution: 1.0 = one-hot, 0.0 = uniform.
    support = policy_target[policy_target > 1e-8]
    if support.size <= 1:
        concentration = 1.0 if support.size == 1 else 0.0
    else:
        entropy     = float(-np.sum(support * np.log(support)))
        max_entropy = float(np.log(support.size))
        concentration = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

    return (1.0 - blend_alpha) * noise * concentration


# ---------------------------------------------------------------------------
# In-process game pool - batched inference across concurrent games
# ---------------------------------------------------------------------------

# Self-play has no move cap: the halfmove clock terminates every game.  Only a
# clone resets the clock (tick_clock), only a clone adds a piece, and the board
# holds 49 from a start of 4 -- so a game admits at most 45 clock resets and must
# end within ~46 * CLOCK_LIMIT plies.  This valve is an order of magnitude above
# that bound and exists solely to stop a rules bug from hanging a training run;
# it is not a data-shaping parameter.  (Rating paths keep their own 200-move cap
# -- see _t_apply -- because the eval DB's ratings were measured under it.)
MOVE_SAFETY_VALVE = 50_000

# Count of positions where the search returned an all-zero policy despite legal
# moves existing -- the slab-overflow signature (see the recovery in
# _advance_group).  Recovery is silent by design, so without this counter the
# only symptom is a sub-0.1 dip in examples-per-game, which rounds away in the
# logs.  Read and reset once per iteration via take_spurious_zero_count().
_SPURIOUS_ZERO_COUNT = 0


def take_spurious_zero_count() -> int:
    """Return the spurious-all-zero (slab overflow) count and reset it."""
    global _SPURIOUS_ZERO_COUNT
    n, _SPURIOUS_ZERO_COUNT = _SPURIOUS_ZERO_COUNT, 0
    return n


class _GameSlot:
    """State for one concurrent game inside self_play_game_pool."""

    __slots__ = [
        'board', 'turn', 'examples', 'move_count',
        'clock', 'legal_move_counts', 'game_start',
        'search', 'mcts', 'full_move',
    ]

    def __init__(self, mcts: MCGS) -> None:
        self.mcts = mcts
        self.board = new_board()
        self.turn = bool(np.random.randint(2))
        self.examples: list = []
        self.move_count = 0
        self.clock = 0     # halfmove clock: plies since the last clone move
        self.legal_move_counts: list = []
        self.game_start = time.time()
        self.search: MCGSSearch | None = None
        self.full_move = True   # whether the in-flight search runs the full cap


def _slot_result(
    slot: _GameSlot,
    winner: float,
    blend_alpha: float = 1.0,
    value_lambda: float | None = None,
    value_q_weight: float | None = None,
) -> tuple:
    """Package a finished slot into a result tuple matching self_play_game_pool's yield contract.

    Returns
    -------
    training_examples : list of (obs, raw_policy, value_target, margin,
                                 ownership, board, turn, root_q, q_weight,
                                 st_targets)
        10-tuples - board and turn included so the caller can apply policy
        relabeling outside the GPU-critical pool loop.  margin is the final
        material margin / 49 from the example's side-to-move perspective.
        ownership is a (7,7) int8 map of the *final* board from the example's
        side-to-move perspective: 0=mine, 1=opponent's, 2=empty.
        value_target is the PURE terminal outcome; root_q (side-to-move
        perspective) and its gated blend weight ride along separately so the
        loss can mix them at the class-distribution level (see module header).
        st_targets is a (len(ST_LAMBDAS),) float32 array of lambda-averaged
        future MCTS root values (side-to-move perspective) for the t7g-net2
        short-term value heads:  s_i = (1-l)*sum_{j in [i,n)} l^(j-i)*q_j
        + l^(n-i)*z  - i.e. the terminal outcome is the value tail beyond
        game end, so s -> z as l -> 1 and late positions approach z.  Fast
        (PCR) moves contribute their shallow root Q: individually noisy but
        damped by the average.
    winner            : +1.0 Blue / −1.0 Green / 0.0 draw.  Always a genuine
        rules outcome -- self-play has no move cap, so there is no
        material-ratio approximation here (see MOVE_SAFETY_VALVE).
    move_count        : number of half-moves played
    elapsed           : wall time in seconds
    legal_move_counts : per-position branching factor samples
    """
    blue, green = count_cells(slot.board)
    margin_blue = float(blue - green) / 49.0

    # Final-ownership class maps (board plane 1 = Blue, plane 0 = Green).
    # Computed once per game; each example gets the map oriented to its own
    # side-to-move so it matches board_to_obs channel semantics.
    own_final = np.full((7, 7), 2, dtype=np.int8)          # 2 = empty
    own_as_blue = own_final.copy()
    own_as_blue[slot.board[:, :, 1]] = 0                   # Blue's cells
    own_as_blue[slot.board[:, :, 0]] = 1                   # Green's cells
    own_as_green = own_final.copy()
    own_as_green[slot.board[:, :, 0]] = 0
    own_as_green[slot.board[:, :, 1]] = 1

    # Short-term value targets: backward recursion in Blue perspective
    # (perspective flips between examples are just sign flips there), one
    # column per lambda.  acc starts at the terminal outcome = the value of
    # every ply beyond game end.
    # One backward pass covers the aux short-term horizons AND (when enabled)
    # the main value target's lambda -- it is the same recursion, so the main
    # target rides along as an extra column rather than a second sweep.
    n_ex = len(slot.examples)
    n_st = len(ST_LAMBDAS)
    all_lams = list(ST_LAMBDAS) + ([value_lambda] if value_lambda is not None else [])
    lambdas = np.asarray(all_lams, dtype=np.float32)
    st_blue = np.empty((n_ex, len(all_lams)), dtype=np.float32)
    acc = np.full(len(all_lams), winner, dtype=np.float32)
    for j in range(n_ex - 1, -1, -1):
        _, _, j_turn, _, j_q, _, _ = slot.examples[j]
        q_blue = j_q if j_turn else -j_q
        acc = (1.0 - lambdas) * q_blue + lambdas * acc
        st_blue[j] = acc
    vt_blue = st_blue[:, n_st] if value_lambda is not None else None
    st_blue = st_blue[:, :n_st]

    examples = []
    for i, (obs, policy_target, example_turn, ex_board, _root_q, move_idx, full_move) in \
            enumerate(slot.examples):
        value_target = winner if example_turn else -winner
        st_targets = st_blue[i] if example_turn else -st_blue[i]
        if value_lambda is not None:
            # Main value target = lambda-return over future root Q.  It rides
            # the root_q slot because lib/training.py already routes that slot
            # through the SOFT W/D/L distribution [(1+v)/2, 0, (1-v)/2]; the z
            # slot is thresholded at +-0.33, which would quantize ~half of a
            # lambda-return away.  The slot therefore means "search-derived
            # value target here", not "root Q at this node".
            # q_weight is a constant, NOT the noise-ramp gate: the ramp exists
            # to down-weight Q where z is trustworthy, but the lambda-return is
            # a better estimate of the outcome than z at every ply (measured:
            # split-half fidelity 0.257-0.269 vs z's 0.198).  The residual
            # 1 - value_q_weight stays on the true outcome, which anchors
            # against bootstrap drift and is the only teacher of DRAW mass
            # (q_dist assigns none).  PCR fast rows keep their value target --
            # only the policy target is zeroed for them.
            root_q = float(vt_blue[i] if example_turn else -vt_blue[i])
            q_weight = value_q_weight
            if not full_move:
                policy_target = np.zeros_like(policy_target)
        elif full_move:
            root_q = _root_q
            q_weight = _q_blend_weight(
                move_idx=move_idx, policy_target=policy_target,
                blend_alpha=blend_alpha,
            )
        else:
            # Playout-cap-randomized fast move: the shallow search is good
            # enough to play but not to teach - zero the policy target (masked
            # out of the policy loss) and drop its root Q from the value blend.
            # z / margin / ownership still train from these rows.
            policy_target = np.zeros_like(policy_target)
            root_q = 0.0
            q_weight = 0.0
        margin = margin_blue if example_turn else -margin_blue
        ownership = own_as_blue if example_turn else own_as_green
        examples.append((obs, policy_target, value_target, margin, ownership,
                         ex_board, example_turn, root_q, q_weight, st_targets))
    elapsed = time.time() - slot.game_start
    return examples, winner, slot.move_count, elapsed, slot.legal_move_counts


def _start_slot_search(
    slot: _GameSlot,
    full_sims: int,
    pcr_p_full: float,
    pcr_fast_sims: int,
    move_count: 'int | None' = None,
) -> None:
    """Start the next move's search on `slot`, rolling its playout cap.

    Playout-cap randomization (KataGo): with probability pcr_p_full the move
    gets the full budget and yields a policy training target; otherwise it runs
    a cheap pcr_fast_sims search whose example is value/aux-only (policy target
    zeroed and Q-blend weight dropped in _slot_result).  pcr_p_full >= 1.0
    disables the mechanism entirely (no C calls, byte-identical behaviour).
    """
    full = pcr_p_full >= 1.0 or np.random.random() < pcr_p_full
    slot.full_move = full
    if pcr_p_full < 1.0:
        slot.mcts.set_num_simulations(full_sims if full else pcr_fast_sims)
    slot.search = slot.mcts.start_search(slot.board, slot.turn,
                                         move_count=move_count, clock=slot.clock)


def _reset_slot(slot: _GameSlot) -> None:
    """Reset a finished slot's game state so it can play a new game."""
    slot.mcts.clear()
    slot.board = new_board()
    slot.turn = bool(np.random.randint(2))
    slot.examples = []
    slot.move_count = 0
    slot.clock = 0
    slot.legal_move_counts = []
    slot.game_start = time.time()
    slot.search = None


def _advance_group(
    active: list,
    target_games: int,
    games_started: int,
    temp_moves: int,
    blend_alpha: float,
    full_sims: int = 0,
    pcr_p_full: float = 1.0,
    pcr_fast_sims: int = 100,
    value_lambda: float | None = None,
    value_q_weight: float | None = None,
) -> tuple[list, int, list]:
    """
    Step each slot's search once, handle completed searches (apply MCTS move,
    check termination, restart finished games), and return:
        (next_active, new_games_started, results)

    Results is a list of `_slot_result` tuples for games that ended during
    this call; the caller is responsible for yielding them downstream.

    Mirrors the per-slot post-forward loop in the original single-pool driver
    exactly - split out so both halves of the double-buffered pool can share
    it.
    """
    next_active: list = []
    results: list = []

    # One C call steps every slot's search and reports which finished.
    done_flags = step_searches([slot.search for slot in active])

    for slot, search_done in zip(active, done_flags):
        if not search_done:
            next_active.append(slot)
            continue

        action_probs = slot.search.result
        root_q = slot.search.root_value
        best_action = slot.search.best_action
        skip_example = False
        if not np.any(action_probs):
            is_terminal, terminal_value = check_terminal(slot.board, slot.turn)
            if is_terminal:
                assert terminal_value is not None
                winner = terminal_value if slot.turn else -terminal_value
                results.append(_slot_result(slot, winner, blend_alpha,
                                            value_lambda, value_q_weight))
                if games_started < target_games:
                    _reset_slot(slot)
                    _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims)
                    next_active.append(slot)
                    games_started += 1
                continue

            # Distinguish genuine forced-pass from spurious all-zero (slab overflow).
            if np.any(action_masks(slot.board, slot.turn)):
                # Spurious all-zero: recover with uniform over legal moves;
                # skip adding this position as a training example.
                global _SPURIOUS_ZERO_COUNT
                _SPURIOUS_ZERO_COUNT += 1
                masks = action_masks(slot.board, slot.turn)
                action_probs = masks.astype(np.float32)
                action_probs /= action_probs.sum()
                root_q = 0.0
                skip_example = True
                # Fall through to normal action-selection below.
            else:
                # Genuine forced pass.
                slot.turn = not slot.turn
                slot.move_count += 1
                slot.clock += 1
                if slot.clock >= CLOCK_LIMIT:
                    results.append(_slot_result(slot, 0.0, blend_alpha,
                                                value_lambda, value_q_weight))
                    if games_started < target_games:
                        _reset_slot(slot)
                        _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims)
                        next_active.append(slot)
                        games_started += 1
                else:
                    _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims)
                    next_active.append(slot)
                continue

        if not skip_example:
            obs = board_to_obs(slot.board, slot.turn,
                               slot.clock if slot.mcts.clock_obs else 0)
            slot.examples.append(
                (obs, action_probs, slot.turn, slot.board.copy(), root_q,
                 slot.move_count, slot.full_move)
            )

        temp = 1.0 if slot.move_count < temp_moves else 0.0
        action = slot.mcts.select_action(
            action_probs, board=slot.board, turn=slot.turn, temperature=temp,
            best_action=best_action,
        )
        slot.board = apply_move(slot.board, action, slot.turn)
        slot.turn = not slot.turn
        slot.move_count += 1
        slot.clock = tick_clock(slot.clock, action)

        is_terminal, terminal_value = check_terminal(slot.board, slot.turn)

        done = False
        winner = 0.0
        if is_terminal:
            assert terminal_value is not None
            winner = terminal_value if slot.turn else -terminal_value
            done = True
        elif slot.clock >= CLOCK_LIMIT:
            winner = 0.0  # halfmove clock expired = draw (libataxx rule)
            done = True
        elif slot.move_count > MOVE_SAFETY_VALVE:
            # Unreachable under the rules (see MOVE_SAFETY_VALVE); a hit means a
            # clock/rules bug, so score it a draw rather than spin forever.
            winner = 0.0
            done = True

        if done:
            results.append(_slot_result(slot, winner, blend_alpha,
                                            value_lambda, value_q_weight))
            if games_started < target_games:
                _reset_slot(slot)
                _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims)
                next_active.append(slot)
                games_started += 1
        else:
            masks = action_masks(slot.board, slot.turn)
            if not np.any(masks):
                slot.turn = not slot.turn
            else:
                slot.legal_move_counts.append(int(masks.sum()))
            _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims,
                               move_count=slot.move_count)
            next_active.append(slot)

    return next_active, games_started, results


def self_play_game_pool(
    mcts: MCGS,
    pool_size: int,
    target_games: int,
    mcts_pool: 'list[MCGS] | None' = None,
    temp_moves: int = 0,
    blend_alpha: float = 1.0,
    pcr_p_full: float = 1.0,
    pcr_fast_sims: int = 100,
    value_lambda: float | None = None,
    value_q_weight: float | None = None,
):
    """
    Play target_games games concurrently with batched network inference.

    Each slot has its own MCGS instance (isolated transposition table).  The
    pool is split into two halves (A and B) that alternate GPU dispatches:
    while the GPU is doing A's forward pass, the Python thread runs the
    CPU-side step/advance work on B (and vice versa).  This overlaps CPU
    and GPU instead of running them sequentially per batch.

    Slots are immediately restarted when a game finishes, keeping all
    pool_size slots active throughout (no draining at the tail).

    mcts_pool: optional list of pre-created MCGS instances to reuse across
    calls (avoids recreating them each iteration).  Must have len >= pool_size.

    pcr_p_full / pcr_fast_sims: playout-cap randomization (see
    _start_slot_search).  The full budget is the driver `mcts` instance's
    num_simulations (pooled instances may carry a mutated value from the
    previous call's last move, so the driver's is the authoritative one).

    Yields result tuples as each game completes:
        (training_examples, winner, move_count, elapsed, legal_move_counts)
    """
    full_sims = mcts.num_simulations
    if mcts_pool is not None:
        slots = [_GameSlot(m) for m in mcts_pool[:pool_size]]
    else:
        slots = [
            _GameSlot(MCGS(
                mcts.network,
                num_simulations=full_sims,
                c_puct=mcts.c_puct,
                gumbel_k=mcts.gumbel_k,
            ))
            for _ in range(pool_size)
        ]
    for slot in slots:
        slot.mcts.clear()  # ensure no stale TT from a previous pool run
        _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims)

    # Split into two halves for double-buffered launch/collect pipelining.
    half = pool_size // 2
    active_a = slots[:half]
    active_b = slots[half:]
    games_started = pool_size

    # Prime both groups: dispatch an initial forward for each so the loop
    # can start in a steady "one in-flight per group" state.
    handle_a = mcts._launch_forward([s.search for s in active_a]) if active_a else None
    handle_b = mcts._launch_forward([s.search for s in active_b]) if active_b else None

    while active_a or active_b:
        # --- Group A: collect its in-flight forward, step CPU work, relaunch.
        # GPU is busy with B's forward during the CPU work here.
        mcts._collect_and_commit(handle_a)
        active_a, games_started, results = _advance_group(
            active_a, target_games, games_started, temp_moves, blend_alpha,
            full_sims, pcr_p_full, pcr_fast_sims,
            value_lambda, value_q_weight,
        )
        for r in results:
            yield r
        handle_a = (mcts._launch_forward([s.search for s in active_a])
                    if active_a else None)

        # --- Group B: same thing, with GPU now busy on A's next forward.
        mcts._collect_and_commit(handle_b)
        active_b, games_started, results = _advance_group(
            active_b, target_games, games_started, temp_moves, blend_alpha,
            full_sims, pcr_p_full, pcr_fast_sims,
            value_lambda, value_q_weight,
        )
        for r in results:
            yield r
        handle_b = (mcts._launch_forward([s.search for s in active_b])
                    if active_b else None)


# ---------------------------------------------------------------------------
# In-process tournament pool - batched inference across concurrent eval games
# ---------------------------------------------------------------------------
#
# The single-game path below (play_eval_game / play_net_vs_net_game) drives one
# search at a time: measured 147 forward passes per 500-sim move with a mean
# batch of 3.35 pending leaves, i.e. ~2.5k sim/s against self-play's 80k+.  Eval
# was ~38% of run wall-clock at that rate.  This pool plays MANY eval games
# concurrently and batches their pending leaves into one forward per network per
# step, exactly like self_play_game_pool - same double-buffered launch/collect,
# so CPU work on one half overlaps the GPU pass on the other.
#
# Search semantics per game are UNCHANGED (same sims, same start_search /
# step / commit mechanism the single-game path already uses through
# _expand_batch); only the batch grows.  Anchor Elos therefore stay valid.
#
# Engine (minimax/Stauf) moves run SERIALLY in the driver thread on purpose:
# the C solvers in src/bb_core.h share a module-global transposition table and
# history-heuristic array, so find_best_move is not reentrant and must never be
# called from a worker thread.  Serial also keeps the TT access pattern the same
# as the per-process single-game path.  The cost hides behind the other half's
# in-flight forward pass.
#
# Agents are:
#     ("net", key)                                 - MCGS player on nets[key]
#     ("engine", depth, noise, engine, vary_depth) - CPU player
# Games are (tag, blue_agent, green_agent, first_turn); `tag` is opaque to the
# pool and comes back on the result so callers can attribute the game.

class _TourneySlot:
    """State for one concurrent game inside tournament_pool."""

    __slots__ = [
        'tag', 'blue', 'green', 'board', 'turn', 'clock', 'move_count',
        'stauf_moves', 'search', 'mover', 'mcts', 'result', 'end_reason',
    ]

    def __init__(self) -> None:
        self.mcts: dict = {}      # net key -> MCGS instance (reused per slot)
        self.tag = None
        self.blue: tuple = ()
        self.green: tuple = ()
        self.board = new_board()
        self.turn = True
        self.clock = 0
        self.move_count = 0
        self.stauf_moves = 0
        self.search: MCGSSearch | None = None
        self.mover = None         # net key whose search is in flight
        self.result = 0.0         # Blue-perspective result
        self.end_reason = "terminal"


def _t_settle(slot: _TourneySlot) -> bool:
    """Advance `slot` past terminal checks and forced passes.

    Returns True if the game ended (slot.result / slot.end_reason set), False if
    the side to move has a legal move and must now choose one.  Mirrors the top
    of play_eval_game's and play_net_vs_net_game's while-loop exactly: terminal
    first, then the halfmove clock, then a pass - on which the turn flips and
    the clock ticks but move_count does NOT advance.
    """
    while True:
        is_terminal, terminal_value = check_terminal(slot.board, slot.turn)
        if is_terminal:
            assert terminal_value is not None
            slot.result = terminal_value if slot.turn else -terminal_value
            slot.end_reason = "terminal"
            return True
        if slot.clock >= CLOCK_LIMIT:
            slot.result = 0.0     # halfmove clock expired = draw (libataxx rule)
            slot.end_reason = "clock"
            return True
        if not np.any(action_masks(slot.board, slot.turn)):
            slot.turn = not slot.turn
            slot.clock += 1
            continue
        return False


def _t_apply(slot: _TourneySlot, action: int) -> bool:
    """Play `action` on `slot`; return True if the 200-move cap ended the game."""
    slot.board = apply_move(slot.board, action, slot.turn)
    slot.turn = not slot.turn
    slot.move_count += 1
    slot.clock = tick_clock(slot.clock, action)
    if slot.move_count > 200:
        blue, green = count_cells(slot.board)
        slot.result = (float(blue - green) / float(blue + green)
                       if blue + green > 0 else 0.0)
        slot.end_reason = "truncated"
        return True
    return False


def _t_engine_move(slot: _TourneySlot, agent: tuple) -> bool:
    """Play one engine move on `slot`; return True if the game ended.

    Faithful to play_eval_game's opponent branch, including drawing the noise
    coin unconditionally and treating a -1/1225 return as a pass.  Single-slot
    path; the pool uses _t_engine_moves to batch these across slots.
    """
    action = _t_engine_moves([(slot, agent)], 0)[0]
    if action is None:
        slot.turn = not slot.turn
        slot.clock += 1
        return False
    return _t_apply(slot, action)


# A minimax move is the most expensive single thing in an eval and it is pure
# CPU: measured 117 ms/move for micro3 at depth 7 in the midgame (218 ms average
# over a whole game), so the 16 MM7 games in one Elo phase are minutes of work
# that would otherwise stall the pool one move at a time.
#
# It cannot simply be threaded: bb_core.h keeps the transposition table and
# history heuristic in module-global state, so one loaded copy of a solver is
# not reentrant.  dlopen'ing distinct COPIES of the .so gives each thread its
# own globals, and ctypes releases the GIL around the call - verified to return
# bit-identical moves to the shared-library path, at ~2.3x on 8 threads (the
# per-copy 2^20-entry TT limits scaling; it is not linear).
#
# Depth <= 4 costs a few ms, below the hand-off, so those stay inline.
_ENGINE_MIN_DEPTH_THREADED = 5
# Absolute throughput on one board kept improving with fleet size on an 8-core
# desktop (74 -> 50 -> 42 ms/move at 4/8/16 threads), so size from the machine.
# Eval is the only thing running at eval time, so taking the cores is free.
_ENGINE_MAX_THREADS = min(os.cpu_count() or 4, 16)

# engine name -> (.so basename, symbol).  Only the (board, depth, as_blue) ->
# int solvers can join a fleet; Stauf takes an extra move index and the UAI
# engines are subprocess-backed, so both stay on the inline path.
_FLEET_ENGINES = {
    "micro3":  ("micro3", "find_best_move"),
    "micro4":  ("micro4", "find_best_move"),
    "minimax": ("micro4", "find_best_move"),
}
_solver_fleets: dict = {}       # (engine, n) -> [callable, ...]
_solver_executor = None


def _get_solver_fleet(engine: str, n: int) -> list:
    """Return n independently-loaded solver entry points for `engine`.

    Cached module-level: the copies and their TTs are reused by every eval phase
    for the life of the process.
    """
    import shutil
    import tempfile
    key = (engine, n)
    fleet = _solver_fleets.get(key)
    if fleet is None:
        import ctypes
        from lib.t7g import _find_dll
        basename, symbol = _FLEET_ENGINES[engine]
        src = str(_find_dll(basename))
        tmpdir = tempfile.mkdtemp(prefix="t7g_solver_fleet_")
        fleet = []
        for i in range(n):
            copy = f"{tmpdir}/{basename}_{i}.so"
            shutil.copy(src, copy)
            lib = ctypes.CDLL(copy)
            fn = getattr(lib, symbol)
            fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
            fn.restype = ctypes.c_int
            fleet.append(fn)
        _solver_fleets[key] = fleet
    return fleet


def _get_solver_executor(n: int):
    global _solver_executor
    if _solver_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        _solver_executor = ThreadPoolExecutor(max_workers=n,
                                              thread_name_prefix="t7g-solver")
    return _solver_executor


def _t_engine_moves(reqs: list, threads: int) -> list:
    """Resolve one engine move for each (slot, agent) in `reqs`.

    Returns a list of actions positionally matching `reqs`, with None meaning
    "no move available - pass".  Noise coins, depth jitter and Stauf's move
    index are all drawn HERE so the threaded solvers stay pure functions of
    their arguments and the RNG stream stays in the driver.
    """
    out: list = [None] * len(reqs)
    farmed: dict = {}       # engine -> [(out_idx, board_bytes, depth, turn), ...]
    for i, (slot, agent) in enumerate(reqs):
        _, depth, noise, engine, vary_depth = agent
        legal = np.where(action_masks(slot.board, slot.turn))[0]
        if np.random.random() < noise:
            out[i] = int(np.random.choice(legal))
            continue
        d = int(np.random.choice([4, depth])) if vary_depth else depth
        if engine in UAI_ENGINES:
            from lib.uai_engine import get_worker_engine
            out[i] = get_worker_engine(engine).find_best_move(slot.board, d, slot.turn)
            continue
        if (threads > 1 and d >= _ENGINE_MIN_DEPTH_THREADED
                and engine in _FLEET_ENGINES):
            farmed.setdefault(engine, []).append(
                (i, slot.board.tobytes(), d, slot.turn))
            continue
        if engine == 'stauf':
            # Canonical Stauf: pass the cumulative move index so its internal
            # depths[] cycle matches the real game (see play_eval_game).
            out[i] = find_best_move(slot.board.tobytes(), d, slot.turn, engine,
                                    slot.stauf_moves)
            slot.stauf_moves += 1
        else:
            out[i] = find_best_move(slot.board.tobytes(), d, slot.turn, engine)
    for engine, batch in farmed.items():
        fleet = _get_solver_fleet(engine, threads)
        ex = _get_solver_executor(threads)
        # Chunked to len(fleet): each in-flight call must own a private copy of
        # the solver, or the shared TT race is back.
        for start in range(0, len(batch), len(fleet)):
            chunk = batch[start:start + len(fleet)]
            for (i, *_), action in zip(chunk, ex.map(
                    lambda a: a[0](a[1], a[2], a[3]),
                    [(fleet[j], bb, d, t) for j, (_, bb, d, t) in enumerate(chunk)])):
                out[i] = action
    return [None if a in (-1, 1225) else a for a in out]


def _t_result(slot: _TourneySlot) -> dict:
    """Package a finished slot.  Results are Blue-perspective; the caller signs
    them for whichever side it cares about."""
    blue, green = count_cells(slot.board)
    return {
        "tag": slot.tag,
        "blue_result": slot.result,
        "blue_margin": int(blue - green),
        "moves": slot.move_count,
        "end_reason": slot.end_reason,
    }


def tournament_pool(
    nets: dict,
    games: list,
    mcts_kwargs: dict,
    pool_size: int = 64,
    engine_threads: int | None = None,
):
    """
    Play a list of eval games concurrently, batching inference across games.

    nets:  {key: nn.Module} covering every ("net", key) agent in `games`.
    games: [(tag, blue_agent, green_agent, first_turn), ...] - see the module
           section header for the agent forms.  first_turn is the side to move
           from the standard start position (play_eval_game uses True; the
           net-vs-net gate randomises it).
    pool_size: concurrent games.  A 500-sim search needs ~147 sequential
           forwards no matter how many games are in flight, and each forward
           costs about the same whether it carries 10 leaves or 200 (it is
           latency-bound, not throughput-bound), so wall time is set by the
           NUMBER of forwards and every extra concurrent game rides along for
           free.  The cap is memory: each slot holds one MCGS per net it plays
           and an arena at 500 sims touches tens of MB over a full game.
    engine_threads: threads for expensive minimax moves, each with its own
           dlopen'd copy of the solver (None = auto, 0/1 = inline in the
           driver).

    Yields one dict per finished game (see _t_result) in completion order.
    """
    if not games:
        return
    drivers = {k: MCGS(net, **mcts_kwargs) for k, net in nets.items()}
    any_driver = next(iter(drivers.values()), None)
    n_slots = min(pool_size, len(games))
    slots = [_TourneySlot() for _ in range(n_slots)]
    queue = iter(range(len(games)))

    # Thread the engine moves out only when some engine in this tournament is
    # deep enough to be worth it (see _ENGINE_MIN_DEPTH_THREADED).
    if engine_threads is None:
        deep = any(a[0] == "engine" and a[1] >= _ENGINE_MIN_DEPTH_THREADED
                   and a[3] in _FLEET_ENGINES
                   for _, blue, green, _ in games for a in (blue, green))
        engine_threads = min(n_slots, _ENGINE_MAX_THREADS) if deep else 0

    def _begin(slot: _TourneySlot, gi: int) -> None:
        tag, blue, green, first_turn = games[gi]
        slot.tag, slot.blue, slot.green = tag, blue, green
        slot.board = new_board()
        slot.turn = bool(first_turn)
        slot.clock = 0
        slot.move_count = 0
        slot.stauf_moves = 0
        slot.search = None
        slot.mover = None
        slot.result = 0.0
        slot.end_reason = "terminal"
        for agent in (blue, green):
            if agent[0] == "net":
                key = agent[1]
                if key not in slot.mcts:
                    slot.mcts[key] = MCGS(nets[key], **mcts_kwargs)
                # Fresh tree per game: the TT persists across moves, so it must
                # be cleared between games (the single-game path built a new
                # MCGS per game for exactly this reason).
                slot.mcts[key].clear()

    def _start_next_game(slot: _TourneySlot) -> bool:
        """Put the next queued game on `slot` (undriven).  False if none left."""
        for gi in queue:
            _begin(slot, gi)
            return True
        return False

    # Slots owing an engine move, parked until a fleet-sized batch accumulates.
    # Resolving them the moment they appear is what made the thread fleet
    # useless in the first place: searches drift out of phase within a few
    # plies, so only 0-2 slots ever want an engine move at the same instant
    # (measured mean batch 0.6) and the deep-minimax games ran serially.
    waiting: list = []
    flush_at = max(1, engine_threads)

    def _drive(work: list) -> tuple[list, list]:
        """Settle each slot in `work`, then either start its search or park it.

        Returns (slots now holding a live search, results for games that ended).
        """
        active: list = []
        results: list = []
        for slot in work:
            while True:
                if _t_settle(slot):
                    results.append(_t_result(slot))
                    if _start_next_game(slot):
                        continue            # same slot, next queued game
                    break
                agent = slot.blue if slot.turn else slot.green
                if agent[0] == "net":
                    slot.mover = agent[1]
                    slot.search = slot.mcts[agent[1]].start_search(
                        slot.board, slot.turn, clock=slot.clock)
                    active.append(slot)
                else:
                    waiting.append((slot, agent))
                break
        return active, results

    def _flush_engine() -> tuple[list, list]:
        """Resolve every parked engine move in one batch, then re-drive."""
        batch, waiting[:] = list(waiting), []
        results: list = []
        work: list = []
        for (slot, agent), action in zip(batch,
                                         _t_engine_moves(batch, engine_threads)):
            if action is None:                  # no legal move: pass
                slot.turn = not slot.turn
                slot.clock += 1
            elif _t_apply(slot, action):        # move cap ended the game
                results.append(_t_result(slot))
                if not _start_next_game(slot):
                    continue
            work.append(slot)
        active, more = _drive(work)
        return active, results + more

    def _launch(group: list) -> list:
        """One forward per distinct network to move in `group`.

        The count of these calls is what eval wall-clock is made of (each costs
        ~1.3-1.8 ms of dispatch regardless of batch size), so the pool keeps
        every slot in ONE stage and issues the minimum: one launch per network
        that has a search waiting.  With both players' games phase-aligned that
        is usually 1-2 per step for the whole pool.
        """
        by_net: dict = {}
        for slot in group:
            by_net.setdefault(slot.mover, []).append(slot.search)
        return [drivers[k]._launch_forward(v) for k, v in by_net.items()]

    def _advance(group: list) -> tuple[list, list]:
        """Step every search, play the moves that completed, re-drive their slots."""
        rerun: list = []
        keep: list = []
        done_flags = step_searches([slot.search for slot in group])
        for slot, search_done in zip(group, done_flags):
            if not search_done:
                keep.append(slot)
                continue
            action_probs = slot.search.result
            best_action = slot.search.best_action
            if not np.any(action_probs):
                # Slab overflow: recover with uniform-over-legal, same as the
                # self-play pool.  A genuine pass or terminal cannot appear here
                # - _drive only starts a search when a legal move exists.
                masks = action_masks(slot.board, slot.turn).astype(np.float32)
                action_probs = masks / masks.sum()
                best_action = -1
            action = drivers[slot.mover].select_action(
                action_probs, board=slot.board, turn=slot.turn,
                temperature=0, best_action=best_action,
            )
            slot.search = None
            rerun.append((slot, _t_apply(slot, action)))
        results: list = []
        work: list = []
        for slot, ended in rerun:
            if ended:
                results.append(_t_result(slot))
                if not _start_next_game(slot):
                    continue
            work.append(slot)
        active, more = _drive(work)
        return keep + active, results + more

    try:
        work = []
        for slot in slots:
            if _start_next_game(slot):
                work.append(slot)
        active, results = _drive(work)
        for r in results:
            yield r

        handles = _launch(active) if active else []
        while active or waiting:
            if handles:
                for h in handles:
                    any_driver._collect_and_commit(h)  # handle carries its slabs
                active, results = _advance(active)
                for r in results:
                    yield r
            # Flush when a full batch has piled up, or when the search side has
            # nothing left to do (which is also what guarantees progress).
            if waiting and (len(waiting) >= flush_at or not active):
                more_active, results = _flush_engine()
                active = active + more_active
                for r in results:
                    yield r
            handles = _launch(active) if active else []
    finally:
        # Free the arenas promptly - at 500 sims a full game touches tens of MB
        # per instance, and the self-play pool's 256-512 instances are still
        # alive alongside these.
        for slot in slots:
            slot.mcts.clear()
        drivers.clear()


# ---------------------------------------------------------------------------
# Evaluation vs minimax
# ---------------------------------------------------------------------------

def play_eval_game(
    mcts: MCGS,
    minimax_depth: int,
    noise: float,
    engine: str,
    vary_depth: bool,
    mcts_is_blue: bool,
) -> tuple[float, str, int, int]:
    """
    Play one evaluation game (MCTS vs minimax/stauf).

    Returns (result, end_reason, margin, moves) where:
      result     : float in [−1, +1] from the MCTS agent's perspective
      end_reason : "terminal" | "clock" | "truncated"
      margin     : final material margin in pieces (blue − green), signed to
                   the MCTS agent
      moves      : half-moves played (game length)
    Decisive terminal positions give ±1.0; halfmove-clock expiry gives 0.0 (draw);
    truncation gives a material ratio: (blue − green) / (blue + green).
    """
    board = new_board()
    mcts.clear()
    turn = True  # Blue moves first (eval games always start standard)
    clock = 0
    move_count = 0
    stauf_moves = 0   # cumulative Stauf move index, for its depths[] cycle
    end_reason = "terminal"

    while True:
        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            end_reason = "terminal"
            break

        if clock >= CLOCK_LIMIT:
            blue_result = 0.0  # halfmove clock expired = draw (libataxx rule)
            end_reason = "clock"
            break

        mcts_turn = (turn == mcts_is_blue)

        if mcts_turn:
            if not np.any(action_masks(board, turn)):
                mcts.advance_tree(1225)
                turn = not turn
                clock += 1
                continue
            action_probs = mcts.search(board, turn, clock=clock)
            action = mcts.select_action(action_probs, board=board, turn=turn, temperature=0,
                                        best_action=mcts.last_best_action)
            mcts.advance_tree(action)
        else:
            legal = np.where(action_masks(board, turn))[0]
            if len(legal) == 0:
                turn = not turn
                clock += 1
                continue
            if np.random.random() < noise:
                action = int(np.random.choice(legal))
            else:
                depth = int(np.random.choice([4, minimax_depth])) if vary_depth else minimax_depth
                if engine == 'stauf':
                    # Canonical Stauf: pass its cumulative move index so the
                    # internal depths[] cycle matches the real game rather than
                    # a random slot (identify_stauf.py / find_stauf_line.py).
                    action = find_best_move(board.tobytes(), depth, turn, engine, stauf_moves)
                    stauf_moves += 1
                elif engine in UAI_ENGINES:
                    from lib.uai_engine import get_worker_engine
                    action = get_worker_engine(engine).find_best_move(board, depth, turn)
                else:
                    action = find_best_move(board.tobytes(), depth, turn, engine)
                if action in (-1, 1225):
                    turn = not turn
                    clock += 1
                    continue

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1
        clock = tick_clock(clock, action)

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            end_reason = "truncated"
            break

    blue, green = count_cells(board)
    margin = (blue - green) if mcts_is_blue else (green - blue)
    return (blue_result if mcts_is_blue else -blue_result), end_reason, int(margin), move_count


# ---------------------------------------------------------------------------
# Gate: network vs network
# ---------------------------------------------------------------------------

def play_net_vs_net_game(
    mcts_new: MCGS,
    mcts_best: MCGS,
    new_is_blue: bool,
) -> tuple[float, int, int]:
    """
    Play one gate game between two MCTS agents.

    Returns (result, margin, moves) from mcts_new's perspective:
      result : float in [−1, +1].  Decisive terminal positions give ±1.0;
               halfmove-clock expiry gives 0.0 (draw); truncation gives a
               material ratio: (blue − green) / (blue + green).
      margin : final material margin in pieces (blue − green), signed to
               mcts_new -- how crushing the win / bad the loss actually was.
      moves  : half-moves played (game length).
    Starting colour is randomised to neutralise first-mover advantage.
    """
    board = new_board()
    mcts_new.root = None
    mcts_best.root = None
    turn = bool(np.random.randint(2))
    clock = 0
    move_count = 0

    while True:
        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        if clock >= CLOCK_LIMIT:
            blue_result = 0.0  # halfmove clock expired = draw (libataxx rule)
            break

        new_turn = (turn == new_is_blue)
        mcts_active = mcts_new if new_turn else mcts_best
        mcts_passive = mcts_best if new_turn else mcts_new

        if not np.any(action_masks(board, turn)):
            mcts_active.advance_tree(1225)
            mcts_passive.advance_tree(1225)
            turn = not turn
            clock += 1
            continue

        action_probs = mcts_active.search(board, turn, clock=clock)
        action = mcts_active.select_action(action_probs, board=board, turn=turn, temperature=0,
                                           best_action=mcts_active.last_best_action)
        mcts_active.advance_tree(action)
        mcts_passive.advance_tree(action)

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1
        clock = tick_clock(clock, action)

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            break

    blue, green = count_cells(board)
    margin = (blue - green) if new_is_blue else (green - blue)
    return (blue_result if new_is_blue else -blue_result), int(margin), move_count


# ---------------------------------------------------------------------------
# Deterministic engine vs engine (low-end ladder rating)
# ---------------------------------------------------------------------------

def play_engine_vs_engine(
    spec_a: tuple[str, int],
    spec_b: tuple[str, int],
    a_is_blue: bool,
    opening_plies: int = 4,
    rng=None,
    max_moves: int = 200,
) -> int:
    """Play one game between two deterministic engines from a random opening.

    Each spec is ``(engine, depth)`` -- e.g. ``("stauf", 6)`` for the canonical
    original-game AI, or ``("micro3", 7)`` for depth-7 minimax.  Both engines
    are deterministic, so *the randomised opening is the only source of game
    variety* and is what makes a Bradley-Terry / WHR rating identifiable: play
    many distinct openings (and both colours) per pairing.

    The opening is ``opening_plies`` uniform-random legal moves (alternating
    colours) from the standard start; the engines then play it out.  Stauf's
    cumulative move index is tracked and passed as ``move_count`` so its
    depths[] cycle matches the real game.

    Returns the discretised result in {+1, 0, -1} from a's perspective
    (material-ratio truncation collapsed to a win/draw/loss, matching how the
    net drivers are scored for BT).
    """
    rng = np.random.default_rng() if rng is None else rng
    board = new_board()
    turn = True  # Blue moves first

    # --- randomised opening: uniform-random legal plies, alternating colours ---
    for _ in range(opening_plies):
        is_terminal, _ = check_terminal(board, turn)
        if is_terminal:
            break
        legal = np.where(action_masks(board, turn))[0]
        if len(legal) == 0:
            turn = not turn
            continue
        board = apply_move(board, int(rng.choice(legal)), turn)
        turn = not turn

    # --- engines play it out (deterministic) ---
    clock = 0
    stauf_moves = {"a": 0, "b": 0}   # per-side cumulative Stauf move index
    move_count = 0
    blue_result = 0.0
    while True:
        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        if clock >= CLOCK_LIMIT:
            blue_result = 0.0  # halfmove clock expired = draw (libataxx rule)
            break

        side = "a" if (turn == a_is_blue) else "b"
        engine, depth = spec_a if side == "a" else spec_b

        if not np.any(action_masks(board, turn)):
            turn = not turn
            clock += 1
            continue
        if engine == "stauf":
            action = find_best_move(board.tobytes(), depth, turn, engine, stauf_moves[side])
            stauf_moves[side] += 1
        elif engine in UAI_ENGINES:
            from lib.uai_engine import get_worker_engine
            action = get_worker_engine(engine).find_best_move(board, depth, turn)
        else:
            action = find_best_move(board.tobytes(), depth, turn, engine)
        if action in (-1, 1225):
            turn = not turn
            clock += 1
            continue

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1
        clock = tick_clock(clock, action)
        if move_count > max_moves:
            blue, green = count_cells(board)
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            break

    result = blue_result if a_is_blue else -blue_result
    return 1 if result > 1e-9 else (-1 if result < -1e-9 else 0)
