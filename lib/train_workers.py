"""
Game-logic helpers for AlphaZero self-play training.

Stateless (no module-level globals) to allow multiprocessing.
"""
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

class _GameSlot:
    """State for one concurrent game inside self_play_game_pool."""

    __slots__ = [
        'board', 'turn', 'examples', 'move_count', 'truncated',
        'clock', 'legal_move_counts', 'game_start',
        'search', 'mcts', 'full_move',
    ]

    def __init__(self, mcts: MCGS) -> None:
        self.mcts = mcts
        self.board = new_board()
        self.turn = bool(np.random.randint(2))
        self.examples: list = []
        self.move_count = 0
        self.truncated = False
        self.clock = 0     # halfmove clock: plies since the last clone move
        self.legal_move_counts: list = []
        self.game_start = time.time()
        self.search: MCGSSearch | None = None
        self.full_move = True   # whether the in-flight search runs the full cap


def _slot_result(
    slot: _GameSlot,
    winner: float,
    blend_alpha: float = 1.0,
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
    winner            : +1.0 Blue / −1.0 Green / material ratio for truncated games
    move_count        : number of half-moves played
    elapsed           : wall time in seconds
    truncated         : True if the 200-move cap triggered
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
    n_ex = len(slot.examples)
    lambdas = np.asarray(ST_LAMBDAS, dtype=np.float32)
    st_blue = np.empty((n_ex, len(ST_LAMBDAS)), dtype=np.float32)
    acc = np.full(len(ST_LAMBDAS), winner, dtype=np.float32)
    for j in range(n_ex - 1, -1, -1):
        _, _, j_turn, _, j_q, _, _ = slot.examples[j]
        q_blue = j_q if j_turn else -j_q
        acc = (1.0 - lambdas) * q_blue + lambdas * acc
        st_blue[j] = acc

    examples = []
    for i, (obs, policy_target, example_turn, ex_board, _root_q, move_idx, full_move) in \
            enumerate(slot.examples):
        value_target = winner if example_turn else -winner
        st_targets = st_blue[i] if example_turn else -st_blue[i]
        if full_move:
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
    return examples, winner, slot.move_count, elapsed, slot.truncated, slot.legal_move_counts


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
    slot.truncated = False
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
                results.append(_slot_result(slot, winner, blend_alpha))
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
                    results.append(_slot_result(slot, 0.0, blend_alpha))
                    if games_started < target_games:
                        _reset_slot(slot)
                        _start_slot_search(slot, full_sims, pcr_p_full, pcr_fast_sims)
                        next_active.append(slot)
                        games_started += 1
                elif slot.move_count > 200:
                    blue, green = count_cells(slot.board)
                    winner = (float(blue - green) / float(blue + green)
                              if blue + green > 0 else 0.0)
                    slot.truncated = True
                    results.append(_slot_result(slot, winner, blend_alpha))
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
        elif slot.move_count > 200:
            blue, green = count_cells(slot.board)
            winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            slot.truncated = True
            done = True

        if done:
            results.append(_slot_result(slot, winner, blend_alpha))
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
        (training_examples, winner, move_count, elapsed, truncated, legal_move_counts)
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
        )
        for r in results:
            yield r
        handle_b = (mcts._launch_forward([s.search for s in active_b])
                    if active_b else None)


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
