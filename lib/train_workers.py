"""
Game-logic helpers for AlphaZero self-play training.

Stateless (no module-level globals) to allow multiprocessing.
"""
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from lib.mcgs import MCGS, MCGSSearch
from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move,
)


# ---------------------------------------------------------------------------
# Value-target blending (Option A + B)
# ---------------------------------------------------------------------------
#
# See docs/value_blending.md for the full rationale and the Option D swap-in
# recipe.  Short version: the final value target is
#
#     value_target = α * terminal + (1 - α) * root_q
#
# where α is computed per-example so that Q influence is suppressed in
# regimes where Q is known to be unreliable:
#
#   (A) Phase ramp: Q weight ramps from 0 → full over `BLEND_RAMP_LEN`
#       moves starting at `temp_moves` (TEMP_THRESHOLD).  Before temp
#       threshold MCTS is sampling with temperature=1 (exploration noise
#       in the visit distribution), so root Q is not a clean value
#       estimate.  After temp threshold MCTS commits, Q becomes informative.
#
#   (B) Visit concentration gate: if the root visit distribution is flat
#       (MCTS uncertain), down-weight Q further.  Concentration is
#       1 - normalised_entropy; a peaked distribution → 1, uniform → 0.
#
# The two gates combine multiplicatively: both must fire for Q to reach
# full weight.  This is the fix for the value-head-collapses-to-zero
# failure we saw with un-gated blending at α=0.5, where early-game
# near-zero root Q taught the network "openings = 0" as a shortcut.
# ---------------------------------------------------------------------------

BLEND_RAMP_LEN = 10  # moves past temp_moves over which Q-weight ramps to full


def _blended_value_target(
    winner: float,
    was_blue: bool,
    root_q: float,
    move_idx: int,
    policy_target: np.ndarray,
    blend_alpha: float,
    temp_moves: int,
    ramp_len: int = BLEND_RAMP_LEN,
) -> float:
    """
    Compute the value target for a single (state, policy, outcome) example.

    When blend_alpha == 1.0 (default), returns pure terminal target (original
    behaviour; no-op fast path). Otherwise applies phase × concentration
    gating described above.

    Parameters
    ----------
    winner         : +1 Blue / -1 Green / material ratio (Blue perspective)
    was_blue       : True if the example's side-to-move was Blue
    root_q         : MCTS root value at example generation (side-to-move perspective)
    move_idx       : move number when the example was recorded
    policy_target  : MCTS visit-weighted policy at the root (used for concentration)
    blend_alpha    : maximum α used for pure terminal (1 - blend_alpha = max Q weight)
    temp_moves     : TEMP_THRESHOLD; Q weight is zero before this move
    ramp_len       : moves over which Q weight ramps to its maximum

    Returns
    -------
    value_target in [-1, +1], side-to-move perspective
    """
    terminal = winner if was_blue else -winner

    if blend_alpha >= 1.0:
        return terminal  # fast path: pure terminal, no gating overhead

    # (A) Phase ramp: 0 before temp_moves, linear to 1 over ramp_len moves.
    phase = max(0.0, min(1.0, (move_idx - temp_moves) / max(1, ramp_len)))

    # (B) Concentration of MCTS visit distribution: 1.0 = one-hot, 0.0 = uniform.
    support = policy_target[policy_target > 1e-8]
    if support.size <= 1:
        concentration = 1.0 if support.size == 1 else 0.0
    else:
        entropy     = float(-np.sum(support * np.log(support)))
        max_entropy = float(np.log(support.size))
        concentration = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

    gate        = phase * concentration     # both must fire
    q_weight    = (1.0 - blend_alpha) * gate
    alpha_eff   = 1.0 - q_weight
    return alpha_eff * terminal + q_weight * root_q


# ---------------------------------------------------------------------------
# Self-play (single game - used for eval and as fallback)
# ---------------------------------------------------------------------------

def self_play_game(mcts: MCGS, temp_moves: int = 0, blend_alpha: float = 1.0):
    """
    Play one game via MCTS self-play, collecting training examples.

    blend_alpha controls terminal vs Q-value blending:
        value_target = blend_alpha * terminal + (1 - blend_alpha) * root_q

    Returns
    -------
    training_examples : list of (obs, policy_target, value_target)
    winner            : +1.0 Blue / -1.0 Green / 0.0 draw (Blue perspective)
    move_count        : number of half-moves played
    elapsed           : wall time in seconds
    truncated         : True if the 200-move cap triggered
    legal_move_counts : per-position branching factor samples
    """
    mcts.clear()

    board = new_board()
    turn = bool(np.random.randint(2))
    examples = []
    move_count = 0
    truncated = False
    board_history: dict = {}
    legal_move_counts: list = []
    game_start = time.time()

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            winner = 0.0  # repetition = draw
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            winner = terminal_value if turn else -terminal_value
            break

        masks = action_masks(board, turn)
        legal_count = int(masks.sum())
        if legal_count == 0:
            mcts.advance_tree(1225)
            turn = not turn
            continue
        legal_move_counts.append(legal_count)

        action_probs = mcts.search(board, turn)
        root_q = getattr(mcts, 'last_root_value', 0.0)
        obs = board_to_obs(board, turn)
        examples.append((obs, action_probs, turn, board.copy(), root_q, move_count))

        temp = 1.0 if move_count < temp_moves else 0.0
        action = mcts.select_action(action_probs, board=board, turn=turn, temperature=temp)

        board = apply_move(board, action, turn)
        mcts.advance_tree(action)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            truncated = True
            break

    blue, green = count_cells(board)
    margin_blue = float(blue - green) / 49.0  # final material margin, Blue perspective

    training_examples = []
    for obs, policy_target, example_turn, _, root_q, move_idx in examples:
        value_target = _blended_value_target(
            winner=winner, was_blue=example_turn, root_q=root_q,
            move_idx=move_idx, policy_target=policy_target,
            blend_alpha=blend_alpha, temp_moves=temp_moves,
        )
        margin = margin_blue if example_turn else -margin_blue
        training_examples.append((obs, policy_target, value_target, margin))

    elapsed = time.time() - game_start
    return training_examples, winner, move_count, elapsed, truncated, legal_move_counts


# ---------------------------------------------------------------------------
# In-process game pool - batched inference across concurrent games
# ---------------------------------------------------------------------------

class _GameSlot:
    """State for one concurrent game inside self_play_game_pool."""

    __slots__ = [
        'board', 'turn', 'examples', 'move_count', 'truncated',
        'board_history', 'legal_move_counts', 'game_start',
        'search', 'mcts',
    ]

    def __init__(self, mcts: MCGS) -> None:
        self.mcts = mcts
        self.board = new_board()
        self.turn = bool(np.random.randint(2))
        self.examples: list = []
        self.move_count = 0
        self.truncated = False
        self.board_history: dict = {}
        self.legal_move_counts: list = []
        self.game_start = time.time()
        self.search: MCGSSearch | None = None


def _slot_result(
    slot: _GameSlot,
    winner: float,
    blend_alpha: float = 1.0,
    temp_moves: int = 0,
) -> tuple:
    """Package a finished slot into a result tuple matching self_play_game_pool's yield contract.

    Returns
    -------
    training_examples : list of (obs, raw_policy, value_target, margin, board, turn)
        6-tuples - board and turn included so the caller can apply policy
        relabeling outside the GPU-critical pool loop.  margin is the final
        material margin / 49 from the example's side-to-move perspective.
    winner            : +1.0 Blue / −1.0 Green / material ratio for truncated games
    move_count        : number of half-moves played
    elapsed           : wall time in seconds
    truncated         : True if the 200-move cap triggered
    legal_move_counts : per-position branching factor samples
    """
    blue, green = count_cells(slot.board)
    margin_blue = float(blue - green) / 49.0

    examples = []
    for obs, policy_target, example_turn, ex_board, root_q, move_idx in slot.examples:
        value_target = _blended_value_target(
            winner=winner, was_blue=example_turn, root_q=root_q,
            move_idx=move_idx, policy_target=policy_target,
            blend_alpha=blend_alpha, temp_moves=temp_moves,
        )
        margin = margin_blue if example_turn else -margin_blue
        examples.append((obs, policy_target, value_target, margin,
                         ex_board, example_turn))
    elapsed = time.time() - slot.game_start
    return examples, winner, slot.move_count, elapsed, slot.truncated, slot.legal_move_counts


def _reset_slot(slot: _GameSlot) -> None:
    """Reset a finished slot's game state so it can play a new game."""
    slot.mcts.clear()
    slot.board = new_board()
    slot.turn = bool(np.random.randint(2))
    slot.examples = []
    slot.move_count = 0
    slot.truncated = False
    slot.board_history = {}
    slot.legal_move_counts = []
    slot.game_start = time.time()
    slot.search = None


def _advance_group(
    active: list,
    target_games: int,
    games_started: int,
    temp_moves: int,
    blend_alpha: float,
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

    for slot in active:
        slot.search.step()

        if not slot.search.done:
            next_active.append(slot)
            continue

        action_probs = slot.search.result
        root_q = slot.search.root_value
        if not np.any(action_probs):
            is_terminal, terminal_value = check_terminal(slot.board, slot.turn)
            if is_terminal:
                assert terminal_value is not None
                winner = terminal_value if slot.turn else -terminal_value
                results.append(_slot_result(slot, winner, blend_alpha, temp_moves))
                if games_started < target_games:
                    _reset_slot(slot)
                    slot.search = slot.mcts.start_search(slot.board, slot.turn, None)
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
                # Fall through to normal action-selection below.
            else:
                # Genuine forced pass.
                slot.turn = not slot.turn
                slot.move_count += 1
                if slot.move_count > 200:
                    blue, green = count_cells(slot.board)
                    winner = (float(blue - green) / float(blue + green)
                              if blue + green > 0 else 0.0)
                    slot.truncated = True
                    results.append(_slot_result(slot, winner, blend_alpha, temp_moves))
                    if games_started < target_games:
                        _reset_slot(slot)
                        slot.search = slot.mcts.start_search(slot.board, slot.turn, None)
                        next_active.append(slot)
                        games_started += 1
                else:
                    slot.search = slot.mcts.start_search(slot.board, slot.turn, None)
                    next_active.append(slot)
                continue

        obs = board_to_obs(slot.board, slot.turn)
        slot.examples.append(
            (obs, action_probs, slot.turn, slot.board.copy(), root_q, slot.move_count)
        )

        temp = 1.0 if slot.move_count < temp_moves else 0.0
        action = slot.mcts.select_action(
            action_probs, board=slot.board, turn=slot.turn, temperature=temp,
        )
        slot.board = apply_move(slot.board, action, slot.turn)
        slot.turn = not slot.turn
        slot.move_count += 1

        state_key = slot.board.tobytes() + bytes([slot.turn])
        slot.board_history[state_key] = slot.board_history.get(state_key, 0) + 1
        is_terminal, terminal_value = check_terminal(slot.board, slot.turn)

        done = False
        winner = 0.0
        if slot.board_history[state_key] >= 3:
            winner = 0.0  # repetition = draw
            done = True
        elif is_terminal:
            assert terminal_value is not None
            winner = terminal_value if slot.turn else -terminal_value
            done = True
        elif slot.move_count > 200:
            blue, green = count_cells(slot.board)
            winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            slot.truncated = True
            done = True

        if done:
            results.append(_slot_result(slot, winner, blend_alpha, temp_moves))
            if games_started < target_games:
                _reset_slot(slot)
                slot.search = slot.mcts.start_search(slot.board, slot.turn, None)
                next_active.append(slot)
                games_started += 1
        else:
            masks = action_masks(slot.board, slot.turn)
            if not np.any(masks):
                slot.turn = not slot.turn
            else:
                slot.legal_move_counts.append(int(masks.sum()))
            slot.search = slot.mcts.start_search(slot.board, slot.turn, move_count=slot.move_count)
            next_active.append(slot)

    return next_active, games_started, results


def self_play_game_pool(
    mcts: MCGS,
    pool_size: int,
    target_games: int,
    mcts_pool: 'list[MCGS] | None' = None,
    temp_moves: int = 0,
    blend_alpha: float = 1.0,
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

    Yields result tuples as each game completes:
        (training_examples, winner, move_count, elapsed, truncated, legal_move_counts)
    """
    if mcts_pool is not None:
        slots = [_GameSlot(m) for m in mcts_pool[:pool_size]]
    else:
        slots = [
            _GameSlot(MCGS(
                mcts.network,
                num_simulations=mcts.num_simulations,
                c_puct=mcts.c_puct,
                gumbel_k=mcts.gumbel_k,
            ))
            for _ in range(pool_size)
        ]
    for slot in slots:
        slot.mcts.clear()  # ensure no stale TT from a previous pool run
        slot.search = slot.mcts.start_search(slot.board, slot.turn, None)

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
        )
        for r in results:
            yield r
        handle_a = (mcts._launch_forward([s.search for s in active_a])
                    if active_a else None)

        # --- Group B: same thing, with GPU now busy on A's next forward.
        mcts._collect_and_commit(handle_b)
        active_b, games_started, results = _advance_group(
            active_b, target_games, games_started, temp_moves, blend_alpha,
        )
        for r in results:
            yield r
        handle_b = (mcts._launch_forward([s.search for s in active_b])
                    if active_b else None)


# ---------------------------------------------------------------------------
# MM-mix pool (MCTS vs minimax, batched)
# ---------------------------------------------------------------------------

class _MMMixSlot:
    """One concurrent MCTS-vs-MM game in generate_mm_mix_pool."""
    __slots__ = (
        'mcts', 'mcts_is_blue',
        'board', 'turn', 'examples',
        'move_count', 'board_history', 'game_start',
        'truncated', 'legal_move_counts',
        'search', 'mm_future',
    )

    def __init__(self, mcts_inst: MCGS, mcts_is_blue: bool) -> None:
        self.mcts            = mcts_inst
        self.mcts_is_blue    = mcts_is_blue
        self.board           = new_board()
        self.turn            = True
        self.examples: list  = []
        self.move_count      = 0
        self.board_history: dict = {}
        self.game_start      = time.time()
        self.truncated       = False
        self.legal_move_counts: list = []
        self.search:    MCGSSearch | None   = None
        self.mm_future: Future[int] | None = None


def _mm_mix_reset(slot: _MMMixSlot, mcts_is_blue: bool) -> None:
    slot.mcts.clear()
    slot.mcts_is_blue    = mcts_is_blue
    slot.board           = new_board()
    slot.turn            = True
    slot.examples        = []
    slot.move_count      = 0
    slot.board_history   = {}
    slot.game_start      = time.time()
    slot.truncated       = False
    slot.legal_move_counts = []
    slot.search          = None
    slot.mm_future       = None


def _mm_mix_start_move(slot: _MMMixSlot, thread_ex, mm_depth: int) -> None:
    """Kick off the next move - MCTS search or threaded MM call."""
    if slot.turn == slot.mcts_is_blue:
        slot.search = slot.mcts.start_search(slot.board, slot.turn, None)
    else:
        board_bytes = slot.board.tobytes()
        turn = slot.turn
        slot.mm_future = thread_ex.submit(find_best_move, board_bytes, mm_depth, turn)


def generate_mm_mix_pool(
    mcts: MCGS,
    num_games: int,
    pool_size: int = 16,
    mm_depth: int = 3,
    mcts_pool: 'list[MCGS] | None' = None,
):
    """
    Play *num_games* MCTS-vs-MM games using a concurrent pool.

    MCTS turns are batched across all active slots for GPU efficiency.
    MM turns run via a thread pool - ctypes releases the GIL so they
    overlap with GPU work from other slots.

    Only MCTS-side positions are recorded as training examples.

    Yields (same tuple as _slot_result)
    ------
    (examples, winner, move_count, elapsed, truncated, legal_move_counts)
        examples : list of (obs, policy, value_target, board, turn)
        winner   : +1 blue / -1 green (blue perspective)
    """
    from concurrent.futures import ThreadPoolExecutor, wait as cf_wait, FIRST_COMPLETED

    if mcts_pool is not None:
        mcts_insts = mcts_pool[:pool_size]
    else:
        mcts_insts = [
            MCGS(mcts.network,
                 num_simulations=mcts.num_simulations,
                 c_puct=mcts.c_puct,
                 gumbel_k=mcts.gumbel_k)
            for _ in range(pool_size)
        ]

    slots = [_MMMixSlot(mi, i % 2 == 0) for i, mi in enumerate(mcts_insts)]
    for s in slots:
        s.mcts.clear()

    games_started = pool_size
    active = list(slots)

    with ThreadPoolExecutor(max_workers=pool_size) as thread_ex:
        for slot in active:
            _mm_mix_start_move(slot, thread_ex, mm_depth)

        while active:
            # Batch-expand pending MCTS leaves across all active searches.
            searching = [s for s in active if s.search is not None]
            if searching:
                mcts._expand_batch([s.search for s in searching if s.search is not None])

            next_active: list = []
            for slot in active:

                #  MCTS turn 
                if slot.search is not None:
                    slot.search.step()  # type: ignore[union-attr]
                    if not slot.search.done:
                        next_active.append(slot)
                        continue

                    action_probs = slot.search.result
                    slot.search = None

                    if not np.any(action_probs):
                        # Forced pass (PASS action unrepresentable in result array).
                        slot.turn = not slot.turn
                        slot.move_count += 1
                    else:
                        obs = board_to_obs(slot.board, slot.turn)
                        slot.examples.append(
                            (obs, action_probs, slot.turn, slot.board.copy())
                        )
                        action = slot.mcts.select_action(
                            action_probs,
                            board=slot.board,
                            turn=slot.turn,
                            temperature=1.0,
                        )
                        slot.board = apply_move(slot.board, action, slot.turn)
                        slot.turn  = not slot.turn
                        slot.move_count += 1

                #  MM turn 
                elif slot.mm_future is not None:
                    if not slot.mm_future.done():
                        next_active.append(slot)
                        continue

                    action = int(slot.mm_future.result())
                    slot.mm_future = None

                    if action not in (-1, 1225):
                        slot.board = apply_move(slot.board, action, slot.turn)
                    slot.turn = not slot.turn
                    slot.move_count += 1

                else:
                    next_active.append(slot)
                    continue

                #  After any move: check termination 
                state_key = slot.board.tobytes() + bytes([slot.turn])
                slot.board_history[state_key] = slot.board_history.get(state_key, 0) + 1
                is_terminal, terminal_value = check_terminal(slot.board, slot.turn)

                done = False
                winner = 0.0
                if slot.board_history[state_key] >= 3:
                    blue, green = count_cells(slot.board)
                    winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
                    done = True
                elif is_terminal:
                    assert terminal_value is not None
                    winner = terminal_value if slot.turn else -terminal_value
                    done = True
                elif slot.move_count > 200:
                    blue, green = count_cells(slot.board)
                    winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
                    slot.truncated = True
                    done = True

                if done:
                    yield _slot_result(slot, winner)
                    if games_started < num_games:
                        _mm_mix_reset(slot, games_started % 2 == 0)
                        games_started += 1
                        _mm_mix_start_move(slot, thread_ex, mm_depth)
                        next_active.append(slot)
                else:
                    _mm_mix_start_move(slot, thread_ex, mm_depth)
                    next_active.append(slot)

            active = next_active

            # Avoid busy-spinning when every remaining slot is waiting on an MM future.
            if active and not any(s.search is not None for s in active):
                pending = [s.mm_future for s in active if s.mm_future is not None]
                if pending:
                    cf_wait(pending, return_when=FIRST_COMPLETED)


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
) -> tuple[float, str]:
    """
    Play one evaluation game (MCTS vs minimax/stauf).

    Returns (result, end_reason) where:
      result     : float in [−1, +1] from the MCTS agent's perspective
      end_reason : "terminal" | "repetition" | "truncated"
    Decisive terminal positions give ±1.0; repetition gives 0.0 (draw);
    truncation gives a material ratio: (blue − green) / (blue + green).
    """
    board = new_board()
    mcts.clear()
    turn = True  # Blue moves first (eval games always start standard)
    board_history: dict = {}
    move_count = 0
    end_reason = "terminal"

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue_result = 0.0  # repetition = draw
            end_reason = "repetition"
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            end_reason = "terminal"
            break

        mcts_turn = (turn == mcts_is_blue)

        if mcts_turn:
            if not np.any(action_masks(board, turn)):
                mcts.advance_tree(1225)
                turn = not turn
                continue
            action_probs = mcts.search(board, turn)
            action = mcts.select_action(action_probs, board=board, turn=turn, temperature=0)
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
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            end_reason = "truncated"
            break

    return (blue_result if mcts_is_blue else -blue_result), end_reason


# ---------------------------------------------------------------------------
# Gate: network vs network
# ---------------------------------------------------------------------------

def play_net_vs_net_game(
    mcts_new: MCGS,
    mcts_best: MCGS,
    new_is_blue: bool,
) -> float:
    """
    Play one gate game between two MCTS agents.

    Returns a float in [−1, +1] from mcts_new's perspective.
    Decisive terminal positions give ±1.0; repetition gives 0.0 (draw);
    truncation gives a material ratio: (blue − green) / (blue + green).
    Starting colour is randomised to neutralise first-mover advantage.
    """
    board = new_board()
    mcts_new.root = None
    mcts_best.root = None
    turn = bool(np.random.randint(2))
    board_history: dict = {}
    move_count = 0

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue_result = 0.0  # repetition = draw
            break

        is_terminal, terminal_value = check_terminal(board, turn)
        if is_terminal:
            assert terminal_value is not None
            blue_result = terminal_value if turn else -terminal_value
            break

        new_turn = (turn == new_is_blue)
        mcts_active = mcts_new if new_turn else mcts_best
        mcts_passive = mcts_best if new_turn else mcts_new

        if not np.any(action_masks(board, turn)):
            mcts_active.advance_tree(1225)
            mcts_passive.advance_tree(1225)
            turn = not turn
            continue

        action_probs = mcts_active.search(board, turn)
        action = mcts_active.select_action(action_probs, board=board, turn=turn, temperature=0)
        mcts_active.advance_tree(action)
        mcts_passive.advance_tree(action)

        board = apply_move(board, action, turn)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            break

    return blue_result if new_is_blue else -blue_result


# ---------------------------------------------------------------------------
# MM-mix data generation (GPU-batched MCTS + threaded MM)
# ---------------------------------------------------------------------------

class _MMDataSlot:
    """State for one concurrent MCTS-vs-MM game inside generate_mm_mix_data."""

    __slots__ = [
        'mcts', 'mcts_is_blue', 'board', 'turn',
        'history', 'board_history', 'move_count',
        'search', 'done', 'outcome',
    ]

    def __init__(self, mcts: MCGS, game_idx: int) -> None:
        self.mcts         = mcts
        self.mcts_is_blue = game_idx % 2 == 0
        self.board        = new_board()
        self.turn         = True
        self.history: list[tuple]  = []
        self.board_history: dict   = {}
        self.move_count   = 0
        self.search       = None
        self.done         = False
        self.outcome      = 0.0


def _mmdata_reset(slot: _MMDataSlot, game_idx: int) -> None:
    slot.mcts.clear()
    slot.mcts_is_blue  = game_idx % 2 == 0
    slot.board         = new_board()
    slot.turn          = True
    slot.history       = []
    slot.board_history = {}
    slot.move_count    = 0
    slot.search        = None
    slot.done          = False
    slot.outcome       = 0.0


def _mmdata_advance(slot: _MMDataSlot, mm_depth: int) -> None:
    """Advance through MM turns until the MCTS player's turn or game over.
    Safe to call from a thread - find_best_move releases the GIL."""
    while True:
        state_key = slot.board.tobytes() + bytes([slot.turn])
        slot.board_history[state_key] = slot.board_history.get(state_key, 0) + 1
        if slot.board_history[state_key] >= 3:
            blue, green = count_cells(slot.board)
            slot.outcome = float(blue - green) / (blue + green) if blue + green else 0.0
            slot.done = True
            return

        is_t, val = check_terminal(slot.board, slot.turn)
        if is_t:
            slot.outcome = (float(val) if slot.turn else -float(val)) if val is not None else 0.0
            slot.done = True
            return

        if slot.turn == slot.mcts_is_blue:
            return   # MCTS's turn - stop

        legal = np.where(action_masks(slot.board, slot.turn))[0]
        if len(legal) == 0:
            slot.turn = not slot.turn
            slot.move_count += 1
        else:
            action = find_best_move(slot.board.tobytes(), mm_depth, slot.turn)
            if action in (-1, 1225):
                slot.turn = not slot.turn
                slot.move_count += 1
            else:
                slot.board = apply_move(slot.board, action, slot.turn)
                slot.turn  = not slot.turn
                slot.move_count += 1

        if slot.move_count > 200:
            blue, green = count_cells(slot.board)
            slot.outcome = float(blue - green) / (blue + green) if blue + green else 0.0
            slot.done = True
            return


def _mmdata_apply_mcts(slot: _MMDataSlot) -> None:
    """Read completed search result, record example, apply MCTS move."""
    action_probs = slot.search.result
    slot.search  = None

    if not np.any(action_probs):
        is_t, val = check_terminal(slot.board, slot.turn)
        if is_t:
            slot.outcome = (float(val) if slot.turn else -float(val)) if val is not None else 0.0
            slot.done = True
        return

    slot.history.append((board_to_obs(slot.board, slot.turn), action_probs, slot.turn))
    action     = slot.mcts.select_action(action_probs, temperature=1)
    slot.board = apply_move(slot.board, action, slot.turn)
    slot.turn  = not slot.turn
    slot.move_count += 1
    if slot.move_count > 200:
        blue, green = count_cells(slot.board)
        slot.outcome = float(blue - green) / (blue + green) if blue + green else 0.0
        slot.done = True


def _mmdata_collect_restart(
    slot: _MMDataSlot,
    all_examples: list,
    games_started: int,
    num_games: int,
    mm_depth: int,
    dest: list,
    pbar=None,
) -> int:
    """Collect examples from a finished slot, restart if games remain."""
    while slot.done:
        blue, green = count_cells(slot.board)
        margin_blue = float(blue - green) / 49.0
        for obs, policy, was_blue in slot.history:
            all_examples.append((obs, policy,
                                 slot.outcome if was_blue else -slot.outcome,
                                 margin_blue if was_blue else -margin_blue))
        if pbar is not None:
            pbar.update(1)
        if games_started >= num_games:
            break
        _mmdata_reset(slot, games_started)
        games_started += 1
        _mmdata_advance(slot, mm_depth)
    if not slot.done:
        dest.append(slot)
    return games_started


def generate_mm_mix_data(
    mcts_ref: MCGS,
    num_games: int,
    mm_depth: int = 3,
    pool_size: int = 32,
    n_workers: int = 16,
) -> list:
    """
    Generate training examples from MCTS vs MM games using a concurrent pool.

    MM turns are advanced via threads (find_best_move releases the GIL);
    MCTS turns are batched into a single GPU forward pass per step.
    ~10x faster than serial play.

    Half the games are MCTS-as-Blue, half MCTS-as-Green.  Only MCTS-side
    positions are recorded as training examples.
    """
    instances = [
        MCGS(mcts_ref.network,
             num_simulations=mcts_ref.num_simulations,
             c_puct=mcts_ref.c_puct,
             gumbel_k=mcts_ref.gumbel_k)
        for _ in range(pool_size)
    ]
    slots = [_MMDataSlot(inst, i) for i, inst in enumerate(instances)]

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(lambda s: _mmdata_advance(s, mm_depth), slots))

    games_started  = pool_size
    all_examples: list = []

    pbar = tqdm(total=num_games, desc="MM-mix", unit="game")
    try:
        active: list[_MMDataSlot] = []
        for slot in slots:
            if slot.done:
                games_started = _mmdata_collect_restart(
                    slot, all_examples, games_started, num_games, mm_depth, active, pbar,
                )
            else:
                slot.search = slot.mcts.start_search(slot.board, slot.turn)
                active.append(slot)

        pending_mm: list[_MMDataSlot] = []

        while active or pending_mm:
            # Advance MM turns for pending slots (threaded)
            slots_for_mm = pending_mm[:]
            pending_mm   = []
            if slots_for_mm:
                with ThreadPoolExecutor(max_workers=min(n_workers, len(slots_for_mm))) as ex:
                    list(ex.map(lambda s: _mmdata_advance(s, mm_depth), slots_for_mm))

            # Batch GPU expand for all active searches
            if active:
                mcts_ref._expand_batch([s.search for s in active])

            # Step all active searches
            just_done:       list[_MMDataSlot] = []
            still_searching: list[_MMDataSlot] = []
            for slot in active:
                slot.search.step()
                if slot.search.done:
                    just_done.append(slot)
                else:
                    still_searching.append(slot)

            # Process MM results -> start new searches
            for slot in slots_for_mm:
                if slot.done:
                    games_started = _mmdata_collect_restart(
                        slot, all_examples, games_started, num_games, mm_depth, [], pbar,
                    )
                if not slot.done:
                    slot.search = slot.mcts.start_search(slot.board, slot.turn)
                    still_searching.append(slot)

            # Apply MCTS results -> queue for MM advancement next iteration
            for slot in just_done:
                _mmdata_apply_mcts(slot)
                if slot.done:
                    games_started = _mmdata_collect_restart(
                        slot, all_examples, games_started, num_games, mm_depth, pending_mm, pbar,
                    )
                else:
                    pending_mm.append(slot)

            active = still_searching
    finally:
        pbar.close()

    mcts_winning = sum(1 for e in all_examples if e[2] > 0)
    mcts_losing  = sum(1 for e in all_examples if e[2] < 0)
    print(f"  MM-mix ({mm_depth}): {num_games} games -> {len(all_examples)} examples"
          f"  | MCTS winning pos {mcts_winning} / losing pos {mcts_losing}")
    return all_examples
