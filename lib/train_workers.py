"""
Game-logic helpers for AlphaZero self-play training.

Stateless (no module-level globals) to allow multiprocessing.
"""
import time
from concurrent.futures import Future

import numpy as np

from lib.mcgs import MCGS, MCGSSearch
from lib.t7g import (
    new_board, apply_move, check_terminal,
    board_to_obs, action_masks, count_cells, find_best_move,
)


# ---------------------------------------------------------------------------
# Self-play (single game — used for eval and as fallback)
# ---------------------------------------------------------------------------

def self_play_game(mcts: MCGS):
    """
    Play one game via MCTS self-play, collecting training examples.

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
            blue, green = count_cells(board)
            winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
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
        obs = board_to_obs(board, turn)
        examples.append((obs, action_probs, turn, board.copy()))

        action = mcts.select_action(action_probs, board=board, turn=turn, temperature=0.0)

        board = apply_move(board, action, turn)
        mcts.advance_tree(action)
        turn = not turn
        move_count += 1

        if move_count > 200:
            blue, green = count_cells(board)
            winner = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
            truncated = True
            break

    training_examples = []
    for obs, policy_target, example_turn, _ in examples:
        value_target = winner if example_turn else -winner
        training_examples.append((obs, policy_target, value_target))

    elapsed = time.time() - game_start
    return training_examples, winner, move_count, elapsed, truncated, legal_move_counts


# ---------------------------------------------------------------------------
# In-process game pool — batched inference across concurrent games
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


def _slot_result(slot: _GameSlot, winner: float) -> tuple:
    """Package a finished slot into a result tuple matching self_play_game_pool's yield contract.

    Returns
    -------
    training_examples : list of (obs, raw_policy, value_target, board, turn)
        5-tuples — board and turn included so the caller can apply policy
        relabeling outside the GPU-critical pool loop.
    winner            : +1.0 Blue / −1.0 Green / material ratio for truncated games
    move_count        : number of half-moves played
    elapsed           : wall time in seconds
    truncated         : True if the 200-move cap triggered
    legal_move_counts : per-position branching factor samples
    """
    examples = []
    for obs, policy_target, example_turn, ex_board in slot.examples:
        value_target = winner if example_turn else -winner
        examples.append((obs, policy_target, value_target, ex_board, example_turn))
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


def self_play_game_pool(
    mcts: MCGS,
    pool_size: int,
    target_games: int,
    mcts_pool: 'list[MCGS] | None' = None,
):
    """
    Play target_games games concurrently with batched network inference.

    Each slot has its own MCGS instance (isolated transposition table).
    All pending leaf nodes across every active search are expanded in a single
    forward pass per step, giving batch_size ≈ pool_size × K_eff.

    Slots are immediately restarted when a game finishes, keeping all pool_size
    slots active throughout (no draining at the tail).

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

    active = list(slots)
    games_started = pool_size

    while active:
        # Single batched forward pass for all pending leaves across every search.
        # Pass search objects directly — _expand_batch fetches boards and commits
        # results with one C call per slot instead of one per leaf.
        mcts._expand_batch([slot.search for slot in active])

        next_active = []
        for slot in active:
            slot.search.step()  # type: ignore[union-attr]

            if not slot.search.done:
                next_active.append(slot)
                continue

            action_probs = slot.search.result
            if not np.any(action_probs):
                is_terminal, terminal_value = check_terminal(slot.board, slot.turn)
                if is_terminal:
                    assert terminal_value is not None
                    winner = terminal_value if slot.turn else -terminal_value
                    yield _slot_result(slot, winner)
                    if games_started < target_games:
                        _reset_slot(slot)
                        slot.search = slot.mcts.start_search(slot.board, slot.turn, None)
                        next_active.append(slot)
                        games_started += 1
                    continue

                # Distinguish genuine forced-pass from spurious all-zero.
                # Spurious all-zero can occur when the C slab overflows:
                # mcgs_start_search returns NULL → MCGSSearch(0) reports done=True
                # with all-zero result even though the position has legal moves.
                if np.any(action_masks(slot.board, slot.turn)):
                    # Spurious all-zero: player has legal moves but MCTS returned nothing.
                    # Recover with a uniform distribution over legal moves so the game
                    # can continue.  Skip adding this position as a training example
                    # since the policy label would be meaningless.
                    masks = action_masks(slot.board, slot.turn)
                    action_probs = masks.astype(np.float32)
                    action_probs /= action_probs.sum()
                    # Fall through to normal action-selection below (no continue).
                else:
                    # Genuine forced pass: current player has no legal moves.
                    slot.turn = not slot.turn
                    slot.move_count += 1
                    if slot.move_count > 200:
                        blue, green = count_cells(slot.board)
                        winner = (float(blue - green) / float(blue + green)
                                  if blue + green > 0 else 0.0)
                        slot.truncated = True
                        yield _slot_result(slot, winner)
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
            slot.examples.append((obs, action_probs, slot.turn, slot.board.copy()))

            action = slot.mcts.select_action(
                action_probs, board=slot.board, turn=slot.turn, temperature=0.0,
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

        active = next_active


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
    """Kick off the next move — MCTS search or threaded MM call."""
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
    MM turns run via a thread pool — ctypes releases the GIL so they
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

                # ── MCTS turn ──────────────────────────────────────────────
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

                # ── MM turn ────────────────────────────────────────────────
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

                # ── After any move: check termination ──────────────────────
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
) -> float:
    """
    Play one evaluation game (MCTS vs minimax/stauf).

    Returns a float in [−1, +1] from the MCTS agent's perspective.
    Decisive terminal positions give ±1.0; draws by repetition or truncation
    give a material ratio: (blue − green) / (blue + green).
    """
    board = new_board()
    mcts.clear()
    turn = True  # Blue moves first (eval games always start standard)
    board_history: dict = {}
    move_count = 0

    while True:
        state_key = board.tobytes() + bytes([turn])
        board_history[state_key] = board_history.get(state_key, 0) + 1
        if board_history[state_key] >= 3:
            blue, green = count_cells(board)
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
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
            break

    return blue_result if mcts_is_blue else -blue_result


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
    Decisive terminal positions give ±1.0; draws by repetition or truncation
    give a material ratio: (blue − green) / (blue + green).
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
            blue, green = count_cells(board)
            blue_result = float(blue - green) / float(blue + green) if blue + green > 0 else 0.0
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
