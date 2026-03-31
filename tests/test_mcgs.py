"""
Correctness tests for the MCGS search engine (micro_mcts.c via ctypes).

These tests specifically document and enforce the invariants that, when broken,
cause silent training corruption rather than crashes:

  - Transposition table is fully cleared between games (the heatmap bug)
  - No state bleeds across the game boundary in the pool
  - Policy output respects the legal-move mask
  - Value perspective is consistent with the observation encoding

Run with: .venv/Scripts/python -m pytest tests/test_mcgs.py -v
"""
import numpy as np
import pytest
import torch

from lib.dual_network import DualHeadNetwork
from lib.mcgs import MCGS
from lib.t7g import (
    new_board, apply_move, action_masks, check_terminal,
    board_to_obs, BLUE, GREEN,
)
from lib.train_workers import _GameSlot, _reset_slot


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_board(blue_positions, green_positions):
    board = np.zeros((7, 7, 2), dtype=bool)
    for x, y in blue_positions:
        board[y, x] = BLUE
    for x, y in green_positions:
        board[y, x] = GREEN
    return board


def run_search(mcts, board, turn):
    """Run a complete search and return the policy distribution."""
    return mcts.search(board, turn)


def advance_n_moves(mcts, board, turn, n):
    """Play n moves using argmax policy, return final (board, turn)."""
    for _ in range(n):
        is_terminal, _ = check_terminal(board, turn)
        if is_terminal:
            break
        masks = action_masks(board, turn)
        if not np.any(masks):
            turn = not turn
            continue
        probs = run_search(mcts, board, turn)
        action = int(np.argmax(probs)) if np.any(probs) else int(np.argmax(masks))
        board = apply_move(board, action, turn)
        turn = not turn
    return board, turn


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def network():
    """Randomly-initialised network on CPU — sufficient for structural tests."""
    torch.manual_seed(42)
    net = DualHeadNetwork(num_actions=1225).to("cpu")
    net.eval()
    return net


@pytest.fixture
def mcts(network):
    """Fresh MCGS instance for each test (16 sims — fast but exercises all phases)."""
    return MCGS(network, num_simulations=16, c_puct=0.75, gumbel_k=8)


# ── Transposition table invariants ───────────────────────────────────────────

def test_tt_empty_at_creation(mcts):
    assert len(mcts.transposition_table) == 0


def test_tt_grows_during_search(mcts):
    probs = run_search(mcts, new_board(), True)
    assert len(mcts.transposition_table) > 0
    assert np.any(probs)


def test_clear_empties_tt(mcts):
    run_search(mcts, new_board(), True)
    assert len(mcts.transposition_table) > 0

    mcts.clear()

    assert len(mcts.transposition_table) == 0, (
        "TT must be completely empty after clear — any leftover nodes "
        "from a previous game corrupt the next game's search statistics"
    )


def test_tt_rebuilds_from_zero_after_clear(mcts):
    """After a clear, the first search must build the TT from scratch, not reuse stale nodes."""
    board = new_board()

    run_search(mcts, board, True)
    size_before_clear = len(mcts.transposition_table)

    mcts.clear()
    assert len(mcts.transposition_table) == 0

    run_search(mcts, board, True)
    size_after_clear = len(mcts.transposition_table)

    # With 16 simulations from a fresh TT we expect far fewer nodes than the
    # accumulated count from before the clear.
    assert size_after_clear <= size_before_clear, (
        "Post-clear search produced more nodes than the pre-clear search, "
        "suggesting stale nodes were retained"
    )
    assert size_after_clear > 0


# ── Game independence (the heatmap bug) ──────────────────────────────────────

def test_game_independence_tt_cleared_between_games(mcts):
    """
    THE critical invariant: game A's transposition table must not contaminate game B.

    Before the arena-allocator fix, mcgs_clear was not actually freeing nodes,
    so every game accumulated on top of all previous games.  The search would
    find positions from completely different games in the TT and reuse their
    visit counts and Q-values — turning the policy signal into a heatmap of
    popular positions across all games rather than the current game's search.
    """
    board = new_board()

    # Game A: play several moves and accumulate a large TT.
    _, _ = advance_n_moves(mcts, board.copy(), True, n=15)
    tt_after_game_a = len(mcts.transposition_table)
    assert tt_after_game_a > 0, "Game A should have populated the TT"

    # Between-game reset (mirrors _reset_slot).
    mcts.clear()

    assert len(mcts.transposition_table) == 0, (
        "TT must be zero after clear — game B must not inherit game A's nodes"
    )

    # Game B: first search sees only freshly-created nodes.
    run_search(mcts, board, True)
    tt_start_game_b = len(mcts.transposition_table)

    assert tt_start_game_b > 0, "Game B search should create nodes"
    assert tt_start_game_b < tt_after_game_a, (
        "Game B TT after one search must be smaller than game A's full "
        "accumulated TT — if equal, clearing did nothing"
    )


def test_repeated_game_cycles_stay_clean(mcts):
    """Running N games back-to-back, each with a clear, must not accumulate state."""
    board = new_board()
    sizes = []

    for _ in range(4):
        mcts.clear()
        assert len(mcts.transposition_table) == 0
        run_search(mcts, board, True)
        sizes.append(len(mcts.transposition_table))

    # Each game starts fresh so TT size after one search should be consistent
    # (within Gumbel noise).  A monotonically growing sequence would indicate
    # that clear isn't working.
    assert max(sizes) < sizes[0] * 3, (
        f"TT sizes across game cycles are growing monotonically: {sizes} — "
        "clear may not be resetting state"
    )


# ── Pool slot boundary ────────────────────────────────────────────────────────

def test_reset_slot_clears_mcts_state(network):
    """_reset_slot must clear the MCTS TT — this is the pool's game boundary."""
    slot = _GameSlot(MCGS(network, num_simulations=16, c_puct=0.75, gumbel_k=8))

    # Simulate a game having run: populate the TT.
    slot.mcts.search(slot.board, slot.turn)
    assert len(slot.mcts.transposition_table) > 0

    _reset_slot(slot)

    assert len(slot.mcts.transposition_table) == 0, (
        "_reset_slot must call mcts.clear() — without this every game in "
        "the pool inherits the previous game's search tree"
    )
    assert slot.move_count == 0
    assert slot.examples == []
    assert slot.board_history == {}


# ── Policy invariants ─────────────────────────────────────────────────────────

def test_policy_sums_to_one(mcts):
    probs = run_search(mcts, new_board(), True)
    assert abs(probs.sum() - 1.0) < 1e-5, f"Policy sum = {probs.sum():.6f}, expected 1.0"


def test_policy_zero_on_illegal_moves(mcts):
    """Moves that are not legal must have exactly zero probability."""
    board = new_board()
    turn = True
    probs = run_search(mcts, board, turn)
    legal = action_masks(board, turn)

    illegal_mass = probs[~legal].sum()
    assert illegal_mass == 0.0, (
        f"Policy placed {illegal_mass:.4f} mass on illegal moves"
    )


def test_policy_only_on_legal_moves(mcts):
    """At least one legal move must receive non-zero probability."""
    board = new_board()
    probs = run_search(mcts, board, True)
    legal = action_masks(board, True)
    assert probs[legal].sum() > 0.0


def test_terminal_position_returns_zero_policy(mcts):
    """A terminal position must return an all-zero policy (no move to make)."""
    # Blue has been eliminated — terminal, Green wins.
    board = make_board(blue_positions=[], green_positions=[(3, 3), (3, 4)])
    is_terminal, _ = check_terminal(board, True)
    assert is_terminal, "Test setup error: board should be terminal"

    probs = run_search(mcts, board, True)
    assert probs.sum() == 0.0, "Terminal position must return zero policy"


# ── Value perspective consistency ─────────────────────────────────────────────

def test_observation_encoding_is_current_player_relative():
    """
    board_to_obs must be current-player-relative: channel 1 = my pieces,
    channel 0 = opponent pieces, regardless of colour.

    The value head receives this observation and must output a value from the
    current player's perspective.  If the encoding were Blue-absolute, the
    value head would need to learn two different representations for the same
    positional concept depending on which colour is playing.
    """
    board = make_board(blue_positions=[(0, 0), (1, 0)], green_positions=[(6, 6)])

    obs_blue = board_to_obs(board, turn=True)   # Blue to move
    obs_green = board_to_obs(board, turn=False)  # Green to move

    # Blue's turn: channel 1 = Blue pieces, channel 0 = Green pieces
    assert obs_blue[:, :, 1].sum() == 2.0, "Blue pieces should be in channel 1 for Blue's turn"
    assert obs_blue[:, :, 0].sum() == 1.0, "Green pieces should be in channel 0 for Blue's turn"

    # Green's turn: channel 1 = Green pieces (mine), channel 0 = Blue pieces (opponent)
    assert obs_green[:, :, 1].sum() == 1.0, "Green pieces should be in channel 1 for Green's turn"
    assert obs_green[:, :, 0].sum() == 2.0, "Blue pieces should be in channel 0 for Green's turn"


def test_value_targets_are_current_player_relative():
    """
    Value targets stored in training examples must match the observation
    perspective: positive = current player is winning.

    This is the convention used by board_to_obs, the MCTS backprop, and the
    training loop.  A sign error here would make half the training data
    point the value head in the wrong direction.
    """
    from lib.train_workers import _slot_result, _GameSlot
    from lib.dual_network import DualHeadNetwork

    torch.manual_seed(0)
    net = DualHeadNetwork(num_actions=1225).to("cpu")
    slot = _GameSlot(MCGS(net, num_simulations=16))

    # Manually craft one example: Blue's turn, Blue wins (+1 from Blue's perspective).
    board = new_board()
    obs = board_to_obs(board, turn=True)
    slot.examples = [(obs, np.ones(1225) / 1225, True, board.copy())]

    # winner=+1.0 means Blue won.
    examples, _, _, _, _, _ = _slot_result(slot, winner=1.0)
    obs_out, policy_out, value_out, _, _ = examples[0]

    assert value_out == 1.0, (
        "Blue-turn example with Blue winning must have value_target=+1.0"
    )

    # Same board, Green's turn, Blue still wins (winner=+1 from Blue).
    slot.examples = [(board_to_obs(board, turn=False), np.ones(1225) / 1225, False, board.copy())]
    examples, _, _, _, _, _ = _slot_result(slot, winner=1.0)
    _, _, value_out_green, _, _ = examples[0]

    assert value_out_green == -1.0, (
        "Green-turn example with Blue winning must have value_target=-1.0 "
        "(Green is losing — current-player-relative)"
    )
