"""
Edge-case tests for find_best_move covering positions that have historically
caused incorrect results (-1 returns, hangs, or wrong moves).

Each test has a known correct answer or a known constraint on the answer.
"""
import numpy as np
import pytest

from lib.t7g import (
    find_best_move, apply_move, action_masks, is_action_valid,
    count_cells, BLUE, GREEN,
)


def board_from(blue_positions, green_positions):
    b = np.zeros((7, 7, 2), dtype=bool)
    for x, y in blue_positions:
        b[y, x] = BLUE
    for x, y in green_positions:
        b[y, x] = GREEN
    return b


# ---------------------------------------------------------------------------
# Helper: assert find_best_move returns a valid move (not -1) for every depth
# ---------------------------------------------------------------------------

def assert_valid_at_depths(board, as_blue, depths=(1, 2, 3, 5), label=""):
    for depth in depths:
        move = find_best_move(board.tobytes(), depth=depth, as_blue=as_blue)
        player = "Blue" if as_blue else "Green"
        assert move >= 0, (
            f"{label}: {player} depth={depth} returned {move}; "
            f"board has {np.sum(board[:,:,1])} blue, {np.sum(board[:,:,0])} green"
        )
        assert is_action_valid(board, move, as_blue), \
            f"{label}: {player} depth={depth} move {move} is not valid"


# ---------------------------------------------------------------------------
# 1. Proven-loss positions — loser must still return a legal move
# ---------------------------------------------------------------------------

class TestProvenLoss:
    def test_blue_1v4(self):
        """Blue 1 piece, Green 4 — proven loss at depth>=2, must still move."""
        board = board_from(
            blue_positions=[(0, 0)],
            green_positions=[(2, 0), (0, 2), (3, 3), (6, 6)],
        )
        assert_valid_at_depths(board, as_blue=True, label="Blue 1v4")

    def test_green_1v4(self):
        """Green 1 piece, Blue 4 — proven loss for Green."""
        board = board_from(
            blue_positions=[(0, 0), (3, 3), (4, 2), (6, 6)],
            green_positions=[(6, 0)],
        )
        assert_valid_at_depths(board, as_blue=False, label="Green 1v4")

    def test_blue_1v6(self):
        """Blue 1 piece, Green 6 — deep proven loss."""
        board = board_from(
            blue_positions=[(3, 3)],
            green_positions=[(0, 0), (6, 0), (0, 6), (6, 6), (3, 0), (0, 3)],
        )
        assert_valid_at_depths(board, as_blue=True, label="Blue 1v6")

    def test_green_1v6(self):
        board = board_from(
            blue_positions=[(0, 0), (6, 0), (0, 6), (6, 6), (3, 0), (0, 3)],
            green_positions=[(3, 3)],
        )
        assert_valid_at_depths(board, as_blue=False, label="Green 1v6")

    def test_blue_cornered_1v3(self):
        """Blue piece in corner, surrounded on two sides."""
        board = board_from(
            blue_positions=[(6, 6)],
            green_positions=[(4, 6), (6, 4), (4, 4)],
        )
        assert_valid_at_depths(board, as_blue=True, label="Blue cornered 1v3")


# ---------------------------------------------------------------------------
# 2. Edge / corner pieces — regression for bounds clamping bugs
# ---------------------------------------------------------------------------

class TestEdgePieces:
    @pytest.mark.parametrize("bx,by,gx,gy", [
        (5, 3, 6, 3),   # right edge clone
        (3, 5, 3, 6),   # bottom edge clone
        (5, 5, 6, 5),   # bottom-right diagonal
        (5, 5, 5, 6),   # bottom-right diagonal (other axis)
        (1, 3, 0, 3),   # left edge clone
        (3, 1, 3, 0),   # top edge clone
        (6, 3, 5, 3),   # piece already at right edge, Green adjacent
        (3, 6, 3, 5),   # piece already at bottom edge, Green adjacent
    ])
    def test_edge_move_captures(self, bx, by, gx, gy):
        """Blue adjacent to Green near an edge — must find a move and capture."""
        board = board_from(blue_positions=[(bx, by)], green_positions=[(gx, gy)])
        move = find_best_move(board.tobytes(), depth=1, as_blue=True)
        assert move >= 0, f"Blue({bx},{by}) vs Green({gx},{gy}): returned -1"
        new_board = apply_move(board, move, as_blue=True)
        _, green_count = count_cells(new_board)
        assert green_count == 0, \
            f"Blue({bx},{by}) vs Green({gx},{gy}): Green not captured (move={move})"


# ---------------------------------------------------------------------------
# 3. Immediate winning move — must be taken at depth >= 1
# ---------------------------------------------------------------------------

class TestImmediateWin:
    def test_blue_captures_last_green(self):
        """Blue adjacent to the only Green piece — must capture."""
        board = board_from(
            blue_positions=[(3, 3), (2, 2), (4, 4)],
            green_positions=[(3, 4)],
        )
        move = find_best_move(board.tobytes(), depth=1, as_blue=True)
        assert move >= 0
        new_board = apply_move(board, move, as_blue=True)
        _, green_count = count_cells(new_board)
        assert green_count == 0, f"Should capture last green, move={move}"

    def test_green_captures_last_blue(self):
        board = board_from(
            blue_positions=[(3, 4)],
            green_positions=[(3, 3), (2, 2), (4, 4)],
        )
        move = find_best_move(board.tobytes(), depth=1, as_blue=False)
        assert move >= 0
        new_board = apply_move(board, move, as_blue=False)
        blue_count, _ = count_cells(new_board)
        assert blue_count == 0, f"Should capture last blue, move={move}"


# ---------------------------------------------------------------------------
# 4. Asymmetric piece counts — various ratios
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blue_n,green_n,as_blue", [
    (1, 1, True),
    (1, 1, False),
    (2, 1, True),
    (1, 2, False),
    (3, 1, True),
    (1, 3, False),
    (4, 1, True),
    (1, 4, False),
    (6, 1, True),
    (1, 6, False),
    (3, 3, True),
    (3, 3, False),
])
def test_piece_ratio(blue_n, green_n, as_blue):
    """find_best_move returns valid move for various piece ratios at all depths."""
    rng = np.random.default_rng(blue_n * 100 + green_n * 10 + int(as_blue))
    # Place pieces randomly but without overlap
    cells = list(rng.choice(49, blue_n + green_n, replace=False))
    board = np.zeros((7, 7, 2), dtype=bool)
    for i, c in enumerate(cells[:blue_n]):
        board[c // 7, c % 7] = BLUE
    for i, c in enumerate(cells[blue_n:]):
        board[c // 7, c % 7] = GREEN

    # Skip if the mover has no legal moves (valid edge case)
    legal = np.where(action_masks(board, as_blue))[0]
    if len(legal) == 0:
        pytest.skip("No legal moves for mover in this random position")

    assert_valid_at_depths(board, as_blue,
                           label=f"blue={blue_n} green={green_n} as_blue={as_blue}")


# ---------------------------------------------------------------------------
# 5. Determinism — same board, same depth, same result every time
# ---------------------------------------------------------------------------

def test_determinism_blue():
    board = board_from(
        blue_positions=[(1, 1), (2, 2)],
        green_positions=[(4, 4), (5, 5)],
    )
    results = [find_best_move(board.tobytes(), depth=3, as_blue=True) for _ in range(5)]
    assert len(set(results)) == 1, f"Non-deterministic: {results}"


def test_determinism_green():
    board = board_from(
        blue_positions=[(4, 4), (5, 5)],
        green_positions=[(1, 1), (2, 2)],
    )
    results = [find_best_move(board.tobytes(), depth=3, as_blue=False) for _ in range(5)]
    assert len(set(results)) == 1, f"Non-deterministic: {results}"


# ---------------------------------------------------------------------------
# 6. Near-full board — very few empty squares
# ---------------------------------------------------------------------------

def test_near_full_board_blue():
    """Near-full board with only a few empty squares."""
    board = np.zeros((7, 7, 2), dtype=bool)
    # Fill most of the board alternating blue/green, leave a few gaps
    for y in range(7):
        for x in range(7):
            if (x + y) % 7 == 0:
                continue  # leave empty
            if (x + y) % 2 == 0:
                board[y, x] = BLUE
            else:
                board[y, x] = GREEN

    legal = np.where(action_masks(board, True))[0]
    if len(legal) == 0:
        pytest.skip("No legal moves")
    assert_valid_at_depths(board, as_blue=True, depths=(1, 2), label="near-full blue")


def test_near_full_board_green():
    board = np.zeros((7, 7, 2), dtype=bool)
    for y in range(7):
        for x in range(7):
            if (x + y) % 7 == 0:
                continue
            if (x + y) % 2 == 0:
                board[y, x] = BLUE
            else:
                board[y, x] = GREEN

    legal = np.where(action_masks(board, False))[0]
    if len(legal) == 0:
        pytest.skip("No legal moves")
    assert_valid_at_depths(board, as_blue=False, depths=(1, 2), label="near-full green")
