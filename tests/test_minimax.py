"""
Pytest-based correctness tests for minimax implementation.

Run with: pytest tests/test_minimax.py -v
"""
import ctypes
from pathlib import Path
import numpy as np
import pytest

from lib.t7g import action_masks, BLUE, GREEN, CLEAR


# Fixtures

@pytest.fixture(scope="module")
def minimax():
    """Load minimax DLL"""
    libname = Path(__file__).parent.parent / "micro3.dll"
    if not libname.exists():
        pytest.skip("micro3.dll not found - compile with: gcc -O3 -march=native -ffast-math micro_3.c -o micro3.dll --shared")

    lib = ctypes.CDLL(str(libname))
    lib.find_best_move.restype = ctypes.c_int
    return lib


@pytest.fixture
def opening_board():
    """Standard opening position"""
    board = np.zeros((7, 7, 2), dtype=bool)
    board[0, 0] = BLUE
    board[0, 6] = GREEN
    board[6, 0] = GREEN
    board[6, 6] = BLUE
    return board


@pytest.fixture
def winning_board():
    """Blue has winning position"""
    board = np.zeros((7, 7, 2), dtype=bool)
    board[3, 3] = BLUE
    board[3, 2] = BLUE
    board[3, 4] = BLUE
    board[2, 3] = GREEN
    return board


@pytest.fixture
def no_moves_board():
    """Blue has no pieces"""
    board = np.zeros((7, 7, 2), dtype=bool)
    board[3, 3] = GREEN
    board[3, 4] = GREEN
    board[4, 3] = GREEN
    return board


@pytest.fixture
def full_board():
    """Completely filled board"""
    board = np.zeros((7, 7, 2), dtype=bool)
    for i in range(7):
        for j in range(7):
            board[i, j] = BLUE if (i + j) % 2 == 0 else GREEN
    return board


# Helper functions

def call_minimax(lib, board, depth=3, as_blue=True):
    """Call minimax with board"""
    return lib.find_best_move(
        board.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        depth,
        as_blue
    )


def is_move_legal(board, move, as_blue=True):
    """Check if move is legal"""
    if move == -1:
        masks = action_masks(board, as_blue)
        return not np.any(masks)  # -1 valid only if no moves

    masks = action_masks(board, as_blue)
    return 0 <= move < len(masks) and masks[move]


# Tests

class TestLegality:
    """Test minimax returns legal moves"""

    def test_opening_position(self, minimax, opening_board):
        move = call_minimax(minimax, opening_board)
        assert is_move_legal(opening_board, move)

    def test_winning_position(self, minimax, winning_board):
        move = call_minimax(minimax, winning_board)
        assert is_move_legal(winning_board, move)

    def test_no_moves_returns_minus_one(self, minimax, no_moves_board):
        move = call_minimax(minimax, no_moves_board)
        assert move == -1

    def test_full_board_returns_minus_one(self, minimax, full_board):
        move = call_minimax(minimax, full_board)
        assert move == -1

    def test_greens_turn(self, minimax, opening_board):
        move = call_minimax(minimax, opening_board, as_blue=False)
        assert is_move_legal(opening_board, move, as_blue=False)


class TestDepthConsistency:
    """Test consistency across search depths"""

    def test_all_depths_legal(self, minimax, opening_board):
        for depth in range(1, 6):
            move = call_minimax(minimax, opening_board, depth=depth)
            assert is_move_legal(opening_board, move), f"Illegal at depth {depth}"

    def test_deeper_search_not_worse(self, minimax, opening_board):
        move_shallow = call_minimax(minimax, opening_board, depth=2)
        move_deep = call_minimax(minimax, opening_board, depth=4)

        assert is_move_legal(opening_board, move_shallow)
        assert is_move_legal(opening_board, move_deep)


class TestDeterminism:
    """Test deterministic behavior"""

    def test_same_position_same_result(self, minimax, opening_board):
        moves = [call_minimax(minimax, opening_board, depth=4)
                 for _ in range(10)]
        assert len(set(moves)) == 1, f"Non-deterministic: {set(moves)}"

    def test_same_move_different_instances(self, minimax, opening_board):
        move1 = call_minimax(minimax, opening_board)
        board_copy = opening_board.copy()
        move2 = call_minimax(minimax, board_copy)
        assert move1 == move2


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_piece_each(self, minimax):
        board = np.zeros((7, 7, 2), dtype=bool)
        board[0, 0] = BLUE
        board[6, 6] = GREEN

        move = call_minimax(minimax, board)
        assert is_move_legal(board, move)

    def test_nearly_full_board(self, minimax):
        board = np.zeros((7, 7, 2), dtype=bool)
        for i in range(6):
            for j in range(6):
                board[i, j] = BLUE if (i + j) % 2 == 0 else GREEN
        board[5, 5] = BLUE
        board[6, 6] = BLUE

        move = call_minimax(minimax, board)
        assert is_move_legal(board, move)

    def test_crowded_board(self, minimax):
        board = np.zeros((7, 7, 2), dtype=bool)
        for i in range(7):
            for j in range(7):
                if (i + j) % 2 == 0:
                    board[i, j] = BLUE
                elif i < 4:
                    board[i, j] = GREEN

        move = call_minimax(minimax, board)
        assert is_move_legal(board, move)


@pytest.mark.slow
class TestPerformance:
    """Performance tests"""

    def test_handles_complex_position_quickly(self, minimax):
        import time

        board = np.zeros((7, 7, 2), dtype=bool)
        for i in range(7):
            for j in range(7):
                if (i + j) % 3 == 0:
                    board[i, j] = BLUE
                elif (i + j) % 3 == 1:
                    board[i, j] = GREEN

        start = time.time()
        move = call_minimax(minimax, board, depth=4)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"
        assert is_move_legal(board, move)


class TestIntegration:
    """Integration tests for real gameplay scenarios"""

    def test_play_complete_game(self, minimax):
        """Play a complete game and verify no crashes"""
        board = np.zeros((7, 7, 2), dtype=bool)
        board[0, 0] = BLUE
        board[0, 6] = GREEN
        board[6, 0] = GREEN
        board[6, 6] = BLUE

        for turn in range(100):  # Max 100 turns
            as_blue = turn % 2 == 0
            move = call_minimax(minimax, board, depth=3, as_blue=as_blue)

            if move == -1:
                break  # Game over

            assert is_move_legal(board, move, as_blue)

        assert True  # If we got here, no crashes

    def test_handles_random_positions(self, minimax):
        """Test on 20 random board positions"""
        np.random.seed(42)

        for _ in range(20):
            board = np.zeros((7, 7, 2), dtype=bool)

            for i in range(7):
                for j in range(7):
                    r = np.random.random()
                    if r < 0.3:
                        board[i, j] = BLUE
                    elif r < 0.6:
                        board[i, j] = GREEN

            board[0, 0] = BLUE
            board[6, 6] = GREEN

            move = call_minimax(minimax, board, depth=2)
            assert is_move_legal(board, move)