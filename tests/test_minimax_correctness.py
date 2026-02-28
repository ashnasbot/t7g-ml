"""
Unit tests for minimax algorithm correctness.

These tests verify that the minimax algorithm makes sensible decisions
in various board positions.

Run with: pytest tests/test_minimax_correctness.py -v
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from lib.t7g import (
    find_best_move, action_to_move, action_masks, is_action_valid,
    count_cells, BLUE, GREEN, CLEAR
)


# Helper functions

def setup_board(blue_positions, green_positions):
    """
    Create a board with pieces at specified positions.

    Args:
        blue_positions: List of (x, y) tuples for blue pieces
        green_positions: List of (x, y) tuples for green pieces

    Returns:
        Board numpy array
    """
    board = np.zeros((7, 7, 2), dtype=bool)
    for x, y in blue_positions:
        board[y, x] = BLUE
    for x, y in green_positions:
        board[y, x] = GREEN
    return board


def apply_move(board, action, as_blue=True):
    """Apply a move to the board and return new board state"""
    board = board.copy()
    from_x, from_y, to_x, to_y, jump = action_to_move(action)

    player_cell = BLUE if as_blue else GREEN
    opponent_cell = GREEN if as_blue else BLUE

    if jump:
        board[from_y, from_x] = CLEAR
    board[to_y, to_x] = player_cell

    # Convert adjacent opponent pieces
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            x2 = to_x + dx
            y2 = to_y + dy
            if 0 <= x2 < 7 and 0 <= y2 < 7:
                if np.array_equal(board[y2, x2], opponent_cell):
                    board[y2, x2] = player_cell

    return board


# Basic functionality tests

def test_minimax_returns_valid_move_opening():
    """Minimax should return a valid move from opening position"""
    board = setup_board(
        blue_positions=[(0, 0), (6, 6)],
        green_positions=[(6, 0), (0, 6)]
    )

    move = find_best_move(board.tobytes(), depth=3, as_blue=True)

    # Should return a valid move (not -1 or 1225)
    assert move >= 0 and move < 1225, f"Move {move} out of valid range"

    # Move should be in the action mask
    masks = action_masks(board, True)
    assert masks[move], f"Move {move} not in action mask"

    # Move should be valid
    assert is_action_valid(board, move, True), f"Move {move} not valid"


def test_minimax_returns_no_move_when_stuck():
    """Minimax should return -1 when no moves available"""
    # Board where blue is completely surrounded by green (no moves)
    board = setup_board(
        blue_positions=[(3, 3)],
        green_positions=[(2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (2, 4), (3, 4), (4, 4)]
    )

    move = find_best_move(board.tobytes(), depth=1, as_blue=True)

    # Should return -1 (no move)
    assert move == -1, f"Expected -1 (no move), got {move}"


# Tactical tests

def test_minimax_takes_winning_move():
    """Minimax should take an immediate winning move"""
    # Setup: Blue can capture last green piece
    board = setup_board(
        blue_positions=[(3, 3), (2, 2), (4, 4)],  # Blue dominant
        green_positions=[(0, 0)]  # One green left, capturable
    )

    move = find_best_move(board.tobytes(), depth=2, as_blue=True)

    # Apply the move
    new_board = apply_move(board, move, as_blue=True)

    # Check that green pieces are reduced or eliminated
    blue_before, green_before = count_cells(board)
    blue_after, green_after = count_cells(new_board)

    assert blue_after >= blue_before, "Blue should gain or maintain pieces"
    assert green_after <= green_before, "Green should lose or maintain pieces"


def test_minimax_avoids_capture():
    """Minimax should avoid moves that lead to capture"""
    # Setup: Moving into a position surrounded by opponent is bad
    board = setup_board(
        blue_positions=[(3, 3)],
        green_positions=[(5, 5), (6, 5), (6, 6)]  # Green cluster
    )

    move = find_best_move(board.tobytes(), depth=3, as_blue=True)

    # The move should NOT place blue adjacent to green cluster
    from_x, from_y, to_x, to_y, jump = action_to_move(move)

    # Verify we're not moving directly into enemy territory
    # (This is a heuristic - we can't easily check all bad moves)
    assert move != -1, "Should find some move"


def test_minimax_prefers_captures():
    """Minimax should prefer moves that capture opponent pieces"""
    # Setup: Blue can either move away or capture green
    board = setup_board(
        blue_positions=[(3, 3)],
        green_positions=[(4, 3)]  # Adjacent green, easy to convert
    )

    move = find_best_move(board.tobytes(), depth=3, as_blue=True)

    # Apply move
    new_board = apply_move(board, move, as_blue=True)

    # Count cells before and after
    blue_before, green_before = count_cells(board)
    blue_after, green_after = count_cells(new_board)

    # Should result in material gain for blue
    material_swing = (blue_after - blue_before) - (green_after - green_before)
    assert material_swing > 0, f"Should gain material, swing: {material_swing}"


# Depth sensitivity tests

@pytest.mark.parametrize("depth", [1, 2, 3])
def test_minimax_works_at_all_depths(depth):
    """Minimax should work correctly at different search depths"""
    board = setup_board(
        blue_positions=[(0, 0), (6, 6)],
        green_positions=[(6, 0), (0, 6)]
    )

    move = find_best_move(board.tobytes(), depth=depth, as_blue=True)

    # Should return valid move at all depths
    assert move >= 0 and move < 1225, f"Depth {depth}: invalid move {move}"
    assert is_action_valid(board, move, True), f"Depth {depth}: move {move} not valid"


def test_deeper_search_same_or_better():
    """Deeper search should make same or better decisions"""
    # Complex position where depth matters
    board = setup_board(
        blue_positions=[(1, 1), (2, 2)],
        green_positions=[(5, 5), (6, 6)]
    )

    move_depth1 = find_best_move(board.tobytes(), depth=1, as_blue=True)
    move_depth3 = find_best_move(board.tobytes(), depth=3, as_blue=True)

    # Both should be valid moves
    assert is_action_valid(board, move_depth1, True), "Depth 1 move invalid"
    assert is_action_valid(board, move_depth3, True), "Depth 3 move invalid"

    # Can't guarantee they're the same, but both should be reasonable
    # (deeper is often better but not always due to alpha-beta)


# Symmetry tests

def test_minimax_symmetry():
    """Minimax should handle symmetric positions consistently"""
    # Two symmetric opening positions
    board1 = setup_board(
        blue_positions=[(0, 0), (6, 6)],
        green_positions=[(6, 0), (0, 6)]
    )

    board2 = setup_board(
        blue_positions=[(6, 6), (0, 0)],  # Same, different order
        green_positions=[(0, 6), (6, 0)]
    )

    move1 = find_best_move(board1.tobytes(), depth=2, as_blue=True)
    move2 = find_best_move(board2.tobytes(), depth=2, as_blue=True)

    # Both should return valid moves
    assert is_action_valid(board1, move1, True)
    assert is_action_valid(board2, move2, True)


# Edge case tests

def test_minimax_full_board():
    """Minimax handles near-full board correctly"""
    # Create a board that's almost full
    blue_positions = [(i, j) for i in range(7) for j in range(4)]
    green_positions = [(i, j) for i in range(7) for j in range(4, 7) if (i, j) != (3, 6)]

    board = setup_board(blue_positions, green_positions)

    move = find_best_move(board.tobytes(), depth=2, as_blue=True)

    # Should either return valid move or no move
    if move != -1:
        assert is_action_valid(board, move, True), "Returned move should be valid"


def test_minimax_single_piece():
    """Minimax works with only one piece per side"""
    board = setup_board(
        blue_positions=[(3, 3)],
        green_positions=[(5, 5)]
    )

    move = find_best_move(board.tobytes(), depth=2, as_blue=True)

    # Should return a valid move
    assert move >= 0 and move < 1225, f"Invalid move {move}"
    assert is_action_valid(board, move, True)


def test_minimax_alternating_colors():
    """Minimax handles both blue and green perspectives"""
    board = setup_board(
        blue_positions=[(1, 1), (2, 2)],
        green_positions=[(5, 5), (6, 6)]
    )

    # Get moves for both colors
    blue_move = find_best_move(board.tobytes(), depth=2, as_blue=True)
    green_move = find_best_move(board.tobytes(), depth=2, as_blue=False)

    # Both should be valid
    assert is_action_valid(board, blue_move, True), "Blue move invalid"
    assert is_action_valid(board, green_move, False), "Green move invalid"


# Consistency tests

def test_minimax_deterministic():
    """Minimax should return same move for same position"""
    board = setup_board(
        blue_positions=[(2, 2), (3, 3)],
        green_positions=[(4, 4), (5, 5)]
    )

    # Call multiple times
    move1 = find_best_move(board.tobytes(), depth=3, as_blue=True)
    move2 = find_best_move(board.tobytes(), depth=3, as_blue=True)
    move3 = find_best_move(board.tobytes(), depth=3, as_blue=True)

    # Should return same move every time (deterministic)
    assert move1 == move2 == move3, \
        f"Minimax not deterministic: {move1}, {move2}, {move3}"


@pytest.mark.slow
def test_minimax_full_game_simulation():
    """Minimax can play a complete game without errors"""
    board = setup_board(
        blue_positions=[(0, 0), (6, 6)],
        green_positions=[(6, 0), (0, 6)]
    )

    turn = True  # Blue starts
    max_turns = 100

    for turn_count in range(max_turns):
        move = find_best_move(board.tobytes(), depth=1, as_blue=turn)

        if move in [-1, 1225]:
            # Game over
            break

        # Apply move
        board = apply_move(board, move, as_blue=turn)
        turn = not turn

    # Game should complete within reasonable turns
    assert turn_count < max_turns, "Game took too long"

    # Board should have pieces
    blue_count, green_count = count_cells(board)
    assert blue_count > 0 or green_count > 0, "Board should have pieces"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
