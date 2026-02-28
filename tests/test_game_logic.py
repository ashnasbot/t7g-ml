"""
Tests for core game logic in util/t7g.py

Tests cover:
- Action encoding/decoding
- Move validation
- Capture mechanics
- Board state updates
- Action mask generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from lib.t7g import (
    action_to_move, is_action_valid, action_masks,
    count_cells, BLUE, GREEN, CLEAR
)


def setup_board(blue_positions=None, green_positions=None):
    """Helper to create a board with specific piece positions"""
    board = np.zeros((7, 7, 2), dtype=bool)

    if blue_positions:
        for x, y in blue_positions:
            board[y, x] = BLUE

    if green_positions:
        for x, y in green_positions:
            board[y, x] = GREEN

    return board


class TestActionEncoding:
    """Test action encoding and decoding"""

    def test_action_to_move_encoding(self):
        """Action correctly encodes to move parameters"""
        # Action 0: piece at (0,0), move to (-2,-2) offset
        from_x, from_y, to_x, to_y, jump = action_to_move(0)

        assert from_x == 0
        assert from_y == 0
        assert to_x == -2
        assert to_y == -2
        assert jump == True  # 2-square move is a jump

    def test_action_to_move_non_jump(self):
        """1-square moves are not jumps"""
        # Action that represents 1-square move
        # Piece at (0,0), move to offset (1,1) = action 0*25 + (1*5 + 1) + 2*5 + 2 = 18
        action = 0 * 25 + (3 * 5 + 3)  # offset (0,0) -> (1,1)
        from_x, from_y, to_x, to_y, jump = action_to_move(action)

        # 1-square moves should not be jumps
        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)
        expected_jump = dx == 2 or dy == 2

        assert jump == expected_jump

    def test_action_space_coverage(self):
        """All 1225 actions decode without error"""
        for action in range(1225):
            from_x, from_y, to_x, to_y, jump = action_to_move(action)

            # Check values are in valid range
            assert 0 <= from_x < 7
            assert 0 <= from_y < 7
            assert -2 <= to_x < 9  # Can be outside board
            assert -2 <= to_y < 9
            assert isinstance(jump, bool)


class TestMoveValidation:
    """Test move validity checking"""

    def test_valid_move_from_occupied_square(self):
        """Can move from square with own piece"""
        board = setup_board(blue_positions=[(3, 3)])

        # Find action: from (3,3) to (3,4) - 1 square move
        # action = piece * 25 + move
        # piece = 3 + 3*7 = 24
        # move = offset to (3,4) = (0,1) = (2+0, 2+1) = (2,3) in 5x5 = 3*5+2 = 17
        action = 24 * 25 + 17

        assert is_action_valid(board, action, True)  # Blue's turn

    def test_cannot_move_from_empty_square(self):
        """Cannot move from empty square"""
        board = setup_board(blue_positions=[(0, 0)])

        # Try to move from (3,3) which is empty
        action = 24 * 25 + 12  # From (3,3) to (3,3) - any move

        assert not is_action_valid(board, action, True)

    def test_cannot_move_to_occupied_square(self):
        """Cannot move to square with any piece"""
        board = setup_board(
            blue_positions=[(3, 3), (3, 4)],
            green_positions=[]
        )

        # Try to move from (3,3) to (3,4) - occupied by blue
        action = 24 * 25 + 17  # (3,3) -> (3,4)

        assert not is_action_valid(board, action, True)

    def test_cannot_move_off_board(self):
        """Cannot move outside 7x7 board"""
        board = setup_board(blue_positions=[(0, 0)])

        # Try to move from (0,0) to (-1,-1)
        action = 0 * 25 + 0  # Offset (-2,-2) from (0,0)

        assert not is_action_valid(board, action, True)

    def test_cannot_move_opponent_piece(self):
        """Cannot move opponent's piece"""
        board = setup_board(green_positions=[(3, 3)])

        # Try to move green piece on blue's turn
        action = 24 * 25 + 12

        assert not is_action_valid(board, action, True)


class TestActionMasks:
    """Test action mask generation"""

    def test_initial_position_has_moves(self):
        """Standard starting position has valid moves"""
        board = setup_board(
            blue_positions=[(0, 0), (6, 6)],
            green_positions=[(0, 6), (6, 0)]
        )

        masks = action_masks(board, True)

        # Should have some valid moves
        assert np.any(masks)
        assert masks.shape == (1225,)
        assert masks.dtype == bool

    def test_stuck_position_has_no_moves(self):
        """Position with no valid moves returns empty mask"""
        # Create board where blue is completely surrounded by both adjacent and jump range
        # Must fill all squares within 2-square radius to block all moves
        board = setup_board(
            blue_positions=[(3, 3)],
            green_positions=[
                # Adjacent squares (1-square range)
                (2, 2), (2, 3), (2, 4),
                (3, 2),         (3, 4),
                (4, 2), (4, 3), (4, 4),
                # Jump squares (2-square range)
                (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 1),                         (2, 5),
                (3, 1),                         (3, 5),
                (4, 1),                         (4, 5),
                (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)
            ]
        )

        masks = action_masks(board, True)

        # Blue has no moves (all squares in range are occupied)
        assert not np.any(masks)

    def test_mask_matches_individual_validation(self):
        """Action mask should match is_action_valid for all actions"""
        board = setup_board(
            blue_positions=[(3, 3)],
            green_positions=[(5, 5)]
        )

        masks = action_masks(board, True)

        # Check consistency for a sample of actions
        for action in range(0, 1225, 50):  # Sample every 50th action
            mask_says_valid = masks[action]
            actually_valid = is_action_valid(board, action, True)

            assert mask_says_valid == actually_valid, \
                f"Mismatch at action {action}: mask={mask_says_valid}, valid={actually_valid}"


class TestCaptureLogic:
    """Test piece capture mechanics"""

    def test_moving_next_to_opponent_captures(self):
        """Moving adjacent to opponent piece captures it"""
        from env.env_virt import MicroscopeEnv

        env = MicroscopeEnv()
        env.reset()

        # Set up a specific board: blue at (3,3), green at (3,5)
        env.game_grid = setup_board(
            blue_positions=[(3, 3)],
            green_positions=[(3, 5)]
        )
        env.turn = True  # Blue's turn

        # Move blue from (3,3) to (3,4) - next to green at (3,5)
        # This should capture the green piece
        action = 24 * 25 + 17  # (3,3) -> (3,4)

        env.move(action)

        # Green piece should be captured (converted to blue)
        assert np.array_equal(env.game_grid[5, 3], BLUE)

    def test_capture_radius_is_3x3(self):
        """Capture affects 3x3 area around destination"""
        from env.env_virt import MicroscopeEnv

        env = MicroscopeEnv()
        env.reset()

        # Blue at (2,2), one green piece adjacent at (3,3)
        env.game_grid = setup_board(
            blue_positions=[(2, 2)],
            green_positions=[(3, 3)]
        )
        env.turn = True

        # Move blue from (2,2) to (3,2) - adjacent to green at (3,3)
        # This should capture the green piece
        # Piece at (2,2) = 2 + 2*7 = 16
        # Offset to (3,2) = (1,0) in moves = (2+1, 2+0) in 5x5 = (3, 2) = 2*5+3 = 13
        action = 16 * 25 + 13

        env.move(action)

        # Green at (3,3) should now be blue (within 3x3 of destination (3,2))
        assert np.array_equal(env.game_grid[3, 3], BLUE), \
            f"Expected blue at (3,3) but got {env.game_grid[3, 3]}"


class TestBoardState:
    """Test board state queries"""

    def test_count_cells_initial(self):
        """Count cells correctly on initial board"""
        board = setup_board(
            blue_positions=[(0, 0), (6, 6)],
            green_positions=[(0, 6), (6, 0)]
        )

        blue_count, green_count = count_cells(board)

        assert blue_count == 2
        assert green_count == 2

    def test_count_cells_asymmetric(self):
        """Count cells correctly on asymmetric board"""
        board = setup_board(
            blue_positions=[(0, 0), (1, 1), (2, 2)],
            green_positions=[(6, 6)]
        )

        blue_count, green_count = count_cells(board)

        assert blue_count == 3
        assert green_count == 1

    def test_count_cells_empty_board(self):
        """Count cells on empty board"""
        board = setup_board()

        blue_count, green_count = count_cells(board)

        assert blue_count == 0
        assert green_count == 0


class TestJumpMechanics:
    """Test jump (2-square) move mechanics"""

    def test_jump_removes_source_piece(self):
        """Jump move removes piece from source square"""
        from env.env_virt import MicroscopeEnv

        env = MicroscopeEnv()
        env.reset()

        # Blue at (3,3)
        env.game_grid = setup_board(blue_positions=[(3, 3)])
        env.turn = True

        # Jump from (3,3) to (3,5) - 2 square move (0,2) offset
        # Piece at (3,3) = 3 + 3*7 = 24
        # Offset (0,2) in 5x5 grid centered at (2,2) = (2+0, 2+2) = (2,4) = 4*5+2 = 22
        action = 24 * 25 + 22

        env.move(action)

        # Source square should be empty (jump removes source)
        assert np.array_equal(env.game_grid[3, 3], CLEAR), \
            f"Source (3,3) should be CLEAR but got {env.game_grid[3, 3]}"

        # Destination should have blue piece
        assert np.array_equal(env.game_grid[5, 3], BLUE), \
            f"Destination (3,5) should be BLUE but got {env.game_grid[5, 3]}"

    def test_non_jump_keeps_source_piece(self):
        """Non-jump move keeps piece at source (clone)"""
        from env.env_virt import MicroscopeEnv

        env = MicroscopeEnv()
        env.reset()

        # Blue at (3,3)
        env.game_grid = setup_board(blue_positions=[(3, 3)])
        env.turn = True

        # Move 1 square from (3,3) to (3,4)
        action = 24 * 25 + 17  # (3,3) -> (3,4) offset (0,1)

        env.move(action)

        # Source square should still have blue piece
        assert np.array_equal(env.game_grid[3, 3], BLUE)

        # Destination should also have blue piece
        assert np.array_equal(env.game_grid[4, 3], BLUE)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_piece_at_corner_has_moves(self):
        """Piece in corner can still move"""
        board = setup_board(blue_positions=[(0, 0)])

        masks = action_masks(board, True)

        # Should have some valid moves even from corner
        assert np.any(masks)

    def test_piece_at_center_has_many_moves(self):
        """Piece at center has maximum mobility"""
        board = setup_board(blue_positions=[(3, 3)])

        masks = action_masks(board, True)

        valid_count = np.sum(masks)

        # Center piece should have many moves (up to 24 in empty board)
        assert valid_count > 10

    def test_full_board_has_no_moves(self):
        """Completely full board has no valid moves"""
        # Create alternating pattern filling the board
        blue_positions = [(x, y) for x in range(7) for y in range(7) if (x + y) % 2 == 0]
        green_positions = [(x, y) for x in range(7) for y in range(7) if (x + y) % 2 == 1]

        board = setup_board(blue_positions=blue_positions, green_positions=green_positions)

        masks = action_masks(board, True)

        # No moves possible on full board
        assert not np.any(masks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
