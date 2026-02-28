"""
Tests for two-stage action space implementation.

These tests verify that the two-stage action system works correctly:
- Stage transitions (0 -> 1 -> 0)
- Observation includes selected piece indicator
- Action masks are correct for each stage
- Full game moves execute properly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from env.env_virt import MicroscopeEnvDense
from lib.t7g import is_action_valid, BLUE, GREEN


class TestTwoStageActions:
    """Test two-stage action space implementation"""

    def test_initial_state_is_stage_zero(self):
        """Environment should start at stage 0 (piece selection)"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        assert env.action_stage == 0, "Should start at stage 0"
        assert env.selected_piece_pos is None, "No piece should be selected"

    def test_stage_transitions(self):
        """Test that stages transition correctly: 0 -> 1 -> 0"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        # Should start at stage 0
        assert env.action_stage == 0
        assert env.selected_piece_pos is None

        # Select a piece (stage 0 action)
        masks = env.action_masks()
        assert masks.shape == (49,), "Stage 0 should have 49 actions"

        valid_actions = np.where(masks)[0]
        assert len(valid_actions) > 0, "Should have valid pieces to select"

        piece_action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(piece_action)

        # Should move to stage 1
        assert env.action_stage == 1, "Should transition to stage 1"
        assert env.selected_piece_pos is not None, "Piece should be selected"
        assert reward == 0.0, "No reward for piece selection"
        assert not terminated, "Should not terminate at stage 1"
        assert not truncated, "Should not truncate at stage 1"

        # Select a move (stage 1 action)
        masks = env.action_masks()
        assert masks.shape == (49,), "Masks should always be 49-dimensional"
        # Only first 25 positions can be valid (moves), rest are padding
        assert not np.any(masks[25:]), "Positions 25-48 should be False"

        valid_moves = np.where(masks)[0]
        assert len(valid_moves) > 0, "Should have valid moves"

        move_action = valid_moves[0]
        obs, reward, terminated, truncated, info = env.step(move_action)

        # Should return to stage 0 (for next player)
        assert env.action_stage == 0, "Should return to stage 0"
        assert env.selected_piece_pos is None, "Selected piece should be cleared"

    def test_selected_piece_in_observation(self):
        """Test that selected piece appears in observation channel 3"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        # Channel 3 should be all zeros at stage 0
        assert obs.shape == (7, 7, 4), "Observation should be 7x7x4"
        assert np.all(obs[:, :, 3] == 0.0), "Channel 3 should be empty at stage 0"

        # Select a piece
        masks = env.action_masks()
        valid_actions = np.where(masks)[0]
        piece_action = valid_actions[0]

        obs, _, _, _, _ = env.step(piece_action)

        # Channel 3 should have exactly one 1.0
        assert np.sum(obs[:, :, 3]) == 1.0, "Exactly one position should be selected"

        # It should be at the selected position
        selected_positions = np.where(obs[:, :, 3] == 1.0)
        assert len(selected_positions[0]) == 1, "Should have one selected position"

        selected_y = selected_positions[0][0]
        selected_x = selected_positions[1][0]

        expected_x = piece_action % 7
        expected_y = piece_action // 7

        assert selected_x == expected_x, f"X mismatch: {selected_x} != {expected_x}"
        assert selected_y == expected_y, f"Y mismatch: {selected_y} != {expected_y}"

    def test_piece_selection_masks_only_current_color(self):
        """Stage 0 masks should only allow pieces of current player's color"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        # Blue's turn first
        assert env.turn == True, "Blue should go first"

        masks = env.action_masks()
        color = BLUE if env.turn else GREEN

        # Check each position
        for pos_idx in range(49):
            x = pos_idx % 7
            y = pos_idx // 7
            has_our_piece = np.array_equal(env.game_grid[y, x], color)
            assert masks[pos_idx] == has_our_piece, \
                f"Position ({x},{y}): mask={masks[pos_idx]}, has_piece={has_our_piece}"

    def test_move_masks_respect_selected_piece(self):
        """Stage 1 masks should only allow valid moves from selected piece"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        # Select a piece
        masks_stage0 = env.action_masks()
        valid_pieces = np.where(masks_stage0)[0]
        piece_action = valid_pieces[0]

        obs, _, _, _, _ = env.step(piece_action)

        # Get move masks
        masks_stage1 = env.action_masks()
        assert masks_stage1.shape == (49,), "Masks should always be 49-dimensional"
        assert not np.any(masks_stage1[25:]), "Positions 25-48 should be False (padding)"

        # Verify all valid moves are actually valid for this piece
        selected_x = piece_action % 7
        selected_y = piece_action // 7

        for move_idx in range(25):
            dx = (move_idx % 5) - 2
            dy = (move_idx // 5) - 2

            # Reconstruct full action (in original 1225 action space)
            full_action = selected_y * 7 * 25 + selected_x * 25 + move_idx

            # Check if mask matches actual validity
            is_valid = is_action_valid(env.game_grid, full_action, env.turn)
            assert masks_stage1[move_idx] == is_valid, \
                f"Move ({dx},{dy}) from ({selected_x},{selected_y}): " \
                f"mask={masks_stage1[move_idx]}, valid={is_valid}"

    def test_full_game_completes(self):
        """Test that a full game can be played with two-stage actions"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        done = False
        steps = 0
        max_steps = 400  # Doubled since each move is 2 steps

        while not done and steps < max_steps:
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]

            if len(valid_actions) == 0:
                break  # No valid actions, game should end

            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            steps += 1

        # Game should complete without errors
        assert steps > 0, "Game should have taken some steps"
        assert obs.shape == (7, 7, 4), "Observation shape should be correct"

    def test_reward_only_after_move_completes(self):
        """Reward should only be given after stage 1 (full move complete)"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        # Play 10 moves and check reward timing
        for _ in range(10):
            # Stage 0: Select piece
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) == 0:
                break

            piece_action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(piece_action)

            # Reward should be 0 after piece selection
            assert reward == 0.0, f"Stage 0 should give 0 reward, got {reward}"
            assert not terminated, "Should not terminate at stage 0"

            # Stage 1: Select move
            masks = env.action_masks()
            valid_moves = np.where(masks)[0]
            if len(valid_moves) == 0:
                break

            move_action = valid_moves[0]
            obs, reward, terminated, truncated, info = env.step(move_action)

            # Reward can be non-zero after move completes
            # (actual value depends on game state)

            if terminated or truncated:
                break

    def test_episode_reset_clears_stage(self):
        """Reset should clear two-stage state"""
        env = MicroscopeEnvDense()
        obs, _ = env.reset()

        # Make a partial move (select piece but don't move)
        masks = env.action_masks()
        valid_actions = np.where(masks)[0]
        piece_action = valid_actions[0]
        env.step(piece_action)

        # Should be at stage 1 with selected piece
        assert env.action_stage == 1
        assert env.selected_piece_pos is not None

        # Reset
        obs, _ = env.reset()

        # Should be back at stage 0 with no selection
        assert env.action_stage == 0, "Reset should clear to stage 0"
        assert env.selected_piece_pos is None, "Reset should clear selected piece"
        assert np.all(obs[:, :, 3] == 0.0), "Reset should clear selected piece indicator"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
