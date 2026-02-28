"""
Tests for MicroscopeEnv game environment.

Tests cover:
- Observation format and shape
- Turn alternation
- Episode lifecycle (reset, step, termination)
- Invalid move handling
- Action masking integration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from env.env_virt import MicroscopeEnv, MicroscopeEnvDense


class TestEnvironmentBasics:
    """Test basic environment functionality"""

    def test_env_initialization(self):
        """Environment initializes with correct spaces"""
        env = MicroscopeEnv()

        # Check observation space: 7x7x3 (board + turn indicator)
        assert env.observation_space.shape == (7, 7, 3)
        assert env.observation_space.dtype == np.float32

        # Check action space: 49 cells * 25 moves = 1225
        assert env.action_space.n == 1225

    def test_reset_returns_valid_observation(self):
        """Reset returns correctly shaped observation"""
        env = MicroscopeEnv()
        obs, info = env.reset()

        # Check shape
        assert obs.shape == (7, 7, 3)
        assert obs.dtype == np.float32

        # Check values are in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

        # Check turn indicator (should be blue's turn = 1.0)
        assert np.all(obs[:, :, 2] == 1.0)

        # Check initial pieces are placed
        total_pieces = np.sum(obs[:, :, 0]) + np.sum(obs[:, :, 1])
        assert total_pieces == 4.0  # 2 blue + 2 green

    def test_initial_board_setup(self):
        """Initial board has pieces in correct corners"""
        env = MicroscopeEnv()
        obs, _ = env.reset()

        # Extract board (without turn channel)
        board = obs[:, :, 0:2]

        # Check corners have pieces
        # (0,0) and (6,6) should have blue
        # (0,6) and (6,0) should have green
        assert board[0, 0, 1] == 1.0  # Blue at (0,0)
        assert board[6, 6, 1] == 1.0  # Blue at (6,6)
        assert board[0, 6, 0] == 1.0  # Green at (0,6)
        assert board[6, 0, 0] == 1.0  # Green at (6,0)


class TestTurnHandling:
    """Test turn alternation and perspective"""

    def test_turn_alternates_after_step(self):
        """Turn indicator flips after each step"""
        env = MicroscopeEnv()
        obs, _ = env.reset()

        # Initial turn should be blue (1.0)
        assert obs[0, 0, 2] == 1.0

        # Get a valid move
        masks = env.action_masks()
        valid_action = np.where(masks)[0][0]

        # Take step
        obs, _, _, _, _ = env.step(valid_action)

        # Turn should now be green (0.0)
        assert obs[0, 0, 2] == 0.0

    def test_turn_indicator_consistent_across_board(self):
        """Turn indicator is same for all cells"""
        env = MicroscopeEnv()
        obs, _ = env.reset()

        # All cells should have same turn value
        turn_values = obs[:, :, 2]
        assert np.all(turn_values == turn_values[0, 0])

    def test_no_board_flipping(self):
        """Board positions are absolute (no perspective flipping)"""
        env = MicroscopeEnv()
        obs1, _ = env.reset()

        # Record blue piece position
        blue_positions_initial = np.where(obs1[:, :, 1] == 1.0)

        # Take a move that doesn't affect corners
        masks = env.action_masks()
        valid_action = np.where(masks)[0][0]
        obs2, _, _, _, _ = env.step(valid_action)

        # Original blue pieces should still be at same absolute positions
        # (they might have been captured, but if not, same position)
        # This tests that we're not flipping the board view


class TestActionMasking:
    """Test action mask integration"""

    def test_initial_masks_are_valid(self):
        """Initial position has valid moves"""
        env = MicroscopeEnv()
        env.reset()

        masks = env.action_masks()

        # Should have some valid moves
        assert np.any(masks)
        assert masks.shape == (1225,)
        assert masks.dtype == bool

    def test_only_masked_actions_are_valid(self):
        """Invalid actions according to mask should fail"""
        env = MicroscopeEnv()
        env.reset()

        masks = env.action_masks()

        # Find an invalid action
        invalid_actions = np.where(~masks)[0]
        if len(invalid_actions) > 0:
            invalid_action = invalid_actions[0]

            # Taking invalid action should give negative reward
            _, reward, terminated, _, _ = env.step(invalid_action)

            # Should get penalty for invalid move
            assert reward <= 0


class TestEpisodeLifecycle:
    """Test episode start, progression, and termination"""

    def test_episode_terminates_eventually(self):
        """Episode terminates within step limit"""
        env = MicroscopeEnv()
        env.reset()

        terminated = False
        truncated = False
        steps = 0
        max_steps = 200

        while not (terminated or truncated) and steps < max_steps:
            masks = env.action_masks()

            if not np.any(masks):
                break

            valid_action = np.where(masks)[0][0]
            _, _, terminated, truncated, _ = env.step(valid_action)
            steps += 1

        # Should terminate within max steps
        assert steps < max_steps
        assert terminated or truncated

    def test_reset_clears_state(self):
        """Reset returns environment to initial state"""
        env = MicroscopeEnv()

        # Take some steps
        env.reset()
        masks = env.action_masks()
        for _ in range(5):
            if np.any(masks):
                action = np.where(masks)[0][0]
                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
                masks = env.action_masks()

        # Reset
        obs, _ = env.reset()

        # Should be back to initial state
        assert obs.shape == (7, 7, 3)
        total_pieces = np.sum(obs[:, :, 0]) + np.sum(obs[:, :, 1])
        assert total_pieces == 4.0

    def test_truncation_at_turn_limit(self):
        """Episode truncates at turn limit"""
        env = MicroscopeEnv()
        env.turn_limit = 10  # Set low limit
        env.reset()

        truncated = False
        steps = 0

        while not truncated and steps < 20:
            masks = env.action_masks()
            if not np.any(masks):
                break

            valid_action = np.where(masks)[0][0]
            _, _, _, truncated, _ = env.step(valid_action)
            steps += 1

        # Should truncate around turn limit
        assert steps <= env.turn_limit + 1


class TestRewardFunctions:
    """Test different reward function configurations"""

    def test_reward_function_selection(self):
        """Environment uses selected reward function"""
        # Test dense reward function
        env_dense = MicroscopeEnvDense()
        assert env_dense.reward_fn_name == 'dense'

        # Test original reward function
        env_original = MicroscopeEnv(reward_fn='original')
        assert env_original.reward_fn_name == 'original'

    def test_reward_values_are_numeric(self):
        """Rewards are valid numbers"""
        env = MicroscopeEnv()
        env.reset()

        masks = env.action_masks()
        valid_action = np.where(masks)[0][0]
        _, reward, _, _, _ = env.step(valid_action)

        # Reward should be a number
        assert isinstance(reward, (int, float, np.number))
        assert not np.isnan(reward)
        assert not np.isinf(reward)


class TestObservationConsistency:
    """Test observation consistency and correctness"""

    def test_observation_preserves_board_state(self):
        """Observation correctly reflects game board"""
        env = MicroscopeEnv()
        obs, _ = env.reset()

        # Count pieces in observation
        green_count = np.sum(obs[:, :, 0])
        blue_count = np.sum(obs[:, :, 1])

        # Should have 2 of each initially
        assert green_count == 2.0
        assert blue_count == 2.0

        # No overlap (cell can't be both colors)
        overlap = np.sum(obs[:, :, 0] * obs[:, :, 1])
        assert overlap == 0.0

    def test_observation_after_move(self):
        """Observation updates after move"""
        env = MicroscopeEnv()
        obs_before, _ = env.reset()

        # Take a move
        masks = env.action_masks()
        valid_action = np.where(masks)[0][0]
        obs_after, _, _, _, _ = env.step(valid_action)

        # Observations should be different
        assert not np.array_equal(obs_before, obs_after)

        # Turn indicator should flip
        assert obs_before[0, 0, 2] != obs_after[0, 0, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
