"""
Tests for action masking numerical stability.

These tests check the specific issues that can cause gradient explosion:
- Action masks with no valid actions
- Numerical stability when applying masks
- Policy output + masking interaction
- Edge cases in the environment
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib.ppo_mask import MaskablePPO

from env.env_virt import MicroscopeEnvDense
from lib.t7g import action_masks, BLUE, GREEN, CLEAR
from lib.networks import MicroscopeCNN


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


class TestActionMaskingStability:
    """Test action masking doesn't cause numerical issues"""

    def test_masks_always_have_valid_action_or_terminal(self):
        """Every non-terminal state must have at least one valid action"""
        env = MicroscopeEnvDense()

        # Play 100 random games
        for game in range(100):
            obs, _ = env.reset()
            done = False
            steps = 0

            while not done and steps < 400:  # Doubled for two-stage actions
                # Get action masks
                masks = env.action_masks()

                # Check that we have at least one valid action
                num_valid = np.sum(masks)
                assert num_valid > 0, f"No valid actions at step {steps} (should be terminal)"

                # Take random valid action
                valid_actions = np.where(masks)[0]
                action = np.random.choice(valid_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

    def test_masked_probabilities_sum_to_one(self):
        """After masking, probabilities should still sum to 1.0"""
        env = MicroscopeEnvDense()

        # Create a simple untrained model for testing
        model = MaskablePPO('CnnPolicy', env, verbose=0, policy_kwargs={
            "features_extractor_class": MicroscopeCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": dict(pi=[128], vf=[128]),
            "activation_fn": nn.ReLU,
        })

        # Test on multiple states
        for _ in range(50):
            obs, _ = env.reset()
            masks = env.action_masks()

            # Get policy logits
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                action_logits = model.policy.action_net(latent_pi).squeeze(0)

            # Apply masking manually (what the policy does internally)
            masks_tensor = torch.from_numpy(masks).to(model.device)
            masked_logits = torch.where(
                masks_tensor,
                action_logits,
                torch.tensor(float('-inf'), device=model.device)
            )

            # Get probabilities
            probs = torch.softmax(masked_logits, dim=0)
            prob_sum = probs.sum().item()

            # Check sum is close to 1.0
            assert abs(prob_sum - 1.0) < 1e-5, \
                f"Masked probabilities sum to {prob_sum}, not 1.0"

            # Check no NaN or Inf
            assert not torch.isnan(probs).any(), "NaN in probabilities"
            assert not torch.isinf(probs).any(), "Inf in probabilities"

    def test_single_valid_action_case(self):
        """When only one action is valid, it should get probability 1.0"""
        env = MicroscopeEnvDense()
        model = MaskablePPO('CnnPolicy', env, verbose=0, policy_kwargs={
            "features_extractor_class": MicroscopeCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "activation_fn": nn.ReLU,
        })

        # Create a state with very few valid moves
        env.game_grid = setup_board(
            blue_positions=[(3, 3)],
            green_positions=[]
        )
        env.turn = True
        obs = env._get_obs()

        masks = env.action_masks()
        num_valid = np.sum(masks)

        # If only one valid action
        if num_valid == 1:
            valid_action = np.where(masks)[0][0]

            # Get model's action distribution
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(model.device)
            masks_tensor = torch.from_numpy(masks).unsqueeze(0).to(model.device)

            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                distribution.apply_masking(masks_tensor)

                # Sample multiple times - should always get the same action
                for _ in range(10):
                    action = distribution.sample()
                    assert action.item() == valid_action, \
                        "With one valid action, should always select it"

    def test_no_extreme_logits_with_masking(self):
        """Check that masking doesn't create extreme logit values"""
        env = MicroscopeEnvDense()
        model = MaskablePPO('CnnPolicy', env, verbose=0, policy_kwargs={
            "features_extractor_class": MicroscopeCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "activation_fn": nn.ReLU,
        })

        for _ in range(20):
            obs, _ = env.reset()
            masks = env.action_masks()

            # Get unmasked logits
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi).squeeze(0)

            # Check logits are in reasonable range
            logit_min = logits.min().item()
            logit_max = logits.max().item()
            logit_range = logit_max - logit_min

            # Logits shouldn't be too extreme (indicates unstable policy)
            assert abs(logit_max) < 100, f"Logit too large: {logit_max}"
            assert abs(logit_min) < 100, f"Logit too small: {logit_min}"
            assert logit_range < 200, f"Logit range too large: {logit_range}"


class TestEnvironmentConsistency:
    """Test environment doesn't produce inconsistent states"""

    def test_turn_indicator_consistency(self):
        """Turn indicator should match env.turn"""
        env = MicroscopeEnvDense()

        for _ in range(100):
            obs, _ = env.reset()

            # Check initial turn
            turn_channel = obs[:, :, 2]
            expected = 1.0 if env.turn else 0.0
            assert np.all(turn_channel == expected), \
                f"Turn indicator {turn_channel[0,0]} doesn't match env.turn {env.turn}"

            # Take random valid action
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

            # Check turn flipped
            turn_channel = obs[:, :, 2]
            expected = 1.0 if env.turn else 0.0
            assert np.all(turn_channel == expected), \
                "Turn indicator doesn't match after step"

    def test_observation_values_in_range(self):
        """Observations should always be 0.0 or 1.0"""
        env = MicroscopeEnvDense()

        for _ in range(50):
            obs, _ = env.reset()

            for step in range(100):
                # Check all values are binary
                assert np.all((obs >= 0.0) & (obs <= 1.0)), \
                    "Observation values outside [0, 1]"

                # Check only 0.0 and 1.0 (no intermediate values)
                unique = np.unique(obs)
                for val in unique:
                    assert val in [0.0, 1.0], \
                        f"Observation has non-binary value: {val}"

                masks = env.action_masks()
                if not np.any(masks):
                    break

                valid_actions = np.where(masks)[0]
                action = np.random.choice(valid_actions)
                obs, _, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    break

    def test_action_masks_match_validity(self):
        """Actions that pass is_action_valid should be in mask for two-stage system"""
        env = MicroscopeEnvDense()

        for _ in range(50):
            obs, _ = env.reset()

            # Stage 0: Test piece selection masks
            assert env.action_stage == 0
            masks_stage0 = env.action_masks()
            assert masks_stage0.shape == (49,), "Stage 0 should have 49 actions"

            # All valid piece positions should have pieces of current color
            color = BLUE if env.turn else GREEN
            for pos_idx in range(49):
                x = pos_idx % 7
                y = pos_idx // 7
                has_piece = np.array_equal(env.game_grid[y, x], color)
                assert masks_stage0[pos_idx] == has_piece, \
                    f"Position ({x},{y}): mask={masks_stage0[pos_idx]}, has_piece={has_piece}"


class TestNumericalStability:
    """Test for numerical issues that cause gradient explosion"""

    def test_no_underflow_with_many_invalid_actions(self):
        """When most actions are invalid, check for numerical underflow"""
        env = MicroscopeEnvDense()
        model = MaskablePPO('CnnPolicy', env, verbose=0, policy_kwargs={
            "features_extractor_class": MicroscopeCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "activation_fn": nn.ReLU,
        })

        # Create board with very few valid actions
        env.game_grid = setup_board(
            blue_positions=[(0, 0)],
            green_positions=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        )
        env.turn = True
        obs = env._get_obs()

        masks = env.action_masks()
        num_valid = np.sum(masks)

        if num_valid < 10:  # Very constrained
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(model.device)
            masks_tensor = torch.from_numpy(masks).to(model.device)

            with torch.no_grad():
                features = model.policy.extract_features(obs_tensor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi).squeeze(0)

                # Apply masking
                masked_logits = torch.where(
                    masks_tensor,
                    logits,
                    torch.tensor(float('-inf'), device=model.device)
                )

                # Get probabilities
                probs = torch.softmax(masked_logits, dim=0)

                # Check for numerical issues
                assert not torch.isnan(probs).any(), "NaN in probabilities"
                assert not torch.isinf(probs).any(), "Inf in probabilities"

                # Check probabilities of valid actions are reasonable
                valid_probs = probs[masks_tensor]
                assert torch.all(valid_probs > 1e-10), \
                    f"Probability underflow: min={valid_probs.min().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
