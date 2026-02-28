"""
Test the MinimaxOpponentWrapper to ensure it properly integrates minimax opponent.

Run with: pytest tests/test_minimax_wrapper.py -v -s
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from env.env_virt import MicroscopeEnv, MicroscopeEnvSimple
from env.minimax_wrapper import MinimaxOpponentWrapper
from lib.t7g import count_cells


def test_wrapper_initialization():
    """Test that wrapper initializes correctly"""
    env = MicroscopeEnv()
    wrapped = MinimaxOpponentWrapper(env, depth=3)

    obs, info = wrapped.reset()
    assert obs.shape == (7, 7, 3), "Observation shape should be preserved"
    assert wrapped.depth == 3, "Depth should be set correctly"


def test_opponent_makes_moves():
    """Test that opponent actually makes moves after agent"""
    env = MicroscopeEnv()
    wrapped = MinimaxOpponentWrapper(env, depth=1)

    obs, info = wrapped.reset()

    # Count initial pieces (should be 2 blue, 2 green)
    blue_before, green_before = count_cells(obs)
    assert blue_before == 2
    assert green_before == 2

    # Get valid action
    masks = wrapped.action_masks()
    valid_actions = np.where(masks)[0]
    assert len(valid_actions) > 0, "Should have valid moves"

    # Agent makes move
    action = valid_actions[0]
    obs, reward, terminated, truncated, info = wrapped.step(action)

    # Check that opponent made a move
    assert 'opponent_moves' in info, "Should track opponent moves"
    assert info['opponent_moves'] == 1, "Opponent should have made 1 move"
    assert info['agent_moves'] == 1, "Agent should have made 1 move"

    # Board should have changed (more pieces after both moves)
    blue_after, green_after = count_cells(obs)
    assert blue_after > blue_before or green_after > green_before, \
        "Board should have changed after moves"


def test_game_completes():
    """Test that games can complete with minimax opponent"""
    env = MicroscopeEnvSimple()
    wrapped = MinimaxOpponentWrapper(env, depth=1)  # Use depth 1 for speed

    obs, info = wrapped.reset()
    max_steps = 100
    step_count = 0
    episode_over = False

    while not episode_over and step_count < max_steps:
        masks = wrapped.action_masks()
        valid_actions = np.where(masks)[0]

        if len(valid_actions) == 0:
            break

        # Random policy
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        step_count += 1
        episode_over = terminated or truncated

    # Game should complete within reasonable number of steps
    assert episode_over or step_count == max_steps, \
        "Game should complete or hit step limit"

    # Info should have move counts
    if 'agent_moves' in info:
        print(f"\n  Agent moves: {info['agent_moves']}")
        print(f"  Opponent moves: {info['opponent_moves']}")
        print(f"  Total steps: {step_count}")


def test_different_depths():
    """Test that different minimax depths work"""
    for depth in [1, 2, 3]:
        env = MicroscopeEnv()
        wrapped = MinimaxOpponentWrapper(env, depth=depth)

        obs, info = wrapped.reset()

        # Take one step
        masks = wrapped.action_masks()
        valid_actions = np.where(masks)[0]
        action = valid_actions[0]

        obs, reward, terminated, truncated, info = wrapped.step(action)

        assert 'opponent_moves' in info
        print(f"\n  Depth {depth}: OK")


@pytest.mark.slow
def test_multiple_games():
    """Test multiple games to ensure consistency"""
    env = MicroscopeEnvSimple()
    wrapped = MinimaxOpponentWrapper(env, depth=2)

    outcomes = []

    for game in range(5):
        obs, info = wrapped.reset()
        step_count = 0
        episode_over = False

        while not episode_over and step_count < 100:
            masks = wrapped.action_masks()
            valid_actions = np.where(masks)[0]

            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = wrapped.step(action)

            step_count += 1
            episode_over = terminated or truncated

        blue, green = count_cells(obs)
        outcome = "WIN" if blue > green else "LOSS" if green > blue else "DRAW"
        outcomes.append(outcome)

        print(f"\n  Game {game+1}: {outcome} ({blue}v{green}, {step_count} steps)")

    print(f"\n  Results: {outcomes.count('WIN')} wins, {outcomes.count('LOSS')} losses, {outcomes.count('DRAW')} draws")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
