"""
Test debug rendering behavior.

Run with: pytest tests/test_debug_rendering.py -v -s
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from env.env_virt import MicroscopeEnvSimple
from env.minimax_wrapper import MinimaxOpponentWrapper


def test_debug_false_no_move_output(capsys):
    """With debug=False, no per-move output should be printed"""
    env = MicroscopeEnvSimple(render_mode='human', debug=False)
    obs, _ = env.reset()

    # Get valid action
    masks = env.action_masks()
    valid_actions = [i for i, m in enumerate(masks) if m]
    assert len(valid_actions) > 0

    # Take one step
    action = valid_actions[0]
    obs, reward, terminated, truncated, info = env.step(action)

    # Capture output
    captured = capsys.readouterr()

    # Should NOT contain move details
    assert "B:" not in captured.out or "=>" not in captured.out.split('\n')[0], \
        "Should not print move details when debug=False"

    # Should NOT contain reward unless game ended
    if not terminated and not truncated:
        lines = captured.out.strip().split('\n')
        # Filter out the initial "Using reward function" message
        non_init_lines = [line for line in lines if 'Using reward function' not in line]
        assert len(non_init_lines) == 0 or "Reward:" not in captured.out, \
            "Should not print reward when debug=False and game not ended"


def test_debug_true_shows_move_output(capsys):
    """With debug=True, per-move output should be printed"""
    env = MicroscopeEnvSimple(render_mode='human', debug=True)
    obs, _ = env.reset()

    # Get valid action
    masks = env.action_masks()
    valid_actions = [i for i, m in enumerate(masks) if m]
    assert len(valid_actions) > 0

    # Take one step
    action = valid_actions[0]
    obs, reward, terminated, truncated, info = env.step(action)

    # Capture output
    captured = capsys.readouterr()

    # Should contain move details
    assert "B:" in captured.out and "=>" in captured.out, \
        "Should print move details when debug=True"

    # Should contain reward
    assert "Reward:" in captured.out, \
        "Should print reward when debug=True"


def test_final_board_always_shown(capsys):
    """Final board should always be shown when game ends, regardless of debug flag"""
    for debug in [False, True]:
        env = MicroscopeEnvSimple(render_mode='human', debug=debug)
        obs, _ = env.reset()

        # Play until game ends (max 100 steps)
        for _ in range(100):
            masks = env.action_masks()
            valid_actions = [i for i, m in enumerate(masks) if m]
            if not valid_actions:
                break

            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Capture output
        captured = capsys.readouterr()

        # Final board should be shown (contains ANSI color codes from show_board)
        if terminated or truncated:
            # show_board() uses ANSI color codes, check for those
            assert "\x1b[48;2" in captured.out, \
                f"Should show board at end when debug={debug}"


def test_minimax_wrapper_shows_final_board(capsys):
    """Minimax wrapper should show final board when game ends"""
    env = MicroscopeEnvSimple(render_mode='human', debug=False)
    env = MinimaxOpponentWrapper(env, depth=1)

    obs, _ = env.reset()

    # Play until game ends
    for _ in range(50):
        masks = env.action_masks()
        valid_actions = [i for i, m in enumerate(masks) if m]
        if not valid_actions:
            break

        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Capture output
    captured = capsys.readouterr()

    # Final board should be shown (contains ANSI color codes from show_board)
    if terminated or truncated:
        assert "\x1b[48;2" in captured.out, \
            "Wrapper should show board at game end"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
