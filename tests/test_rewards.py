"""
Test and compare different reward functions.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from env.env_virt import MicroscopeEnv
from lib.reward_functions import (
    calc_reward_simple, calc_reward_strategic, calc_reward_dense
)
from lib.t7g import show_board, action_to_move


def simulate_game_with_reward(reward_func, num_moves=10, render=False):
    """Simulate a game and track rewards"""
    env = MicroscopeEnv()
    obs, _ = env.reset()

    if render:
        print(f"\n{'='*60}")
        print(f"Testing: {reward_func.__name__}")
        print(f"{'='*60}")
        show_board(obs)

    rewards = []
    for move_num in range(num_moves):
        # Get valid actions
        valid_actions = env.action_masks()
        valid_indices = np.where(valid_actions)[0]

        if len(valid_indices) == 0:
            break

        # Random valid action
        action = np.random.choice(valid_indices)

        # Get environment's step
        obs, env_reward, terminated, truncated, _ = env.step(action)

        # Get our custom reward
        custom_reward, custom_term = reward_func(obs, env.turn)

        if render:
            from_x, from_y, to_x, to_y, jump = action_to_move(action)
            print(f"\nMove {move_num + 1}: [{from_x},{from_y}] -> [{to_x},{to_y}]")
            print(f"  Env reward: {env_reward:.3f}")
            print(f"  Custom reward: {custom_reward:.3f}")
            show_board(obs)

        rewards.append(custom_reward)

        if terminated or truncated:
            break

    env.close()
    return rewards


def compare_reward_functions():
    """Compare all reward functions on the same game states"""
    print("\n" + "="*70)
    print("REWARD FUNCTION COMPARISON")
    print("="*70)

    # Create consistent game states to test
    env = MicroscopeEnv()
    np.random.seed(42)  # For reproducibility

    print("\nTesting on 5 random game states...")

    reward_funcs = [
        ('Simple', calc_reward_simple),
        ('Strategic', calc_reward_strategic),
        ('Dense', calc_reward_dense),
    ]

    for game_num in range(5):
        obs, _ = env.reset()
        print(f"\n{'='*70}")
        print(f"Game State {game_num + 1}")
        print(f"{'='*70}")

        # Make a few random moves
        for _ in range(np.random.randint(3, 8)):
            valid_actions = env.action_masks()
            valid_indices = np.where(valid_actions)[0]
            if len(valid_indices) == 0:
                break
            action = np.random.choice(valid_indices)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        # Show board state
        from lib.t7g import count_cells
        blue_cells, green_cells = count_cells(obs)
        print(f"Board state: Blue={blue_cells}, Green={green_cells}")

        # Compare all reward functions
        print(f"\n{'Function':<15} {'Reward':<10} {'Terminated':<12} {'Notes'}")
        print("-" * 70)

        for name, func in reward_funcs:
            try:
                reward, terminated = func(obs, True)
                notes = ""
                if abs(reward) > 0.5:
                    notes = "Strong signal" if not terminated else "Terminal"
                elif abs(reward) > 0.2:
                    notes = "Moderate signal"
                else:
                    notes = "Weak signal"

                print(f"{name:<15} {reward:<10.3f} {str(terminated):<12} {notes}")
            except Exception as e:
                print(f"{name:<15} ERROR: {e}")

    env.close()

    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print("""
1. ORIGINAL:
   - BUG: as_blue parameter overwritten (always treats player as Blue)
   - Sparse rewards, mostly 0 until game end
   - Squared differences are confusing for learning
   - NOT RECOMMENDED

2. SIMPLE:
   - Fixes bugs from original
   - Linear rewards (better gradients)
   - Normalized [-1, 1] range
   - Win/loss bonuses for margin
   - RECOMMENDED for baseline

3. STRATEGIC:
   - Multiple reward components (material, mobility, center, potential)
   - Encourages sophisticated play
   - More complex, might be noisier
   - RECOMMENDED for advanced training

4. DENSE:
   - More frequent feedback than simple
   - Material + mobility only (lighter than strategic)
   - Good middle ground
   - RECOMMENDED for initial training

TRAINING PROGRESSION:
   Stage 1: Use DENSE for initial learning (more feedback)
   Stage 2: Switch to SIMPLE once agent learns basics
   Stage 3: Use STRATEGIC for final polish (sophisticated play)
""")


def test_reward_variance():
    """Test how much variance each reward function has"""
    print("\n" + "="*70)
    print("REWARD VARIANCE TEST (50 random games)")
    print("="*70)

    reward_funcs = [
        ('Simple', calc_reward_simple),
        ('Strategic', calc_reward_strategic),
        ('Dense', calc_reward_dense),
    ]

    np.random.seed(123)

    for name, func in reward_funcs:
        all_rewards = []
        for _ in range(50):
            rewards = simulate_game_with_reward(func, num_moves=15, render=False)
            all_rewards.extend(rewards)

        all_rewards = np.array(all_rewards)
        print(f"\n{name}:")
        print(f"  Mean reward: {np.mean(all_rewards):.4f}")
        print(f"  Std dev: {np.std(all_rewards):.4f}")
        print(f"  Min/Max: [{np.min(all_rewards):.4f}, {np.max(all_rewards):.4f}]")
        print(f"  % non-zero: {100 * np.count_nonzero(all_rewards) / len(all_rewards):.1f}%")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MICROSCOPE REWARD FUNCTION TEST SUITE")
    print("="*70)

    # Run comparison
    compare_reward_functions()

    # Run variance test
    test_reward_variance()

    print("\n" + "="*70)
    print("Testing complete! Check results above.")
    print("="*70)
    print("\nTo use a new reward function in training:")
    print("1. Import: from reward_functions import calc_reward_simple")
    print("2. Replace in env: self.calc_reward = calc_reward_simple")
    print("3. Or modify env_virt.py to use the new function")
