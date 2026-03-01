"""
Noisy minimax opponent — epsilon-greedy blundering.

With probability `noise` takes a random legal move; otherwise defers to
minimax at the configured depth. Gives a smooth difficulty knob between
pure random (noise=1.0) and pure minimax (noise=0.0).

Usage:
    from env.noisy_minimax_wrapper import NoisyMinimaxOpponentWrapper

    env = NoisyMinimaxOpponentWrapper(MicroscopeEnvAggressive(), depth=3, noise=0.3)
"""
import numpy as np
import gymnasium as gym
from lib.t7g import find_best_move, action_masks, show_board


class NoisyMinimaxOpponentWrapper(gym.Wrapper):
    """
    Epsilon-greedy minimax opponent.

    Takes a random legal move with probability `noise`, otherwise plays
    the minimax best move at `depth`. Creates a smooth difficulty curve
    between random (noise=1.0) and pure minimax (noise=0.0).

    Args:
        env:   The Microscope environment to wrap
        depth: Minimax search depth (1-5)
        noise: Probability of taking a random move instead of minimax (0.0-1.0)
    """

    def __init__(self, env, depth=3, noise=0.3):
        super().__init__(env)
        self.depth = depth
        self.noise = noise
        self.opponent_moves = 0
        self.agent_moves = 0
        self._base_env = self._find_base_env(env)

    def _find_base_env(self, env):
        """Walk wrapper stack to find the base env with game_grid."""
        current = env
        while hasattr(current, 'env'):
            if hasattr(current, 'game_grid'):
                return current
            current = current.env
        return current

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._base_env.action_stage == 0:
            self.agent_moves += 1

        if terminated or truncated:
            info['agent_moves'] = self.agent_moves
            info['opponent_moves'] = self.opponent_moves
            return obs, reward, terminated, truncated, info

        # Only respond after agent completes a full move (back at stage 0)
        if self._base_env.action_stage != 0:
            return obs, reward, terminated, truncated, info

        board = self._base_env.game_grid

        if np.random.random() < self.noise:
            # Random move: sample uniformly from the 1225-action legal set
            legal = np.where(action_masks(board, False))[0]
            if len(legal) == 0:
                terminated = True
                reward = 1.0
                info.update({'agent_moves': self.agent_moves,
                             'opponent_moves': self.opponent_moves,
                             'opponent_stuck': True})
                return obs, reward, terminated, truncated, info
            opponent_action_1225 = int(np.random.choice(legal))
        else:
            # Minimax best move
            opponent_action_1225 = find_best_move(board.tobytes(), self.depth, False)
            if opponent_action_1225 in (-1, 1225):
                terminated = True
                reward = 1.0
                info.update({'agent_moves': self.agent_moves,
                             'opponent_moves': self.opponent_moves,
                             'opponent_stuck': True})
                if self._base_env.render_mode == 'human':
                    show_board(board)
                return obs, reward, terminated, truncated, info

        # Convert 1225-action → two-stage piece + move actions
        from_x = (opponent_action_1225 // 25) % 7
        from_y = opponent_action_1225 // (7 * 25)
        move_idx = opponent_action_1225 % 25

        piece_action = from_y * 7 + from_x
        obs, _, terminated, truncated, info = self.env.step(piece_action)
        if terminated or truncated:
            return obs, -reward, terminated, truncated, info

        obs, reward, terminated, truncated, info = self.env.step(move_idx)
        self.opponent_moves += 1
        reward = -reward

        if not terminated and not truncated:
            if not np.any(self.env.action_masks()):
                terminated = True
                reward = -1.0
                info['agent_stuck'] = True

        info['agent_moves'] = self.agent_moves
        info['opponent_moves'] = self.opponent_moves
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.agent_moves = 0
        self.opponent_moves = 0
        return self.env.reset(**kwargs)

    def action_masks(self):
        return self.env.action_masks()