"""
Wrapper that adds minimax opponent to any Microscope environment.

Usage:
    from env.minimax_wrapper import MinimaxOpponentWrapper
    from env.env_virt import MicroscopeEnvSimple

    # Wrap any environment to add minimax opponent
    env = MinimaxOpponentWrapper(MicroscopeEnvSimple(), depth=3)

Note: Compatible with two-stage action space. Minimax returns moves in
old 1225-action space, which are converted to two-stage actions.
"""
import numpy as np
import gymnasium as gym
from lib.t7g import find_best_move, show_board


class MinimaxOpponentWrapper(gym.Wrapper):
    """
    Wraps a Microscope environment to automatically play minimax opponent moves.

    After the agent takes an action, the minimax opponent responds automatically.
    This allows evaluation against a strong baseline opponent.

    Args:
        env: The Microscope environment to wrap
        depth: Minimax search depth (1-5). Higher = stronger but slower
               - depth 1: ~instant, weak
               - depth 3: ~0.5s, competitive (default)
               - depth 5: ~10s, strong
    """

    def __init__(self, env, depth=3):
        super().__init__(env)
        self.depth = depth
        self.opponent_moves = 0
        self.agent_moves = 0

        # Find the base MicroscopeEnv through any wrapper stack
        # (gymnasium __getattr__ forwarding is unreliable across wrappers)
        self._base_env = self._find_base_env(env)

    def _find_base_env(self, env):
        """Walk wrapper stack to find the base env with game_grid"""
        current = env
        while hasattr(current, 'env'):
            if hasattr(current, 'game_grid'):
                return current
            current = current.env
        return current

    def step(self, action):
        """
        Execute agent move, then minimax opponent move.

        Returns observation, reward, terminated, truncated, info from agent's perspective.
        """
        # Agent makes move (could be stage 0 or stage 1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track agent moves only when they complete a full move
        if self._base_env.action_stage == 0:
            self.agent_moves += 1

        # If game ended after agent move, return immediately
        if terminated or truncated:
            info['agent_moves'] = self.agent_moves
            info['opponent_moves'] = self.opponent_moves
            return obs, reward, terminated, truncated, info

        # Only let opponent respond after agent completes full move (env is back at stage 0)
        if self._base_env.action_stage != 0:
            # Agent is mid-move (just selected piece), don't let opponent move yet
            return obs, reward, terminated, truncated, info

        # Minimax opponent responds (makes full two-stage move)
        board_bytes = self._base_env.game_grid.tobytes()

        # Find best move for green (as_blue=False) - returns action in old 1225 space
        opponent_action_1225 = find_best_move(board_bytes, self.depth, False)

        # Check if opponent has valid move
        if opponent_action_1225 in [-1, 1225]:
            # Opponent has no moves - game over, agent wins
            terminated = True
            reward = 1.0
            info['agent_moves'] = self.agent_moves
            info['opponent_moves'] = self.opponent_moves
            info['opponent_stuck'] = True
            if self._base_env.render_mode == 'human':
                show_board(self._base_env.game_grid)
            return obs, reward, terminated, truncated, info

        # Convert 1225-action to two-stage actions
        # Action format: y * 7 * 25 + x * 25 + move_idx
        from_x = (opponent_action_1225 // 25) % 7
        from_y = opponent_action_1225 // (7 * 25)
        move_idx = opponent_action_1225 % 25

        # Stage 0: Select piece
        piece_action = from_y * 7 + from_x
        obs, _, terminated, truncated, info = self.env.step(piece_action)
        if terminated or truncated:
            return obs, -reward, terminated, truncated, info

        # Stage 1: Select move
        obs, reward, terminated, truncated, info = self.env.step(move_idx)
        self.opponent_moves += 1

        # Flip reward (opponent's reward is negative of agent's)
        reward = -reward

        # Check if agent can move
        if not terminated and not truncated:
            agent_masks = self.env.action_masks()
            if not np.any(agent_masks):
                # Agent has no valid pieces to move - opponent wins
                terminated = True
                reward = -1.0
                info['agent_stuck'] = True

        info['agent_moves'] = self.agent_moves
        info['opponent_moves'] = self.opponent_moves

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and counters"""
        self.agent_moves = 0
        self.opponent_moves = 0
        return self.env.reset(**kwargs)

    def action_masks(self):
        """Forward action_masks to wrapped environment"""
        return self.env.action_masks()
