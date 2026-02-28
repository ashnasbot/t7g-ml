"""
Random opponent wrapper for Microscope environment.

Wraps any Microscope environment to add a random opponent that picks
uniformly from valid moves. Useful for:
- Early training stages (easier than minimax)
- Baseline comparisons
- Testing environment mechanics

Note: Compatible with two-stage action space. Opponent makes full moves
(piece selection + move selection) after agent completes their move.
"""
import gymnasium as gym
import numpy as np


class RandomOpponentWrapper(gym.Wrapper):
    """
    Wrapper that adds a random opponent to the environment.

    After the agent takes an action, the opponent randomly selects
    a valid move. Episodes terminate when either player has no moves.

    Args:
        env: The environment to wrap
        render_mode: Optional render mode (passed to wrapped env)
    """

    def __init__(self, env, render_mode=None):
        super().__init__(env)
        if render_mode is not None:
            self.env.render_mode = render_mode
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

    def reset(self, **kwargs):
        self.opponent_moves = 0
        self.agent_moves = 0
        return self.env.reset(**kwargs)

    def action_masks(self):
        """Forward action_masks to wrapped environment"""
        return self.env.action_masks()

    def step(self, action):
        # Agent makes move (could be stage 0: piece selection, or stage 1: move selection)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track agent moves only when they complete a full move (stage 0 -> stage 1 transition)
        if self._base_env.action_stage == 0:
            self.agent_moves += 1

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Only let opponent respond after agent completes full move (env is back at stage 0)
        if self._base_env.action_stage != 0:
            # Agent is mid-move (just selected piece), don't let opponent move yet
            return obs, reward, terminated, truncated, info

        # Random opponent responds (makes full two-stage move)
        # Stage 0: Select piece
        opponent_masks = self.env.action_masks()
        if not np.any(opponent_masks):
            # Opponent has no valid pieces to move - agent wins
            terminated = True
            reward = 1.0
            return obs, reward, terminated, truncated, info

        valid_pieces = np.where(opponent_masks)[0]
        piece_action = np.random.choice(valid_pieces)

        # Execute piece selection
        obs, _, terminated, truncated, info = self.env.step(piece_action)
        if terminated or truncated:
            return obs, -reward, terminated, truncated, info

        # Stage 1: Select move
        move_masks = self.env.action_masks()
        if not np.any(move_masks):
            # Selected piece has no valid moves (shouldn't happen with proper masking)
            terminated = True
            reward = 1.0
            return obs, reward, terminated, truncated, info

        valid_moves = np.where(move_masks)[0]
        move_action = np.random.choice(valid_moves)
        self.opponent_moves += 1

        # Execute move
        obs, reward, terminated, truncated, info = self.env.step(move_action)

        # Flip reward (opponent's reward is negative of agent's)
        reward = -reward

        # Check if agent can move
        if not terminated and not truncated:
            agent_masks = self.env.action_masks()
            if not np.any(agent_masks):
                # Agent has no valid pieces to move - opponent wins
                terminated = True
                reward = -1.0

        return obs, reward, terminated, truncated, info
