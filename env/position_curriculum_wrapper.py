"""
Position Curriculum Wrapper for Microscope Environment

Wraps the base environment to reset from curriculum-sampled positions
instead of always starting from the initial game state.
"""
import gymnasium as gym
from env.position_curriculum import PositionCurriculum


class PositionCurriculumWrapper(gym.Wrapper):
    """
    Resets environment from curriculum-based positions.

    This accelerates learning by exposing the agent to diverse game phases
    without requiring it to play through full games every episode.

    Usage:
        env = MicroscopeEnv()
        env = PositionCurriculumWrapper(env)

        # During training, pass model timesteps to track curriculum progress
        obs, info = env.reset(options={'timestep': model.num_timesteps})
    """

    def __init__(self, env, curriculum=None):
        """
        Args:
            env: Base Microscope environment (must wrap MicroscopeEnv directly)
            curriculum: Optional PositionCurriculum instance (creates default if None)

        IMPORTANT: This wrapper must be the innermost wrapper, directly around
        MicroscopeEnv, so that outer wrappers (SymmetryAugmentationWrapper etc.)
        can process its observations correctly.

        Correct order:
            env = MicroscopeEnv()
            env = PositionCurriculumWrapper(env)   # innermost
            env = SymmetryAugmentationWrapper(env)  # processes curriculum obs
            env = OpponentWrapper(env)
            ...
        """
        super().__init__(env)
        self.curriculum = curriculum or PositionCurriculum()
        self.current_timestep = 0

    def reset(self, seed=None, options=None):
        """
        Reset environment to a curriculum-sampled position.

        Does a normal reset first, then overrides board state with a
        curriculum-sampled position.
        """
        # Get timestep from options if provided
        if options and 'timestep' in options:
            self.current_timestep = options['timestep']

        # Normal reset first (initializes base env state)
        obs, info = self.env.reset(seed=seed, options=options)

        # Sample position from curriculum and override base env state
        board, turn = self.curriculum.sample_position(self.current_timestep)

        self.env.game_grid = board
        self.env.turn = turn
        self.env.turns = 0
        self.env.action_stage = 0
        self.env.selected_piece_pos = None

        # Re-get observation with the new board state
        observation = self.env._get_obs()

        return observation, {}

    def set_timestep(self, timestep):
        """Manually set current timestep for curriculum progression"""
        self.current_timestep = timestep

    def action_masks(self):
        """Forward action_masks to wrapped environment"""
        return self.env.action_masks()


class TimestepTrackingCallback:
    """
    Callback to pass timesteps to PositionCurriculumWrapper during training.

    Use with stable-baselines3 PPO training:
        callback = TimestepTrackingCallback(env)
        model.learn(total_timesteps=1_000_000, callback=callback)
    """

    def __init__(self, env):
        """
        Args:
            env: The wrapped environment (should have PositionCurriculumWrapper)
        """
        self.env = env

        # Find the PositionCurriculumWrapper in the wrapper stack
        self.curriculum_wrapper = self._find_curriculum_wrapper(env)

        if self.curriculum_wrapper is None:
            raise ValueError(
                "No PositionCurriculumWrapper found in environment stack. "
                "Make sure to wrap your environment with PositionCurriculumWrapper."
            )

    def _find_curriculum_wrapper(self, env):
        """Walk the wrapper stack to find PositionCurriculumWrapper"""
        current = env
        while current is not None:
            if isinstance(current, PositionCurriculumWrapper):
                return current
            current = getattr(current, 'env', None)
        return None

    def __call__(self, locals_dict, globals_dict):
        """
        Called by stable-baselines3 during training.

        Updates the curriculum wrapper with current timestep.
        """
        # Get current model timesteps
        if 'self' in locals_dict:
            model = locals_dict['self']
            timestep = model.num_timesteps

            # Update curriculum wrapper
            self.curriculum_wrapper.set_timestep(timestep)

        # Return True to continue training
        return True
