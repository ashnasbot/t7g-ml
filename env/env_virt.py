"""
Virtual Microscope environment with configurable reward functions.

Usage:
    from env.env_virt import MicroscopeEnv, MicroscopeEnvDense, MicroscopeEnvSimple, MicroscopeEnvStrategic

    # Base environment with original rewards
    env = MicroscopeEnv()

    # Or with specific reward function
    env = MicroscopeEnv(reward_fn='dense')  # Best for initial training
    env = MicroscopeEnv(reward_fn='simple')  # Good baseline
    env = MicroscopeEnv(reward_fn='strategic')  # Advanced training

    # Or use convenience classes
    env = MicroscopeEnvDense()  # Same as reward_fn='dense'
    env = MicroscopeEnvSimple()  # Same as reward_fn='simple'
    env = MicroscopeEnvStrategic()  # Same as reward_fn='strategic'
"""
import random

import gymnasium
from gymnasium import Env
import numpy
import numpy.typing as npt
from typing import Any, Optional

from lib.t7g import (
    calc_reward, show_board, action_to_move, action_masks, is_action_valid,
    apply_move, encode_action, new_board,
    BLUE, GREEN
)
from lib.reward_functions import get_reward_function


Board = npt.NDArray[numpy.bool_]    # shape (7, 7, 2)
Obs = npt.NDArray[numpy.float32]   # shape (7, 7, 4)
ActionMask = npt.NDArray[numpy.bool_]
StepResult = tuple[Obs, float, bool, bool, dict[str, Any]]


class MicroscopeEnv(Env):
    """
    Virtual Microscope board game environment.

    Supports both original and improved reward functions for better training.

    Args:
        reward_fn: Reward function to use:
            - 'simple': Linear rewards, good baseline
            - 'dense': High feedback, best for initial training
            - 'strategic': Multi-component rewards, advanced training
        render_mode: 'human' for visualization, None for training
        debug: If True, print per-move details (default: False)

    Note: Use MinimaxOpponentWrapper to add an opponent for evaluation/play
    """

    metadata = {"name": "Microscope Virt", "render_modes": [None, "human"]}

    def __init__(self, reward_fn: str = 'original', render_mode: Optional[str] = None, debug: bool = False) -> None:
        super().__init__()
        self.games_played = 0
        self.turn = True
        self.turns = 0
        self.turn_limit = 100
        self.render_mode = render_mode
        self.debug = debug

        # Reward function configuration
        self.reward_fn_name = reward_fn
        if reward_fn == 'original':
            self.reward_fn = calc_reward
        else:
            self.reward_fn = get_reward_function(reward_fn)
            print(f"Using reward function: {reward_fn}")

        # Flags
        self.masks = True
        self.random_start = True

        # Two-stage action tracking
        self.action_stage = 0  # 0 = select piece, 1 = select move
        self.selected_piece_pos: Optional[tuple[int, int]] = None  # (x, y) of selected piece

        # Observation: 7x7x4 (green pieces, blue pieces, turn indicator, selected piece)
        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(7, 7, 4), dtype=numpy.float32)
        # Action space: max(49 positions, 25 moves) = 49
        self.action_space = gymnasium.spaces.Discrete(49)

    def _get_obs(self) -> Obs:
        # Return board with turn indicator and selected piece as 3rd and 4th channels
        obs = numpy.zeros((7, 7, 4), dtype=numpy.float32)
        obs[:, :, 0:2] = self.game_grid  # Green and blue pieces
        obs[:, :, 2] = 1.0 if self.turn else 0.0  # Turn indicator (1=blue, 0=green)

        # Selected piece indicator (channel 3)
        if self.action_stage == 1 and self.selected_piece_pos is not None:
            x, y = self.selected_piece_pos
            obs[y, x, 3] = 1.0

        return obs

    def get_observation(self) -> bytes:
        # Return serialized observation with turn indicator
        return self._get_obs().tobytes()

    def move(self, action: int) -> bool:
        from_x, from_y, to_x, to_y, _ = action_to_move(action)

        if self.render_mode == "human" and self.debug:
            t = "B:" if self.turn else "G:"
            print(f"{t} [{from_x}, {from_y}] => [{to_x}, {to_y}]")

        if is_action_valid(self.game_grid, action, self.turn):
            self.game_grid = apply_move(self.game_grid, action, self.turn)
            return True
        return False

    def step(self, action: int) -> StepResult:
        """
        Two-stage action processing:
        - Stage 0: action is position index (0-48) selecting piece to move
        - Stage 1: action is move index (0-24) selecting how to move selected piece
        """
        if self.action_stage == 0:
            # STAGE 0: SELECT PIECE
            return self._step_select_piece(action)
        else:
            # STAGE 1: SELECT MOVE
            return self._step_select_move(action)

    def _step_select_piece(self, position_action: int) -> StepResult:
        """Stage 0: Select which piece to move"""
        # VALIDATION: position_action must be in range [0, 48]
        if position_action >= 49:
            raise ValueError(f"position_action {position_action} >= 49 (valid range: 0-48)")

        # Decode position (0-48 -> x, y)
        self.selected_piece_pos = (position_action % 7, position_action // 7)

        # Move to next stage
        self.action_stage = 1

        # Return observation with selected piece highlighted
        obs = self._get_obs()

        # No reward yet, not terminal, stage 1 pending
        return obs, 0.0, False, False, {'stage': 1, 'selected_pos': self.selected_piece_pos}

    def _step_select_move(self, move_action: int) -> StepResult:
        """Stage 1: Execute the move"""
        # VALIDATION: move_action must be in range [0, 24]
        if move_action >= 25:
            raise ValueError(f"move_action {move_action} >= 25 (valid range: 0-24)")

        # Decode move (0-24 -> dx, dy)
        dx = (move_action % 5) - 2  # -2, -1, 0, 1, 2
        dy = (move_action // 5) - 2  # -2, -1, 0, 1, 2

        # Combine with selected position to get full action
        x, y = self.selected_piece_pos
        full_action = encode_action(x, y, dx, dy)

        # Execute the move using original logic
        reward = 0
        terminated = False
        truncated = False
        self.turns += 1

        if not self.move(full_action):
            if self.masks:
                if numpy.any(action_masks(self.game_grid, self.turn)):
                    terminated = True
                    reward = -5
                    self.turns -= 1
            reward = -1

        # Flip turn for self-play (agent plays both colors)
        self.turn = not self.turn

        # Use configured reward function (expects 7x7x2 board)
        reward, terminated = self.reward_fn(self.game_grid, not self.turn)

        # Get observation with turn indicator
        observation = self._get_obs()

        if self.turns >= self.turn_limit - 1:
            reward = -1
            truncated = True

        if self.render_mode == 'human':
            # Render final board when episode ends
            if self.debug:
                print(f"Reward: {reward}")
            if terminated or truncated:
                show_board(self.game_grid)

        # Reset to stage 0 for next turn
        self.action_stage = 0
        self.selected_piece_pos = None

        return observation, reward, terminated, truncated, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)

        if not self.random_start or random.randint(0, 10) < 1:
            self.game_grid = new_board()
        else:
            self.game_grid = numpy.zeros((7, 7, 2), dtype=numpy.bool_)
            pieces = [BLUE, GREEN, BLUE, GREEN]
            for p in random.sample(range(49), 4):
                y = p // 7
                x = p % 7
                self.game_grid[y, x] = pieces.pop()

        self.turn = True
        self.turns = 0

        # Reset two-stage state
        self.action_stage = 0
        self.selected_piece_pos = None

        observation = self._get_obs()

        return observation, {}

    def action_masks(self) -> ActionMask:
        """Return valid actions for current stage"""
        if self.action_stage == 0:
            return self._action_masks_select_piece()
        else:
            return self._action_masks_select_move()

    def _action_masks_select_piece(self) -> ActionMask:
        """Stage 0: Mask positions with pieces of current color that have valid moves"""
        mask = numpy.zeros(49, dtype=bool)

        color = BLUE if self.turn else GREEN

        for y in range(7):
            for x in range(7):
                pos_idx = y * 7 + x
                # Position is valid if it has a piece of our color AND that piece has valid moves
                if numpy.array_equal(self.game_grid[y, x], color):
                    # Check if this piece has any valid moves
                    has_valid_move = False
                    for move_idx in range(25):
                        dx = (move_idx % 5) - 2
                        dy = (move_idx // 5) - 2
                        full_action = y * 7 * 25 + x * 25 + move_idx
                        if is_action_valid(self.game_grid, full_action, self.turn):
                            has_valid_move = True
                            break

                    if has_valid_move:
                        mask[pos_idx] = True

        return mask

    def _action_masks_select_move(self) -> ActionMask:
        """Stage 1: Mask valid moves from selected position (padded to 49 for consistency)"""
        mask = numpy.zeros(49, dtype=bool)  # Always 49 to match action space

        if self.selected_piece_pos is None:
            return mask  # Shouldn't happen

        x, y = self.selected_piece_pos

        # Put move masks in first 25 positions
        for move_idx in range(25):
            dx = (move_idx % 5) - 2
            dy = (move_idx // 5) - 2

            # Check if this move is valid
            full_action = encode_action(x, y, dx, dy)
            if is_action_valid(self.game_grid, full_action, self.turn):
                mask[move_idx] = True

        # Positions 25-48 are always False (padding)

        return mask

    def close(self) -> None:
        pass


# Convenience classes for specific reward functions

class MicroscopeEnvDense(MicroscopeEnv):
    """Environment with dense rewards - best for initial training"""
    metadata = {"name": "Microscope Virt (Dense)", "render_modes": [None, "human"]}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(reward_fn='dense', **kwargs)


class MicroscopeEnvSimple(MicroscopeEnv):
    """Environment with simple linear rewards - good baseline"""
    metadata = {"name": "Microscope Virt (Simple)", "render_modes": [None, "human"]}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(reward_fn='simple', **kwargs)


class MicroscopeEnvStrategic(MicroscopeEnv):
    """Environment with strategic rewards - for advanced training"""
    metadata = {"name": "Microscope Virt (Strategic)", "render_modes": [None, "human"]}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(reward_fn='strategic', **kwargs)


class MicroscopeEnvAggressive(MicroscopeEnv):
    """Environment with aggressive rewards - material + frontier, matches minimax objective"""
    metadata = {"name": "Microscope Virt (Aggressive)", "render_modes": [None, "human"]}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(reward_fn='aggressive', **kwargs)
