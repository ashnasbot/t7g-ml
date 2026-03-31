"""
Virtual Microscope environment for testing MCTS agents without the real game.

The agent always plays as Blue. After each Blue move, the opponent (minimax
or random) plays as Green automatically, so step() always returns a board
that is ready for the next Blue move.

Interface:
    env = MicroscopeVirtEnv(opponent='micro4', depth=2)
    board, _ = env.reset()       # board is (7,7,2) bool_
    while True:
        action = ...             # 1225-dim action chosen by MCTS
        board, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""
import random

import numpy
import numpy.typing as npt

from lib.t7g import (
    new_board, apply_move, check_terminal, action_masks,
    is_action_valid, find_best_move, count_cells,
)

Board = npt.NDArray[numpy.bool_]


class MicroscopeVirtEnv:
    """
    Virtual Microscope environment with a built-in opponent.

    The MCTS agent always plays as Blue. After each Blue move the opponent
    plays as Green, so step() always returns with turn=True ready for Blue.

    Args:
        opponent:   'micro3', 'micro4', 'stauf', or 'random'
        depth:      minimax search depth (ignored for 'random')
        turn_limit: maximum half-moves before truncation
    """

    def __init__(
        self,
        opponent: str = 'micro4',
        depth: int = 2,
        turn_limit: int = 200,
    ) -> None:
        self.opponent = opponent
        self.depth = depth
        self.turn_limit = turn_limit
        self.game_grid: Board = new_board()
        self.turn = True   # Agent is always Blue
        self.turns = 0

    def _opponent_move(self) -> None:
        """Play one Green move (random fallback if no legal move returned)."""
        legal = numpy.where(action_masks(self.game_grid, False))[0]
        if len(legal) == 0:
            return
        if self.opponent == 'random':
            action = int(numpy.random.choice(legal))
        else:
            action = find_best_move(
                self.game_grid.tobytes(), self.depth, False,
                engine=self.opponent, move_count=-1,
            )
            if action < 0 or action >= 1225 or action not in legal:
                action = int(numpy.random.choice(legal))
        self.game_grid = apply_move(self.game_grid, action, False)

    def step(self, action: int) -> tuple:
        """
        Apply Blue's 1225-dim action, then play Green's response.

        Returns (board, reward, terminated, truncated, info).
        board:   (7,7,2) bool_, ready for Blue's next move
        reward:  +1.0 Blue wins / -1.0 Green wins / 0.0 in progress or draw
        """
        if not is_action_valid(self.game_grid, action, True):
            return self.game_grid.copy(), -1.0, False, False, {'invalid': True}

        self.game_grid = apply_move(self.game_grid, action, True)
        self.turns += 1

        # Check terminal after Blue's move
        is_terminal, terminal_value = check_terminal(self.game_grid, False)
        if is_terminal:
            assert terminal_value is not None
            # terminal_value is from Green's perspective; flip to Blue's
            return self.game_grid.copy(), -float(terminal_value), True, False, {}

        if self.turns >= self.turn_limit:
            blue, green = count_cells(self.game_grid)
            reward = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            return self.game_grid.copy(), reward, False, True, {}

        # Green's response
        self._opponent_move()

        # Check terminal after Green's move
        is_terminal, terminal_value = check_terminal(self.game_grid, True)
        if is_terminal:
            assert terminal_value is not None
            return self.game_grid.copy(), float(terminal_value), True, False, {}

        if self.turns >= self.turn_limit:
            blue, green = count_cells(self.game_grid)
            reward = 1.0 if blue > green else (-1.0 if green > blue else 0.0)
            return self.game_grid.copy(), reward, False, True, {}

        return self.game_grid.copy(), 0.0, False, False, {}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        if seed is not None:
            numpy.random.seed(seed)
            random.seed(seed)
        self.game_grid = new_board()
        self.turn = True
        self.turns = 0
        return self.game_grid.copy(), {}

    def close(self) -> None:
        pass
