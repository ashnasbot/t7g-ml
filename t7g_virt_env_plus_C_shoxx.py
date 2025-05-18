import ctypes
from functools import cache
import pathlib
import random

import gymnasium
from gymnasium import Env
import numpy


from t7g_utils import (
    calc_reward, show_board, action_to_move,
    BLUE, GREEN, CLEAR
)


class MicroscopeEnv(Env):

    def __init__(self):
        super().__init__()
        self.turn = True  # Blue = True, Green = False - Blue starts
        self.turns = 0
        self.turn_limit = 100
        # Flags
        self.masks = False
        self.debug = False
        self.random_start = False

        libname = pathlib.Path().absolute() / "micro3.dll"
        self.scopelib = ctypes.CDLL(libname)
        self.scopelib.find_best_move.restype = ctypes.c_int

        self.observation_space = gymnasium.spaces.Dict({
            "board": gymnasium.spaces.Box(0, 1, shape=(7, 7, 2), dtype=numpy.bool),
            "turn": gymnasium.spaces.Discrete(2),
            "turns": gymnasium.spaces.Discrete(self.turn_limit),
        })
        self.action_space = gymnasium.spaces.Discrete(49 * 25)

    def _get_obs(self):
        return {
            "board": self.game_grid,
            "turn": self.turn,
            "turns": self.turns
        }

    def move(self, action):
        if self.turn:
            player_cell = BLUE
            opponent_cell = GREEN
        else:
            player_cell = GREEN
            opponent_cell = BLUE

        from_x, from_y, to_x, to_y, jump = action_to_move(action)

        if self.debug:
            t = "B:" if self.turn else "G:"
            print(f"{t} [{from_x}, {from_y}]=> [{to_x}, {to_y}]")

        if self.is_action_valid(action):
            if jump:
                self.game_grid[from_y, from_x] = CLEAR
            self.game_grid[to_y, to_x] = player_cell

            for x, y in numpy.ndindex((3, 3)):
                x2 = to_x - 1 + x
                y2 = to_y - 1 + y
                if 0 <= x2 < 7 and\
                   0 <= y2 < 7:

                    if numpy.array_equal(self.game_grid[y2, x2], opponent_cell):
                        self.game_grid[y2, x2] = player_cell
        elif self.debug:
            print("Not valid")

    @cache
    def find_best_move(self, board: bytes, depth: int, turn: bool) -> int:
        return self.scopelib.find_best_move(board, depth, turn)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if not self.move(action):
            if self.masks:
                terminated = True
                if numpy.any(self.action_masks()):
                    print("==INVALID ACTION==")
                    # t = "B:" if self.turn else "G:"
                    # print(f"{t} [{from_x}, {from_y}]=> [{to_x}, {to_y}]")
                    reward = -5
                    show_board(self._get_obs()["board"])
                    print("==================")
                # else no valid moves remain, end game
        self.turns += 1

        observation = self._get_obs()

        self.turn = not self.turn
        input = observation["board"].tobytes()
        opponent_move = self.find_best_move(input, 3, True)

        if opponent_move != 1225:
            if self.move(opponent_move):
                self.turns += 1
        else:
            show_board(self._get_obs()["board"])
            print("INVALID GREEEEEN")
        self.turn = not self.turn

        observation = self._get_obs()

        # Round over - how did we do?
        reward, terminated = calc_reward(observation["board"], self.turn)

        if self.debug:
            print("Reward:", reward)

            if terminated:
                show_board(observation["board"])

        if self.turns >= self.turn_limit - 2:  # There are 2 turns per step
            truncated = True

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_grid = numpy.zeros((7, 7, 2), dtype=numpy.bool)
        pieces = [BLUE, GREEN, BLUE, GREEN]
        if not self.random_start or random.randint(0, 10) < 1:
            self.game_grid[0, 0] = BLUE
            self.game_grid[0, 6] = GREEN
            self.game_grid[6, 0] = GREEN
            self.game_grid[6, 6] = BLUE

        else:
            for p in random.sample(range(49), 4):
                y = p // 7
                x = p % 7
                self.game_grid[y, x] = pieces.pop()

        observation = self._get_obs()
        self.turn = True  # Blue to start
        observation["turn"] = self.turn

        self.turns = 0

        return observation, {}

    def is_action_valid(self, action):
        if self.turn:
            player_cell = BLUE
        else:
            player_cell = GREEN

        from_x, from_y, to_x, to_y, _ = action_to_move(action)
        if numpy.array_equal(self.game_grid[from_y, from_x], player_cell):
            # We are trying to move our own piece

            if 0 <= to_x < 7 and\
               0 <= to_y < 7:

                if not any(self.game_grid[to_y, to_x]):
                    # The Dest is free
                    return True
        return False

    def action_masks(self):

        if self.turn:
            player_cell = BLUE
        else:
            player_cell = GREEN

        actions = numpy.zeros((7, 7, 5, 5), dtype=numpy.bool)

        for y, x in numpy.ndindex((7, 7)):
            if numpy.array_equal(self.game_grid[y, x], player_cell):
                # We're moving our own piece
                for u, v in numpy.ndindex((5, 5)):
                    to_x = x + u - 2
                    to_y = y + v - 2

                    if 0 <= to_x < 7 and\
                       0 <= to_y < 7:
                        if not any(self.game_grid[to_y][to_x]):
                            actions[y, x, v, u] = 1

        return actions.flatten()

    def close(self):
        pass
