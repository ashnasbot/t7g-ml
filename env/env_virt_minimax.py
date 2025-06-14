import ctypes
from functools import cache
import pathlib
import random

import gymnasium
from gymnasium import Env
import numpy


from util.t7g import (
    calc_reward, show_board, action_to_move, action_masks, is_action_valid,
    BLUE, GREEN, CLEAR
)


class MicroscopeEnv(Env):

    def __init__(self):
        super().__init__()
        self.turn = True  # Blue = True, Green = False - Blue starts
        self.turns = 0
        self.turn_limit = 100
        self.minimax_depth = 5
        # Flags
        self.masks = True
        self.debug = False
        self.random_start = False

        libname = pathlib.Path().absolute() / "micro3.dll"
        self.scopelib = ctypes.CDLL(libname)
        self.scopelib.find_best_move.restype = ctypes.c_int
        self.scopelib.minimax.restype = ctypes.c_float
        self.scopelib.minimax.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]

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
            show_board(self.game_grid)

        if is_action_valid(self.game_grid, action, self.turn):
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
            return True
        elif self.debug:
            print("Not valid")
        return False

    @cache
    def find_best_move(self, board: bytes, depth: int, is_player_turn: bool) -> int:
        return self.scopelib.find_best_move(board, depth, is_player_turn)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        player_had_no_move = False

        if not self.move(action):
            if self.masks:
                if numpy.any(self.action_masks()):
                    print("==INVALID ACTION==")
                    from_x, from_y, to_x, to_y, _ = action_to_move(action)
                    t = "B:" if self.turn else "G:"
                    print(f"{t} [{from_x}, {from_y}]=> [{to_x}, {to_y}]")
                    reward = -5
                    show_board(self._get_obs()["board"])
                    print("==================")
                    terminated = True
                # else no valid moves remain, end game
            reward -= 1
            player_had_no_move = True
        self.turns += 1

        observation = self._get_obs()

        self.turn = not self.turn
        input = observation["board"].tobytes()
        opponent_move = self.find_best_move(input, self.minimax_depth, self.turn)

        if self.move(opponent_move):
            self.turns += 1
        elif player_had_no_move:
            terminated = True

        self.turn = not self.turn

        observation = self._get_obs()

        # Round over - how did we do?
        reward, terminated = calc_reward(observation["board"], self.turn)
        # reward -= self.turns / 100

        if self.debug:
            print("Reward:", reward)

            if terminated:
                show_board(observation["board"])

        if self.turns >= self.turn_limit - 2:  # There are 2 turns per step
            reward = -1
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

    def action_masks(self):
        return action_masks(self.game_grid, self.turn)

    def close(self):
        pass
