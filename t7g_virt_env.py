import random

import gymnasium
from gymnasium import Env
import numpy

from t7g_utils import (
    count_cells, show_board, action_to_move,
    BLUE, GREEN, CLEAR
)


class MicroscopeEnv(Env):

    def __init__(self):
        super().__init__()
        self.games_played = 0
        self.turn = True  # Blue = True, Green = False - Blue starts
        self.turns = 0
        self.turn_limit = 50
        # Flags
        self.masks = False
        self.debug = False
        self.random_start = False
        self.cum_reward = 0

        self.blue_cells = None
        self.green_cells = None

        self.observation_space = gymnasium.spaces.Dict({
            "board": gymnasium.spaces.Box(0, 1, shape=(7, 7, 2), dtype=bool),
            "turn": gymnasium.spaces.Discrete(2),
            "turns": gymnasium.spaces.Discrete(self.turn_limit)
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

            for x in range(3):
                for y in range(3):
                    x2 = to_x - 1 + x
                    y2 = to_y - 1 + y
                    if 0 <= x2 < 7 and\
                       0 <= y2 < 7:

                        if numpy.array_equal(self.game_grid[y2, x2], opponent_cell):
                            self.game_grid[y2, x2] = player_cell
            return True
        return False

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
            # we've not made a move, don't count it
            self.turns -= 1
            reward = -10

        observation = self._get_obs()

        # Round over - how did we do?
        # TODO: Evaluate in a utils function
        new_blue, new_green = count_cells(observation["board"])

        if self.turn:
            player_cells = new_blue
            opponent_cells = new_green
            prev_player_cells = self.blue_cells
            prev_opponent_cells = self.green_cells
        else:
            player_cells = new_green
            opponent_cells = new_blue
            prev_player_cells = self.green_cells
            prev_opponent_cells = self.blue_cells

        if player_cells == 0:
            # We have lost
            reward = -100
            self.games_played += 1
            terminated = True
        elif opponent_cells == 0:
            # We have won!
            reward = 2 * (self.turn_limit - self.turns)
            self.games_played += 1
            terminated = True
        else:
            cell_diff = (
                (player_cells - prev_player_cells) -
                (opponent_cells - prev_opponent_cells)
            )

            if terminated:
                reward = (player_cells - opponent_cells) * 10
            else:
                reward += cell_diff * 2

        # Switch to the other player
        self.turn = not self.turn
        observation["turn"] = self.turn
        self.turns += 1
        observation["turns"] = self.turns

        reward -= (self.turns / 7)
        if self.debug:
            print("Reward:", reward)

        self.cum_reward += reward
        if terminated and self.debug:
            show_board(observation["board"])
            print("Total reward:", self.cum_reward)
        if self.turns >= self.turn_limit - 1:
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
        self.blue_cells, self.green_cells = count_cells(observation["board"])
        self.turn = True  # Blue to start
        observation["turn"] = self.turn

        self.cum_reward = 0
        self.turns = 0

        return observation, None

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

        player_cell = BLUE if self.turn else GREEN

        actions = numpy.zeros((49, 25), dtype=numpy.bool)

        # TODO: This could be so much faster if vectorised
        for y, x in numpy.ndindex((7, 7)):
            if numpy.array_equal(self.game_grid[y, x], player_cell):
                # We're moving our own piece
                valid_moves = numpy.zeros((25), dtype=numpy.bool)
                for u, v in numpy.ndindex((5, 5)):
                    to_x = x + u - 2
                    to_y = y + v - 2

                    if 0 <= to_x < 7 and\
                        0 <= to_y < 7:
                        if not any(self.game_grid[to_y, to_x]):
                            valid_moves[v * 5 + u] = 1

                actions[y * 7 + x] = valid_moves

        return actions.flatten()

    def close(self):
        pass
