import time

import gymnasium
from gymnasium import Env
import numpy
from PIL import Image
from term_image.image import AutoImage

BLUE = [0, 0, 1]
GREEN = [0, 1, 0]
CLEAR = [0, 0, 0]


def count_cells(observation):
    b = observation.reshape(-1, 3)
    blue = 0
    green = 0
    for triplet in b:
        if triplet[2] == 1:
            blue += 1
        elif triplet[1] == 1:
            green += 1

    return blue, green


def show_board(observation):
    img_arr = numpy.copy(observation)
    img_arr[img_arr == 1] = 255
    img = Image.fromarray(img_arr, 'RGB')
    #img = img.resize((100, 100), Image.Resampling.BOX)
    AutoImage(img).draw(h_align="left")
    time.sleep(0.1)


class MicroscopeEnv(Env):

    def __init__(self):
        super().__init__()
        self.proc = None
        self.loaded = False
        self.games_played = 0
        self.turn = True  # Blue = True, Green = False - Blue starts

        self.blue_cells = None
        self.green_cells = None

        self.observation_space = gymnasium.spaces.Dict({
            "board": gymnasium.spaces.Box(0, 1, shape=(7, 7, 3), dtype=bool),
            "turn": gymnasium.spaces.Discrete(2)
        })
        self.action_space = gymnasium.spaces.Discrete(49 * 25)

    def _get_obs(self):
        return {
            "board": self.game_grid,
            "turn": self.turn
        }

    def step(self, action):
        reward = 0
        terminated = False

        if self.turn:
            player_cell = BLUE
            opponent_cell = GREEN
        else:
            player_cell = GREEN
            opponent_cell = BLUE
        t = "B:" if self.turn else "G:"

        piece = action // 25
        move = action % 25
        from_x = piece % 7
        from_y = piece // 7
        mv_x = (move % 5) - 2
        mv_y = (move // 5) - 2

        to_x = from_x + mv_x
        to_y = from_y + mv_y

        #print(f"{t} [{from_x}, {from_y}]=> [{to_x}, {to_y}]")
        if self.is_action_valid(action):
            if mv_x == 2 or mv_y == 2:
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

        else:
            terminated = True
            if numpy.any(self.action_masks()):
                print("==INVALID ACTION==")
                print(f"{t} [{from_x}, {from_y}]=> [{to_x}, {to_y}]")
                reward = -5
                show_board(self._get_obs()["board"])
                print("==================")
            # else no valid moves remain, end game

        observation = self._get_obs()

        # Round over - how did we do?
        new_blue, new_green = count_cells(observation["board"])

        if self.turn:
            if new_blue == 0:
                # We have lost
                reward = -5
                self.games_played += 1
                terminated = True
            elif new_green == 0:
                # We have won!
                reward = 500
                self.games_played += 1
                terminated = True
            else:
                if new_green < self.green_cells:
                    reward += self.green_cells - new_green
                if not terminated:
                    reward += new_blue - self.blue_cells
        else:
            # TODO reduce with above
            if new_green == 0:
                # We have lost
                reward = -5
                self.games_played += 1
                terminated = True
            elif new_blue == 0:
                # We have won!
                reward = 500
                self.games_played += 1
                terminated = True
            else:
                if new_blue < self.blue_cells:
                    reward += self.blue_cells - new_blue
                if not terminated:
                    reward += new_green - self.green_cells

        if new_blue + new_green == 49:
            terminated = True
            # board is full, see who won
            if new_blue > new_green:
                # Blue win
                if self.turn:
                    reward = 20
                else:
                    reward = -20
            else:
                # Green win
                if self.turn:
                    reward = -20
                else:
                    reward = 20

        if terminated:
            new_blue = 2
            new_green = 2

        self.blue_cells = new_blue
        self.green_cells = new_green

        #show_board(observation["board"])
        self.turn = not self.turn
        observation["turn"] = self.turn

        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_grid = numpy.zeros((7,7,3), dtype=numpy.uint8)
        self.game_grid[0, 0] = BLUE
        self.game_grid[0, 6] = GREEN
        self.game_grid[6, 0] = GREEN
        self.game_grid[6, 6] = BLUE

        observation = self._get_obs()
        self.blue_cells, self.green_cells = count_cells(observation["board"])
        self.turn = True  # Blue to start
        observation["turn"] = self.turn

        return observation, None

    def is_action_valid(self, action):
        if self.turn:
            player_cell = BLUE
        else:
            player_cell = GREEN

        piece = action // 25
        move = action % 25
        from_x = piece % 7
        from_y = piece // 7
        mv_x = (move % 5) - 2
        mv_y = (move // 5) - 2

        to_x = from_x + mv_x
        to_y = from_y + mv_y
        if numpy.array_equal(self.game_grid[from_y][from_x], player_cell):
            # We are trying to move our own piece

            if 0 <= to_x < 7 and\
               0 <= to_y < 7:

                if not any(self.game_grid[to_y][to_x]):
                    # The Dest is free
                    return True
        return False

    def action_masks(self):

        if self.turn:
            player_cell = BLUE
        else:
            player_cell = GREEN

        actions = numpy.zeros((49, 25), dtype=numpy.int8)

        for y in range(len(self.game_grid)):
            for x in range(len(self.game_grid[y])):
                if numpy.array_equal(self.game_grid[y, x], player_cell):
                    # We're moving our own piece
                    valid_moves = numpy.zeros((25), dtype=numpy.int8)
                    for v in range(5):
                        for u in range(5):
                            to_x = x + u - 2
                            to_y = y + v - 2

                            if 0 <= to_x < 7 and\
                               0 <= to_y < 7:
                                if not any(self.game_grid[to_y][to_x]):
                                    valid_moves[v * 5 + u] = 1

                    actions[y * 7 + x] = valid_moves

        return actions.flatten()

    def close(self):
        pass
