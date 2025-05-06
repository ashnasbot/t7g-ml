import gymnasium
from gymnasium import Env
import numpy


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


class MicroscopeEnv(Env):

    def __init__(self, cell_id):
        super().__init__()
        self.proc = None
        self.loaded = False
        self.games_played = 0
        self.cell_id = cell_id
        self.opponent_id = numpy.array([0,1, 1]).subtract(cell_id)

        self.blue_cells = None
        self.green_cells = None

        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(7, 7, 3), dtype=int)
        self.action_space = gymnasium.spaces.Discrete(49 * 25)

    def _get_obs(self):
        return self.game_grid

    def step(self, action):
        reward = 0
        terminated = False

        piece = action // 25
        move = action % 25
        from_x = piece % 7
        from_y = piece // 7
        mv_x = (move % 5) - 2
        mv_y = (move // 5) - 2

        to_x = from_x + mv_x
        to_y = from_y + mv_y

        print(f"[{from_x}, {from_y}] => [{to_x}, {to_y}]: ", end="")
        if self.is_action_valid(action):
            self.game_grid[from_x, from_y] = [0, 0, 0]
            self.game_grid[from_x, from_y] = self.cell_id

            for x in range(3):
                for y in range(3):
                    if self.game_grid[to_x - 1 + x, to_y - 1 + y] == self.opponent_id:
                        self.game_grid[to_x - 1 + x, to_y - 1 + y] == self.cell_id

        else:
            print("==INVALID ACTION==")
            print(f"[{from_x}, {from_y}]=> [{to_x}, {to_y}]")
            print("==================")
            terminated = True
            reward = -5

        observation = self._get_obs()

        # Round over - how did we do?
        new_blue, new_green = count_cells(observation)
        print(f"(B: {new_blue}, G: {new_green})", end="")

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

        if terminated:
            new_blue = 2
            new_green = 2

        self.blue_cells = new_blue
        self.green_cells = new_green

        termstr = f"Lost - games played: {self.games_played}" if terminated else ""
        print(f" {reward:>2} {termstr}")

        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_grid = numpy.zeros((7,7,3))

        observation = self._get_obs()
        self.blue_cells, self.green_cells = count_cells(observation)

        return observation, None

    def is_action_valid(self, action):
        piece = action // 25
        move = action % 25
        from_x = piece % 7
        from_y = piece // 7
        mv_x = (move % 5) - 2
        mv_y = (move // 5) - 2

        to_x = from_x + mv_x
        to_y = from_y + mv_y
        if self.game_grid[from_y][from_x][2]:
            # We are trying to move a Blue piece

            if 0 <= to_x < 7 and\
               0 <= to_y < 7:

                if not any(self.game_grid[to_y][to_x]):
                    # The Dest is free
                    return True
        return False

    def action_masks(self):

        actions = numpy.zeros((49, 25), dtype=numpy.int8)

        for y in range(len(self.game_grid)):
            for x in range(len(self.game_grid[y])):
                if self.game_grid[y, x][2]:
                    # We're moving a blue piece
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
