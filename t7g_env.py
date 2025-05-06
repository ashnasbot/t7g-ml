import subprocess
import time

import gymnasium
from gymnasium import Env
import numpy
from PIL import ImageGrab, Image, ImageChops, ImageEnhance, ImageOps
import win32gui
import win32api
import win32con


def get_game():
    toplist, winlist = [], []

    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
    win32gui.EnumWindows(enum_cb, toplist)

    game_exe = [(hwnd, title) for hwnd, title in winlist if 'the 7th guest' in title.lower()]
    if not game_exe:
        return None
    game_exe = game_exe[0]
    return game_exe[0]


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((5, 5)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -40)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def click(left_or_right, x, y, delay):
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    if left_or_right == 'left':
        win32api.SetCursorPos((x, y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
    elif left_or_right == 'right':
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
    time.sleep(delay)


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

# TODO:
# Discover valid moves and mask


class MicroscopeEnv(Env):

    def __init__(self):
        super().__init__()
        self.proc = None
        self.loaded = False
        self.games_played = 0
        self.scale = 2.0
        # mouse posttions of each grid cell
        self.grid = numpy.zeros((7, 7, 2), dtype=int)
        self.safe_mouse_pos = (0, 0)

        self.blue_cells = None
        self.green_cells = None

        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(7, 7, 3), dtype=int)
        self.action_space = gymnasium.spaces.Discrete(49 * 25)

    def _get_obs(self):
        hwnd = get_game()

        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1)

        win32api.SetCursorPos(self.safe_mouse_pos)

        rect = win32gui.GetClientRect(hwnd)
        rect = list(rect)
        rect[0] = int(rect[0] * self.scale)
        rect[1] = int(rect[1] * self.scale)
        rect[2] = int(rect[2] * self.scale)
        rect[3] = int(rect[3] * self.scale)
        rect = tuple(rect)
        client_pos = win32gui.ClientToScreen(hwnd, rect[:2])
        bbox = (client_pos[0], client_pos[1], client_pos[0] + rect[2], client_pos[1] + rect[3])
        img = ImageGrab.grab(bbox)

        # Trim the edge of the client due to the stupid rounded borders / cursor
        img = img.crop((img.width//7, img.height//7, img.width-img.width//7, img.height-img.height//7))

        # The inner background isnt actually black - make it
        img_arr = numpy.array(img)
        img_arr[img_arr < 10] = 0
        img = Image.fromarray(img_arr, 'RGB')

        # Crop the image to the bounding box of the game grid
        # (everything else should be black now)
        bbox = img.getbbox()
        img = img.crop(bbox)
        img = trim(img)

        # remove red
        red = Image.new(img.mode, img.size, (255, 0, 0))
        img = ImageChops.subtract(img, red)

        im3 = ImageEnhance.Brightness(img)
        img = im3.enhance(1.5)

        # crush the noise
        img = ImageOps.posterize(img, 3)

        # Reduce to the 7x7 Grid
        img.thumbnail((7, 7), Image.Resampling.BOX)

        # The resampling introduces noise, crush it again
        img = ImageOps.posterize(img, 3)

        # The only remaining non-black pixels are our green & blue blobs
        # jack the contrast up to saturate them
        im3 = ImageEnhance.Brightness(img)
        img = im3.enhance(255.0)

        # At this point we either have 255 in a channel or 0
        # Convert to (0, 1, 0) or (0, 0, 1) (Green / Blue)
        gamegrid = numpy.array(img)
        gamegrid[gamegrid > 20] = 1

        self.game_grid = gamegrid

        return gamegrid

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

            click_x, click_y = self.grid[from_x][from_y]
            click("left", click_x, click_y, 0)
            click("right", click_x, click_y, 0.1)

            click_x, click_y = self.grid[to_x][to_y]
            click("left", click_x, click_y, 0)
            click("right", click_x, click_y, 0.1)
        else:
            print("==INVALID ACTION==")
            print(f"[{from_x}, {from_y}]=> [{to_x}, {to_y}]")
            print("==================")
            terminated = True
            reward = -5

        # end of turn, wait for green
        click("right", *self.grid[0, 0], 0)

        # Detect when it's our go again
        times = 0
        while True:
            observation = self._get_obs()
            time.sleep(0.1)
            observation2 = self._get_obs()
            if numpy.array_equal(observation, observation2):
                times += 1
            else:
                times = 0

            if times == 3:
                break

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
                if self.green_cells - new_green > 6 and new_green == 2:
                    print("Broken, we must have lost")
                    reward = 0
                    terminated = True
                    new_blue = 2
                else:
                    reward += self.green_cells - new_green
            if not terminated:
                reward += new_blue - self.blue_cells

        if terminated:
            new_blue = 2
            new_green = 2
            for _ in range(8):
                click("right", *self.grid[0, 0], 0)

        self.blue_cells = new_blue
        self.green_cells = new_green

        termstr = f"Lost - games played: {self.games_played}" if terminated else ""
        print(f" {reward:>2} {termstr}")

        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.proc:
            # self.proc.kill()
            time.sleep(1)
        hwnd = get_game()

        load = None
        if hwnd is None:
            load = True

        if load:
            self.proc = subprocess.Popen(["C:\\Program Files\\ScummVM\\scummvm.exe", "--save-slot=18", "t7g"])
            time.sleep(27)
        hwnd = get_game()
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.2)

        if load:
            pos = win32gui.GetWindowRect(hwnd)
            # Click the bench
            x = pos[0] + (pos[2] - pos[0])//3
            y = pos[1] + (pos[3] - pos[1])//2
            click("left", x, y, 0.1)
            click("right", x, y, 0.3)
            click("right", x, y, 0.8)

            # Click the microscope
            x = int(pos[0] + (pos[2] - pos[0])//1.5)
            y = int(pos[1] + (pos[3] - pos[1])//2)
            click("left", x, y, 0.3)
            click("right", x, y, 3.7)

            # The 4 blobs are separate anims
            click("right", x, y, 0.2)
            click("right", x, y, 0.2)
            click("right", x, y, 0.2)
            click("right", x, y, 0.2)
            click("right", x, y, 1.0)
            print("READY!")

        if not self.loaded:

            # We're now looking at the game grid
            # Calc the game grid

            rect = win32gui.GetClientRect(hwnd)
            rect = list(rect)
            rect[0] = int(rect[0] * self.scale)
            rect[1] = int(rect[1] * self.scale)
            rect[2] = int(rect[2] * self.scale)
            rect[3] = int(rect[3] * self.scale)
            rect = tuple(rect)
            pos = win32gui.ClientToScreen(hwnd, rect[:2])
            bbox = (pos[0], pos[1], pos[0] + rect[2], pos[1] + rect[3])
            img = ImageGrab.grab(bbox)
            # Move cursor to center to avoid trimming issues
            cx = int((pos[0] + (bbox[2] - bbox[0])//2) // self.scale)
            cy = int((pos[1] + (bbox[3] - bbox[1])//2) // self.scale)
            win32api.SetCursorPos((cx, cy))

            # Trim the edge of the client due to the stupid rounded borders
            # and then get the bounding box (just the game grid)
            cropped = (img.width//10, img.height//10)
            img = img.crop((cropped[0], cropped[1], img.width-cropped[0], img.height-cropped[1]))

            img_arr = numpy.array(img)
            img_arr[img_arr < 10] = 0
            img = Image.fromarray(img_arr, 'RGB')
            bbox2 = img.getbbox()

            # Add to pos to get the grid in screenspace
            grid = (
                pos[0] + bbox2[0] + cropped[0],
                pos[1] + bbox2[1] + cropped[1],
                pos[0] + bbox2[2] + cropped[0],
                pos[1] + bbox2[3] + cropped[1],
            )

            grid_w = grid[2] - grid[0]
            grid_h = grid[3] - grid[1]
            cell_w = (grid_w // 7)
            cell_h = (grid_h // 7)

            for x in range(7):
                for y in range(7):
                    cell_x = grid[0] + (x * cell_w) + (cell_w // 2)
                    cell_y = grid[1] + (y * cell_h) + (cell_h // 2)
                    self.grid[x][y] = (cell_x // self.scale, cell_y//self.scale)

            # Put cursor back to end of screen
            # TODO calc this properly
            self.safe_mouse_pos = (pos[0] // 2 + 8, pos[1] //2 +8)
            win32api.SetCursorPos(self.safe_mouse_pos)
            self.loaded = True

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
        if self.proc:
            self.proc.kill()
        pass
