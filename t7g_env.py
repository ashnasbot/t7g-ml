import subprocess
import time

import gymnasium
from gymnasium import Env
import numpy
# Imports for interacting with the Game
from PIL import ImageGrab, Image, ImageChops, ImageEnhance, ImageOps
import win32gui
import win32api
import win32con

from t7g_utils import calc_reward, action_to_move, is_action_valid


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


class MicroscopeEnv(Env):

    def __init__(self, scale=1.0):
        super().__init__()
        self.proc = None
        self.loaded = False
        self.scale = scale
        self.turns = 0
        self.turn_limit = 50
        # mouse posttions of each grid cell
        self.grid = numpy.zeros((7, 7, 2), dtype=int)
        self.safe_mouse_pos = (0, 0)

        self.observation_space = gymnasium.spaces.Dict({
            "board": gymnasium.spaces.Box(0, 1, shape=(7, 7, 2), dtype=bool),
            "turn": gymnasium.spaces.Discrete(2),
            "turns": gymnasium.spaces.Discrete(self.turn_limit)
        })
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
        # TODO: this factor changes with screen size and scaling, do better
        img = img.crop((img.width//10, img.height//10, img.width-img.width//10, img.height-img.height//10))

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
        gamegrid = numpy.array(img, dtype=numpy.uint8)
        gamegrid[gamegrid > 20] = 1

        # Now slice off the red channel as it is not needed
        gamegrid = gamegrid[:, :, 1:3]

        self.game_grid = gamegrid

        return {
            "board": gamegrid,
            "turn": 1,
            "turns": self.turns
        }

    def move(self, action):
        from_x, from_y, to_x, to_y, _ = action_to_move(action)

        print(f"[{from_x}, {from_y}] => [{to_x}, {to_y}]: ", end="")
        if is_action_valid(self.game_grid, action, True):

            click_x, click_y = self.grid[from_x][from_y]
            click("left", click_x, click_y, 0)
            click("right", click_x, click_y, 0.1)

            click_x, click_y = self.grid[to_x][to_y]
            click("left", click_x, click_y, 0)
            click("right", click_x, click_y, 0.1)
            return True
        return False

    def step(self, action):
        reward = 0
        terminated = False

        if not self.move(action):
            # Made an inpossible move
            observation = self._get_obs()

            return observation, 0, False, False, {}

        # end of turn, wait for green
        click("right", *self.grid[0, 0], 0)
        self.turns += 1

        # Detect when it's our go again
        # We wait for the grid to stop changing
        times = 0
        while True:
            observation = self._get_obs()
            time.sleep(0.2)
            observation2 = self._get_obs()
            if numpy.array_equal(observation["board"], observation2["board"]):
                times += 1
            else:
                times = 0

            if times == 4:
                break

        click("right", *self.grid[0, 0], 0)
        self.turns += 1

        # Round over - how did we do?
        reward, terminated = calc_reward(observation["board"], True)
        print(f" {reward:>2}", end="")

        if terminated:
            print("- Lost")
            for _ in range(8):
                click("right", *self.grid[0, 0], 0)

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
            self.safe_mouse_pos = (int(pos[0] // self.scale + 8), int(pos[1] // self.scale + 8))
            win32api.SetCursorPos(self.safe_mouse_pos)
            self.loaded = True

        self.turns = 0
        observation = self._get_obs()

        return observation, None

    def close(self):
        if self.proc:
            self.proc.kill()
        pass
