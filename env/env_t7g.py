"""
Real-game interface for Microscope (The 7th Guest) via ScummVM.

The agent always plays as Blue. After each move, step() waits for the Green
AI to finish before returning the updated board.

Interface:
    env = MicroscopeRealEnv()
    board, _ = env.reset()       # board is (7,7,2) bool_
    while True:
        action = ...             # 1225-dim action chosen by MCTS
        board, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""
import subprocess
import time

import numpy
from PIL import ImageGrab, Image, ImageChops, ImageEnhance, ImageOps
import win32gui
import win32api
import win32con

from lib.t7g import action_to_move, is_action_valid, check_terminal, count_cells

_WAIT_TIMEOUT = 30.0


def get_game():
    winlist = []
    win32gui.EnumWindows(lambda hwnd, _: winlist.append((hwnd, win32gui.GetWindowText(hwnd))), [])
    matches = [(hwnd, t) for hwnd, t in winlist if 'the 7th guest' in t.lower()]
    return matches[0][0] if matches else None


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((5, 5)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -40)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


def click(left_or_right, x, y, delay):
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    if left_or_right == 'left':
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    time.sleep(delay)


class MicroscopeRealEnv:
    """
    Interface to the real Microscope game running in ScummVM.

    The agent plays as Blue. step() clicks the agent's move, waits for
    the Green AI to respond, then returns the new board state.
    """

    def __init__(self, scale: float = 1.0, turn_limit: int = 100, debug: bool = False) -> None:
        self.scale = scale
        self.turn_limit = turn_limit
        self.debug = debug
        self.turn = True   # Agent is always Blue
        self.turns = 0
        self.grid = numpy.zeros((7, 7, 2), dtype=numpy.int64)
        self.safe_mouse_pos = (0, 0)
        self._hwnd = None
        self.game_grid = numpy.zeros((7, 7, 2), dtype=numpy.bool_)
        self.proc = None
        self.loaded = False

    def _capture_board(self) -> None:
        """Screenshot the game window and update self.game_grid (7x7x2 bool_)."""
        if not self._hwnd or not win32gui.IsWindow(self._hwnd):
            self._hwnd = get_game()
        hwnd = self._hwnd
        if win32gui.GetForegroundWindow() != hwnd:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)
        win32api.SetCursorPos(self.safe_mouse_pos)

        rect = win32gui.GetClientRect(hwnd)
        width  = int(rect[2] * self.scale)
        height = int(rect[3] * self.scale)
        client_pos = win32gui.ClientToScreen(hwnd, (0, 0))
        screen_bbox = (client_pos[0], client_pos[1], client_pos[0] + width, client_pos[1] + height)
        img = ImageGrab.grab(screen_bbox)

        # Trim borders and cursor interference
        img = img.crop((img.width//10, img.height//10,
                        img.width - img.width//10, img.height - img.height//10))
        img_arr = numpy.array(img)
        img_arr[img_arr < 10] = 0
        img = Image.fromarray(img_arr, 'RGB')
        img = img.crop(img.getbbox())
        img = trim(img)

        # Remove red channel noise, boost brightness, posterize, reduce to 7x7
        img = ImageChops.subtract(img, Image.new(img.mode, img.size, (255, 0, 0)))
        img = ImageEnhance.Brightness(img).enhance(1.5)
        img = ImageOps.posterize(img, 3)
        img.thumbnail((7, 7), Image.Resampling.BOX)
        img = ImageOps.posterize(img, 3)
        img = ImageEnhance.Brightness(img).enhance(255.0)

        gamegrid = numpy.array(img, dtype=numpy.uint8)
        gamegrid[gamegrid > 20] = 1
        self.game_grid = gamegrid[:, :, 1:3].astype(numpy.bool_)  # drop red channel

    def move(self, action: int) -> bool:
        """Execute a 1225-dim action via mouse clicks. Returns True if the move was valid."""
        from_x, from_y, to_x, to_y, _ = action_to_move(action)
        if self.debug:
            print(f"B: [{from_x},{from_y}] => [{to_x},{to_y}]", end="")
        if not is_action_valid(self.game_grid, action, True):
            return False
        click('left',  *self.grid[from_x][from_y], 0)
        click('right', *self.grid[from_x][from_y], 0.1)
        click('left',  *self.grid[to_x][to_y], 0)
        click('right', *self.grid[to_x][to_y], 0.1)
        return True

    def step(self, action: int) -> tuple:
        """
        Execute a 1225-dim action, wait for Green AI to respond, return new board.

        Returns (board, reward, terminated, truncated, info).
        board: (7,7,2) bool_
        reward: +1.0 Blue wins / -1.0 Green wins / 0.0 otherwise
        """
        if not self.move(action):
            self._capture_board()
            return self.game_grid.copy(), -1.0, False, False, {'invalid': True}

        click('right', *self.grid[0, 0], 0)
        self.turns += 1

        # Wait for opponent's move to complete (4 stable frames)
        prev_grid = None
        stable_count = 0
        deadline = time.monotonic() + _WAIT_TIMEOUT
        while stable_count < 4:
            if time.monotonic() > deadline:
                print("[WARNING] Timed out waiting for board to stabilise")
                break
            self._capture_board()
            if prev_grid is not None and numpy.array_equal(self.game_grid, prev_grid):
                stable_count += 1
            else:
                stable_count = 0
            prev_grid = self.game_grid.copy()
            time.sleep(0.2)

        click('right', *self.grid[0, 0], 0)

        is_terminal, terminal_value = check_terminal(self.game_grid, self.turn)
        terminated = bool(is_terminal)
        truncated = self.turns >= self.turn_limit
        reward = float(terminal_value) if (terminated and terminal_value is not None) else 0.0

        if terminated:
            if self.debug:
                print(" — Game over")
            for _ in range(8):
                click('right', *self.grid[0, 0], 0)
        elif self.debug:
            blue, green = count_cells(self.game_grid)
            print(f"  B:{blue} G:{green}")

        return self.game_grid.copy(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None) -> tuple:
        """Launch ScummVM if needed, calibrate grid coordinates, capture initial board."""
        if self.proc:
            time.sleep(1)
        hwnd = get_game()
        load = hwnd is None

        if load:
            self.proc = subprocess.Popen(
                ["C:\\Program Files\\ScummVM\\scummvm.exe", "--save-slot=18", "t7g"]
            )
            time.sleep(27)
            hwnd = get_game()

        self._hwnd = hwnd
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.2)

        if load:
            pos = win32gui.GetWindowRect(hwnd)
            x = pos[0] + (pos[2] - pos[0]) // 3
            y = pos[1] + (pos[3] - pos[1]) // 2
            click('left', x, y, 0.1)
            click('right', x, y, 0.3)
            click('right', x, y, 0.8)
            x = int(pos[0] + (pos[2] - pos[0]) // 1.5)
            y = int(pos[1] + (pos[3] - pos[1]) // 2)
            click('left', x, y, 0.3)
            click('right', x, y, 3.7)
            for _ in range(4):
                click('right', x, y, 0.2)
            click('right', x, y, 1.0)
            print("READY!")

        if not self.loaded:
            rect = win32gui.GetClientRect(hwnd)
            rect = tuple(int(v * self.scale) for v in rect)
            pos = win32gui.ClientToScreen(hwnd, rect[:2])
            bbox = (pos[0], pos[1], pos[0] + rect[2], pos[1] + rect[3])
            img = ImageGrab.grab(bbox)

            cx = int((pos[0] + (bbox[2] - bbox[0]) // 2) // self.scale)
            cy = int((pos[1] + (bbox[3] - bbox[1]) // 2) // self.scale)
            win32api.SetCursorPos((cx, cy))

            cropped = (img.width // 10, img.height // 10)
            img = img.crop((cropped[0], cropped[1],
                            img.width - cropped[0], img.height - cropped[1]))
            img_arr = numpy.array(img)
            img_arr[img_arr < 10] = 0
            img = Image.fromarray(img_arr, 'RGB')
            bbox2 = img.getbbox()

            grid = (
                pos[0] + bbox2[0] + cropped[0],
                pos[1] + bbox2[1] + cropped[1],
                pos[0] + bbox2[2] + cropped[0],
                pos[1] + bbox2[3] + cropped[1],
            )
            cell_w = (grid[2] - grid[0]) // 7
            cell_h = (grid[3] - grid[1]) // 7
            for x in range(7):
                for y in range(7):
                    self.grid[x][y] = (
                        (grid[0] + x * cell_w + cell_w // 2) // self.scale,
                        (grid[1] + y * cell_h + cell_h // 2) // self.scale,
                    )
            self.safe_mouse_pos = (int(pos[0] // self.scale + 8), int(pos[1] // self.scale + 8))
            win32api.SetCursorPos(self.safe_mouse_pos)
            self.loaded = True

        self.turns = 0
        self.turn = True
        self._capture_board()
        return self.game_grid.copy(), {}

    def close(self) -> None:
        if self.proc:
            self.proc.kill()
