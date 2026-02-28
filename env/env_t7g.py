import subprocess
import time

import gymnasium
from gymnasium import Env
import numpy
from PIL import ImageGrab, Image, ImageChops, ImageEnhance, ImageOps
import win32gui
import win32api
import win32con

from lib.t7g import calc_reward, action_to_move, is_action_valid, BLUE
from lib.reward_functions import get_reward_function

# Max seconds to wait for the board to stabilise after a move
_WAIT_TIMEOUT = 30.0


def get_game():
    toplist, winlist = [], []

    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
    win32gui.EnumWindows(enum_cb, toplist)

    game_exe = [(hwnd, title) for hwnd, title in winlist if 'the 7th guest' in title.lower()]
    if not game_exe:
        return None
    return game_exe[0][0]


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((5, 5)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -40)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im  # Return original if already uniform (avoids None crash)


def click(left_or_right, x, y, delay):
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    if left_or_right == 'left':
        # dx/dy are 0 — cursor is already positioned via SetCursorPos
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    elif left_or_right == 'right':
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    time.sleep(delay)


class MicroscopeEnv(Env):

    metadata = {"name": "Microscope (Real Game)", "render_modes": [None]}

    def __init__(self, scale=1.0, reward_fn='original', debug=False):
        super().__init__()
        self.proc = None
        self.loaded = False
        self.scale = scale
        self.turns = 0
        self.turn_limit = 100
        self.debug = debug
        self.turn = True  # Always blue in the real game

        # Screen positions of each grid cell
        self.grid = numpy.zeros((7, 7, 2), dtype=numpy.int64)
        self.safe_mouse_pos = (0, 0)
        self._hwnd = None  # Cached window handle — avoids EnumWindows on every capture

        # Reward function
        self.reward_fn_name = reward_fn
        if reward_fn == 'original':
            self.reward_fn = calc_reward
        else:
            self.reward_fn = get_reward_function(reward_fn)

        # Two-stage action tracking (matches env_virt interface)
        self.action_stage = 0       # 0 = select piece, 1 = select move
        self.selected_piece_pos = None  # (x, y) of selected piece

        # Observation: 7x7x4 (green, blue, turn indicator, selected piece)
        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(7, 7, 4), dtype=numpy.float32)
        # Two-stage action space: 49 positions (piece or move)
        self.action_space = gymnasium.spaces.Discrete(49)

    def _capture_board(self):
        """Capture game screenshot and update self.game_grid (7x7x2 bool_)."""
        # Use cached hwnd; re-query only if the window has gone away
        if not self._hwnd or not win32gui.IsWindow(self._hwnd):
            self._hwnd = get_game()

        hwnd = self._hwnd

        # Only bring window to front if it doesn't already have focus
        if win32gui.GetForegroundWindow() != hwnd:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)

        win32api.SetCursorPos(self.safe_mouse_pos)

        rect = win32gui.GetClientRect(hwnd)
        width = int(rect[2] * self.scale)
        height = int(rect[3] * self.scale)
        client_pos = win32gui.ClientToScreen(hwnd, (0, 0))
        screen_bbox = (client_pos[0], client_pos[1], client_pos[0] + width, client_pos[1] + height)
        img = ImageGrab.grab(screen_bbox)

        # Trim edge of client (rounded borders / cursor interference)
        img = img.crop((img.width//10, img.height//10, img.width-img.width//10, img.height-img.height//10))

        # Make near-black pixels fully black
        img_arr = numpy.array(img)
        img_arr[img_arr < 10] = 0
        img = Image.fromarray(img_arr, 'RGB')

        # Crop to game grid bounding box
        bbox = img.getbbox()
        img = img.crop(bbox)
        img = trim(img)

        # Remove red channel noise
        red = Image.new(img.mode, img.size, (255, 0, 0))
        img = ImageChops.subtract(img, red)

        im3 = ImageEnhance.Brightness(img)
        img = im3.enhance(1.5)

        # Crush noise
        img = ImageOps.posterize(img, 3)

        # Reduce to 7x7 grid
        img.thumbnail((7, 7), Image.Resampling.BOX)

        # Crush noise again after resampling
        img = ImageOps.posterize(img, 3)

        # Saturate remaining non-black pixels (green & blue blobs)
        im3 = ImageEnhance.Brightness(img)
        img = im3.enhance(255.0)

        gamegrid = numpy.array(img, dtype=numpy.uint8)
        gamegrid[gamegrid > 20] = 1

        # Slice off red channel — only green and blue channels remain
        gamegrid = gamegrid[:, :, 1:3]

        self.game_grid = gamegrid.astype(numpy.bool_)

    def _get_obs(self):
        """Build 4-channel observation from current self.game_grid."""
        obs = numpy.zeros((7, 7, 4), dtype=numpy.float32)
        obs[:, :, 0:2] = self.game_grid
        obs[:, :, 2] = 1.0  # Always blue's turn in the real game
        if self.action_stage == 1 and self.selected_piece_pos is not None:
            x, y = self.selected_piece_pos
            obs[y, x, 3] = 1.0
        return obs

    def _encode_action(self, x, y, dx, dy):
        """Convert position + delta to 1225-action encoding used by game logic."""
        move_idx = (dy + 2) * 5 + (dx + 2)
        return y * 7 * 25 + x * 25 + move_idx

    def move(self, action):
        """Execute a move in the real game via mouse clicks."""
        from_x, from_y, to_x, to_y, _ = action_to_move(action)

        if self.debug:
            print(f"B: [{from_x}, {from_y}] => [{to_x}, {to_y}]", end="")

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
        """
        Two-stage action processing:
        - Stage 0: action is position index (0-48) selecting the piece to move
        - Stage 1: action is move index (0-24) selecting how to move it
        """
        if self.action_stage == 0:
            return self._step_select_piece(action)
        else:
            return self._step_select_move(action)

    def _step_select_piece(self, position_action):
        """Stage 0: Record which piece to move. No game interaction."""
        self.selected_piece_pos = (position_action % 7, position_action // 7)
        self.action_stage = 1
        obs = self._get_obs()
        return obs, 0.0, False, False, {'stage': 1, 'selected_pos': self.selected_piece_pos}

    def _step_select_move(self, move_action):
        """Stage 1: Execute the move via mouse, wait for opponent, return reward."""
        dx = (move_action % 5) - 2
        dy = (move_action // 5) - 2
        x, y = self.selected_piece_pos
        full_action = self._encode_action(x, y, dx, dy)

        if not self.move(full_action):
            # Invalid move — refresh board and return penalty
            self._capture_board()
            self.action_stage = 0
            self.selected_piece_pos = None
            return self._get_obs(), -1.0, False, False, {}

        # Advance past move confirmation, then wait for opponent
        click("right", *self.grid[0, 0], 0)
        self.turns += 1

        # Wait for the grid to stabilise (opponent finishes their turn)
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

        reward, terminated = self.reward_fn(self.game_grid, True)
        truncated = self.turns >= self.turn_limit

        # Advance past opponent's turn display
        click("right", *self.grid[0, 0], 0)

        if terminated:
            if self.debug:
                print(" — Game over")
            for _ in range(8):
                click("right", *self.grid[0, 0], 0)
        elif self.debug:
            print(f" reward={reward:.3f}")

        # Reset stage before building obs (no selected piece highlight in final obs)
        self.action_stage = 0
        self.selected_piece_pos = None

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.proc:
            time.sleep(1)
        hwnd = get_game()
        load = hwnd is None

        if load:
            self.proc = subprocess.Popen(["C:\\Program Files\\ScummVM\\scummvm.exe", "--save-slot=18", "t7g"])
            time.sleep(27)
            hwnd = get_game()  # re-query after launch

        self._hwnd = hwnd
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

            # The 4 blobs are separate animations
            for _ in range(4):
                click("right", x, y, 0.2)
            click("right", x, y, 1.0)
            print("READY!")

        if not self.loaded:
            # Calculate screen positions for each grid cell
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

            # Move cursor to centre to avoid trimming issues
            cx = int((pos[0] + (bbox[2] - bbox[0])//2) // self.scale)
            cy = int((pos[1] + (bbox[3] - bbox[1])//2) // self.scale)
            win32api.SetCursorPos((cx, cy))

            cropped = (img.width//10, img.height//10)
            img = img.crop((cropped[0], cropped[1], img.width-cropped[0], img.height-cropped[1]))

            img_arr = numpy.array(img)
            img_arr[img_arr < 10] = 0
            img = Image.fromarray(img_arr, 'RGB')
            bbox2 = img.getbbox()

            # Grid bounding box in screen coordinates
            grid = (
                pos[0] + bbox2[0] + cropped[0],
                pos[1] + bbox2[1] + cropped[1],
                pos[0] + bbox2[2] + cropped[0],
                pos[1] + bbox2[3] + cropped[1],
            )

            grid_w = grid[2] - grid[0]
            grid_h = grid[3] - grid[1]
            cell_w = grid_w // 7
            cell_h = grid_h // 7

            for x in range(7):
                for y in range(7):
                    cell_x = grid[0] + (x * cell_w) + (cell_w // 2)
                    cell_y = grid[1] + (y * cell_h) + (cell_h // 2)
                    self.grid[x][y] = (cell_x // self.scale, cell_y // self.scale)

            self.safe_mouse_pos = (int(pos[0] // self.scale + 8), int(pos[1] // self.scale + 8))
            win32api.SetCursorPos(self.safe_mouse_pos)
            self.loaded = True

        self.turns = 0
        self.action_stage = 0
        self.selected_piece_pos = None

        self._capture_board()
        return self._get_obs(), {}

    def action_masks(self):
        """Return valid actions for the current stage (matches env_virt interface)."""
        if self.action_stage == 0:
            return self._action_masks_select_piece()
        else:
            return self._action_masks_select_move()

    def _action_masks_select_piece(self):
        """Stage 0: Positions with blue pieces that have at least one valid move."""
        mask = numpy.zeros(49, dtype=bool)
        for y in range(7):
            for x in range(7):
                if numpy.array_equal(self.game_grid[y, x], BLUE):
                    for move_idx in range(25):
                        dx = (move_idx % 5) - 2
                        dy = (move_idx // 5) - 2
                        if is_action_valid(self.game_grid, self._encode_action(x, y, dx, dy), True):
                            mask[y * 7 + x] = True
                            break
        return mask

    def _action_masks_select_move(self):
        """Stage 1: Valid moves from selected position (first 25 entries, padded to 49)."""
        mask = numpy.zeros(49, dtype=bool)
        if self.selected_piece_pos is None:
            return mask
        x, y = self.selected_piece_pos
        for move_idx in range(25):
            dx = (move_idx % 5) - 2
            dy = (move_idx // 5) - 2
            if is_action_valid(self.game_grid, self._encode_action(x, y, dx, dy), True):
                mask[move_idx] = True
        return mask

    def close(self):
        if self.proc:
            self.proc.kill()
