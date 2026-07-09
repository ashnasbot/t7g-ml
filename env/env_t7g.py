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
from PIL import ImageGrab, Image, ImageDraw
import win32gui
import win32api
import win32con

from lib.t7g import action_to_move, apply_move, is_action_valid, check_terminal, count_cells

_WAIT_TIMEOUT = 30.0      # hard cap (seconds) for full stability wait
_STAUF_START_TIMEOUT = 8.0  # seconds to wait for Stauf to start moving before assuming pass


def get_game():
    winlist = []
    win32gui.EnumWindows(lambda hwnd, _: winlist.append((hwnd, win32gui.GetWindowText(hwnd))), [])
    matches = [(hwnd, t) for hwnd, t in winlist if 'the 7th guest' in t.lower()]
    return matches[0][0] if matches else None



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
        """Sample each calibrated cell centre and classify pieces by colour."""
        if not self._hwnd or not win32gui.IsWindow(self._hwnd):
            self._hwnd = get_game()
        hwnd = self._hwnd
        rect = win32gui.GetClientRect(hwnd)
        width      = int(rect[2] * self.scale)
        height     = int(rect[3] * self.scale)
        client_pos = win32gui.ClientToScreen(hwnd, (0, 0))
        screen_bbox = (client_pos[0], client_pos[1],
                       client_pos[0] + width, client_pos[1] + height)
        img     = ImageGrab.grab(screen_bbox)
        img_arr = numpy.array(img, dtype=numpy.int32)

        # self.grid[gx, gy] = (screen_x, screen_y) / self.scale
        # Convert back to image-relative pixel coordinates.
        game_grid = numpy.zeros((7, 7, 2), dtype=numpy.bool_)
        for gx in range(7):
            for gy in range(7):
                sx = int(self.grid[gx, gy, 0] * self.scale) - client_pos[0]
                sy = int(self.grid[gx, gy, 1] * self.scale) - client_pos[1]
                # Average a 7×7 patch around the cell centre
                r0, r1 = max(0, sy - 3), min(img_arr.shape[0], sy + 4)
                c0, c1 = max(0, sx - 3), min(img_arr.shape[1], sx + 4)
                if r1 <= r0 or c1 <= c0:
                    continue
                patch = img_arr[r0:r1, c0:c1]
                r = int(patch[:, :, 0].mean())
                g = int(patch[:, :, 1].mean())
                b = int(patch[:, :, 2].mean())
                # Subtract red background; classify by whichever non-red channel
                # is dominant (Blue piece -> high B-R; Green/Stauf piece -> high G-R).
                g_adj = g - r
                b_adj = b - r
                if b_adj > 30 and b_adj >= g_adj:
                    game_grid[gy, gx, 1] = True   # Blue piece
                elif g_adj > 30:
                    game_grid[gy, gx, 0] = True   # Green (Stauf) piece

        self.game_grid = game_grid

    def move(self, action: int) -> bool:
        """Execute a 1225-dim action via mouse clicks. Returns True if the move was valid."""
        from_x, from_y, to_x, to_y, _ = action_to_move(action)
        if self._hwnd != win32gui.GetForegroundWindow():
            print("lost focus, exiting")
            exit(1)

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

        # Compute the expected post-Blue board from game logic - avoids any timing
        # dependency on when Blue's animation settles on screen.
        board_after_blue = apply_move(self.game_grid, action, True)

        # Check if Blue's move already ended the game (winning capture).
        is_terminal, terminal_value = check_terminal(board_after_blue, self.turn)
        if is_terminal:
            self.game_grid = board_after_blue
            terminated = True
            truncated = self.turns >= self.turn_limit
            reward = float(terminal_value) if terminal_value is not None else 0.0
            if self.debug:
                blue, green = count_cells(self.game_grid)
                print(f"  B:{blue} G:{green} - Game over (Blue wins)")
            for _ in range(8):
                click('right', *self.grid[0, 0], 0)
            return self.game_grid.copy(), reward, terminated, truncated, {'after_blue': board_after_blue}

        # Park cursor away from the board once, before polling starts.
        win32api.SetCursorPos(self.safe_mouse_pos)

        # Wait for opponent's move to complete.
        # Phase 1: wait for the board to change from board_after_blue (Stauf starts moving).
        #   If no change within _STAUF_START_TIMEOUT seconds, assume Stauf has no moves.
        # Phase 2: once Stauf's move has started, enforce a minimum animation wait
        #   (_STAUF_ANIM_MIN) then require 4 consecutive stable frames.
        #   If Phase 2 exits with the board still equal to board_after_blue it was a
        #   false trigger (game flicker); discard and restart Phase 1.
        _STAUF_ANIM_MIN = 1.2   # seconds to let all captures animate before checking stability
        stauf_moved = False
        stauf_move_time = 0.0
        prev_grid = None
        stable_count = 0
        t_start = time.monotonic()
        deadline = t_start + _WAIT_TIMEOUT
        while True:
            if time.monotonic() > deadline:
                print("[WARNING] Timed out waiting for board to stabilise")
                break
            self._capture_board()
            if not stauf_moved:
                if not numpy.array_equal(self.game_grid, board_after_blue):
                    stauf_moved = True
                    stauf_move_time = time.monotonic()
                elif time.monotonic() - t_start > _STAUF_START_TIMEOUT:
                    break                # Stauf has no moves - accept current board
            if stauf_moved:
                if time.monotonic() - stauf_move_time < _STAUF_ANIM_MIN:
                    # Still inside minimum animation window - don't count stability yet
                    prev_grid = None
                    stable_count = 0
                elif prev_grid is not None and numpy.array_equal(self.game_grid, prev_grid):
                    stable_count += 1
                    if stable_count >= 4:
                        if numpy.array_equal(self.game_grid, board_after_blue):
                            # False trigger - board returned to original; restart Phase 1
                            stauf_moved = False
                            stauf_move_time = 0.0
                            stable_count = 0
                            prev_grid = None
                            t_start = time.monotonic()
                        else:
                            break        # Animation complete, board genuinely changed
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
                print(" - Game over")
            for _ in range(8):
                click('right', *self.grid[0, 0], 0)
        elif self.debug:
            blue, green = count_cells(self.game_grid)
            print(f"  B:{blue} G:{green}")

        return self.game_grid.copy(), reward, terminated, truncated, {'after_blue': board_after_blue}

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
            # Wait for general midi to ready his armies
            time.sleep(28)
            hwnd = get_game()

        self._hwnd = hwnd
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.2)

        if load:
            pos = win32gui.GetWindowRect(hwnd)

            # Click towards the microscope
            x = pos[0] + (pos[2] - pos[0]) // 3
            y = pos[1] + (pos[3] - pos[1]) // 2
            click('left', x, y, 0.1)
            click('right', x, y, 0.3)
            click('right', x, y, 0.8)

            # click the scope itself
            x = int(pos[0] + (pos[2] - pos[0]) // 1.5)
            y = int(pos[1] + (pos[3] - pos[1]) // 2)
            click('left', x, y, 0.3)
            click('right', x, y, 3.7)
            # cells spawning
            for _ in range(4):
                click('right', x, y, 0.2)
            # "A puzzle! speech"
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
            if bbox2 is None or (bbox2[2] - bbox2[0]) < 100 or (bbox2[3] - bbox2[1]) < 100:
                print("ERROR: Could not detect the Microscope game board.\n"
                      "       Make sure the game is loaded to the Microscope puzzle screen\n"
                      "       (save slot 18) and the board is fully visible.")
                raise SystemExit(1)

            grid = (
                pos[0] + bbox2[0] + cropped[0],
                pos[1] + bbox2[1] + cropped[1],
                pos[0] + bbox2[2] + cropped[0],
                pos[1] + bbox2[3] + cropped[1],
            )
            raw_w = grid[2] - grid[0]
            raw_h = grid[3] - grid[1]
            bi_x = round(raw_w * 0.02)
            bi_y = round(raw_h * 0.03)
            inner = (grid[0] + bi_x, grid[1] + bi_y, grid[2] - bi_x, grid[3] - bi_y)
            total_w = inner[2] - inner[0]
            total_h = inner[3] - inner[1]
            for x in range(7):
                for y in range(7):
                    self.grid[x][y] = (
                        round((inner[0] + (x + 0.5) * total_w / 7) / self.scale),
                        round((inner[1] + (y + 0.5) * total_h / 7) / self.scale),
                    )
            self.safe_mouse_pos = (int(pos[0] // self.scale + 8), int(pos[1] // self.scale + 8))
            win32api.SetCursorPos(self.safe_mouse_pos)
            self.loaded = True
            if self.debug:
                print(f"Grid calibration: board_px=({grid[0]},{grid[1]})-({grid[2]},{grid[3]}) "
                      f"inset=({bi_x},{bi_y})  cell_grid={total_w}x{total_h}  cell={total_w/7:.1f}x{total_h/7:.1f}")
                print(f"  cell(0,0)={tuple(self.grid[0,0])}  cell(6,6)={tuple(self.grid[6,6])}"
                      f"  cell(3,3)={tuple(self.grid[3,3])}")
                # Save annotated screenshot so cell centres can be visually verified.
                dbg_img = ImageGrab.grab(bbox).copy()
                draw = ImageDraw.Draw(dbg_img)
                origin_x, origin_y = pos[0], pos[1]
                for gx in range(7):
                    for gy in range(7):
                        cx = int(self.grid[gx, gy, 0] * self.scale) - origin_x
                        cy = int(self.grid[gx, gy, 1] * self.scale) - origin_y
                        r = 6
                        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                                     outline='red', width=2)
                        draw.line([cx - r, cy, cx + r, cy], fill='red', width=1)
                        draw.line([cx, cy - r, cx, cy + r], fill='red', width=1)
                dbg_img.save('calibration_debug.png')
                print("  Saved calibration_debug.png - check cell centres are centred on pieces")

        self.turns = 0
        self.turn = True
        win32api.SetCursorPos(self.safe_mouse_pos)
        self._capture_board()
        return self.game_grid.copy(), {}

    def close(self) -> None:
        if self.proc:
            self.proc.kill()
