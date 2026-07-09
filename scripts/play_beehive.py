"""
Interactive Pyglet GUI to play The Beehive Puzzle.

Controls:
    Click piece -> click destination  (clone = 1 hex, jump = 2 hexes)
    E   – toggle edit mode (click cells to cycle Yellow/Red/Empty; T swaps active player)
    R   – restart
    Esc – quit

Usage:
    python scripts/play_beehive.py
    python scripts/play_beehive.py --opponent minimax --ai-time 2000
    python scripts/play_beehive.py --human-color red --opponent minimax
    python scripts/play_beehive.py --opponent random --ai-delay 0.3

    # Replay a solver solution (both sides auto-play from the saved path)
    python scripts/play_beehive.py --solution solution.json
    python scripts/play_beehive.py --solution solution.json --ai-delay 0.8
"""
import argparse
import json
import math
import pathlib
import sys
import threading

# Ensure the project root is on the path so `lib.*` imports work whether this
# script is run directly (`python scripts/play_beehive.py`) or as a module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pyglet
import pyglet.shapes as shapes

from lib.beehive import (
    new_board, apply_move, check_terminal, action_masks,
    count_cells, CELLS, CELL_TO_IDX, N_DIRS, DEST_TABLE,
    YELLOW, RED,
)


# ── Layout ────────────────────────────────────────────────────────────────────

HEX_R  = 36           # outer radius (centre → vertex), pixels
PAD    = 24           # outer padding around the board
INFO_H = 56           # status bar height at bottom
_SQ3   = math.sqrt(3)

def _hex_center_raw(q, r):
    """Cell (q,r) pixel centre relative to board origin.

    Signs chosen so the board matches the in-game orientation:
      px = -q·√3·R   (horizontal mirror)
      py = +(2r+q)·R  (vertical axis inverted vs raw axial formula)
    Net effect: cell (-4,4) appears top-right; move Δ(q,r)=(+1,-1) is down-left.
    """
    return -q * _SQ3 * HEX_R, (2 * r + q) * HEX_R

# Board bounding box (relative to board origin)
_raw = [_hex_center_raw(q, r) for q, r in CELLS]
_bw  = max(p[0] for p in _raw) - min(p[0] for p in _raw)
_bh  = max(p[1] for p in _raw) - min(p[1] for p in _raw)

_HEX_MARGIN = int(2 * HEX_R / _SQ3) + 2
WIN_W = int(_bw + 2 * _HEX_MARGIN + 2 * PAD)
WIN_H = int(_bh + 2 * _HEX_MARGIN + 2 * PAD + INFO_H)

# Board centre on screen
_BCX = WIN_W // 2
_BCY = INFO_H + PAD + _HEX_MARGIN + int(_bh // 2)

# Flat-top hex geometry (tessellating)
_INSET       = 3
_HEX_DRAW_R  = 2 * HEX_R / _SQ3 - _INSET

def _hex_verts(cx, cy, r):
    """Flat-top hex vertices: angles 0°, 60°, 120°, 180°, 240°, 300°."""
    return tuple(
        (int(cx + r * math.cos(math.radians(60 * i))),
         int(cy + r * math.sin(math.radians(60 * i))))
        for i in range(6)
    )

CELL_PX    = [(_BCX + int(px), _BCY + int(py)) for px, py in _raw]
CELL_VERTS = [_hex_verts(cx, cy, _HEX_DRAW_R) for cx, cy in CELL_PX]


# ── Hit-testing: screen → cell ────────────────────────────────────────────────

def _screen_to_cell(mx, my):
    """Return cell index under the mouse, or None if off-board."""
    rel_x = mx - _BCX
    rel_y = my - _BCY
    q_f = -rel_x / (_SQ3 * HEX_R)
    r_f = (rel_y / HEX_R - q_f) / 2
    s_f = -q_f - r_f
    q, r, s = round(q_f), round(r_f), round(s_f)
    dq, dr, ds = abs(q - q_f), abs(r - r_f), abs(s - s_f)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return CELL_TO_IDX.get((q, r))


# ── Colours ───────────────────────────────────────────────────────────────────

BG          = (18, 18, 18)
C_EMPTY     = (52, 44, 36)
C_RED       = (205, 55, 50)
C_YELLOW    = (215, 180, 30)
C_SEL       = (255, 225, 90)
C_CLONE     = (70, 150, 255)
C_JUMP      = (175, 75, 255)
C_STATUS    = (200, 200, 200, 255)
C_BAR       = (14, 14, 14)
C_BAR_EDIT  = (38, 28, 0)       # amber tint for edit mode


# ── States ────────────────────────────────────────────────────────────────────

S_SELECT = "select"
S_DEST   = "dest"
S_AI     = "ai"
S_OVER   = "over"
S_EDIT   = "edit"


# ── Main window ───────────────────────────────────────────────────────────────

def _move_to_action(from_cell: int, to_cell: int) -> int | None:
    """Convert a (from, to) cell pair to an action index, or None if invalid."""
    for di in range(N_DIRS):
        if DEST_TABLE[from_cell, di] == to_cell:
            return from_cell * N_DIRS + di
    return None


class BeehiveApp(pyglet.window.Window):

    def __init__(self, opponent="minimax", ai_delay=0.4, ai_time_ms=1000,
                 human_yellow=True, solution=None):
        super().__init__(WIN_W, WIN_H, caption="Beehive Puzzle", resizable=False)
        self.opponent     = opponent
        self.ai_delay     = ai_delay
        self.ai_time_ms   = ai_time_ms
        self.human_yellow = human_yellow   # True = human plays Yellow, False = human plays Red

        # Replay mode: solution is a list of {"yellow": [f,t], "stauf": [f,t]} dicts
        self._solution      = solution       # full list of move pairs
        self._solution_step = 0             # which pair we're on
        self._replay        = solution is not None

        self._reset()

    def _is_human_turn(self):
        if self._replay:
            return False   # both sides auto-play in replay mode
        return self.turn == self.human_yellow

    def _human_name(self):
        return "Yellow" if self.human_yellow else "Red"

    def _ai_name(self):
        return "Red" if self.human_yellow else "Yellow"

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset(self):
        self.board       = new_board()
        self.turn        = True            # Yellow always goes first
        self.selected    = None
        self.valid_dests = {}
        self.board_hist  = {}
        self._last_pass  = None
        self._solution_step = 0            # restart replay from the beginning

        if self._replay:
            self.state  = S_AI
            n = len(self._solution) if self._solution else 0
            self.status = f"Replaying solution ({n} Yellow plies) — step 1/{n}"
            self._start_ai_move()
        elif self._is_human_turn():
            self.state  = S_SELECT
            self.status = f"Your turn ({self._human_name()})  [{self._fmt_score()}]"
        else:
            self.state  = S_AI
            self.status = f"{self._ai_name()} thinking…"
            self._start_ai_move()

    def _fmt_score(self):
        y, r = count_cells(self.board)
        return f"Y {y}  R {r}"

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.R:
            self._reset()
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()
        elif symbol == pyglet.window.key.E:
            if self.state == S_EDIT:
                self._exit_edit()
            elif self.state != S_AI:
                self._enter_edit()
        elif symbol == pyglet.window.key.T and self.state == S_EDIT:
            self.turn = not self.turn
            self._edit_status()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def on_mouse_press(self, mx, my, button, modifiers):
        ci = _screen_to_cell(mx, my)
        if ci is None:
            return

        if self.state == S_EDIT:
            self._edit_cycle_cell(ci)
            return

        if self.state not in (S_SELECT, S_DEST):
            return

        human_ch = YELLOW if self.human_yellow else RED
        if self.state == S_SELECT:
            self._try_select(ci)
        else:
            if ci == self.selected:
                self._deselect()
            elif ci in self.valid_dests:
                self._human_move(self.valid_dests[ci][0])
            elif self.board[ci, human_ch]:
                self._try_select(ci)

    def _try_select(self, ci):
        human_ch = YELLOW if self.human_yellow else RED
        if not self.board[ci, human_ch]:
            return
        dests = self._legal_dests_from(ci)
        if dests:
            self.selected    = ci
            self.valid_dests = dests
            self.state       = S_DEST
            q, r = CELLS[ci]
            self.status = f"({q},{r}) selected – click highlighted cell  [{self._fmt_score()}]"
        else:
            q, r = CELLS[ci]
            self.status = f"({q},{r}) has no legal moves  [{self._fmt_score()}]"

    def _deselect(self):
        self.selected    = None
        self.valid_dests = {}
        self.state       = S_SELECT
        self.status      = f"Your turn  [{self._fmt_score()}]"

    # ── Edit mode ─────────────────────────────────────────────────────────────

    def _enter_edit(self):
        self.selected    = None
        self.valid_dests = {}
        self.state       = S_EDIT
        self._edit_status()

    def _edit_status(self):
        turn_name = "Yellow" if self.turn else "Red"
        self.status = (f"EDIT – {turn_name} to move  [{self._fmt_score()}]  "
                       f"[click=cycle  T=swap turn  E=resume]")

    def _edit_cycle_cell(self, ci):
        """Cycle cell ci: Yellow → Red → Empty → Yellow."""
        if self.board[ci, YELLOW]:
            self.board[ci, YELLOW] = False
            self.board[ci, RED]    = True
        elif self.board[ci, RED]:
            self.board[ci, RED]    = False
        else:
            self.board[ci, YELLOW] = True
        self._edit_status()

    def _exit_edit(self):
        """Return to play, resetting history."""
        self.board_hist  = {}
        self.selected    = None
        self.valid_dests = {}
        self._last_pass  = None

        is_terminal, terminal_value = check_terminal(self.board, self.turn)
        if is_terminal:
            assert terminal_value is not None
            yellow_val = terminal_value if self.turn else -terminal_value
            if yellow_val > 0:
                self._end_game("Yellow wins!")
            elif yellow_val < 0:
                self._end_game("Red wins!")
            else:
                self._end_game("Draw")
            return

        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = f"Your turn  [{self._fmt_score()}]"
        else:
            self.state  = S_AI
            self.status = f"{self._ai_name()} thinking…"
            self._start_ai_move()

    # ── Move legality ─────────────────────────────────────────────────────────

    def _legal_dests_from(self, ci):
        """Dict of dest_cell_idx → (action, is_jump) for all legal moves from cell ci."""
        masks  = action_masks(self.board, self.turn)
        result = {}
        for di in range(N_DIRS):
            action = ci * N_DIRS + di
            if masks[action]:
                dest_ci = int(DEST_TABLE[ci, di])
                result[dest_ci] = (action, di >= 6)
        return result

    # ── Move application ──────────────────────────────────────────────────────

    def _human_move(self, action):
        ci   = action // N_DIRS
        di   = action % N_DIRS
        dest = int(DEST_TABLE[ci, di])
        q1, r1 = CELLS[ci]
        q2, r2 = CELLS[dest]
        label  = "jump" if di >= 6 else "clone"

        self._apply_move(action)
        if self.state == S_OVER:
            return

        info = f"You: ({q1},{r1})→({q2},{r2}) [{label}]"
        if self._last_pass:
            self.state  = S_SELECT
            self.status = f"{info}  |  {self._ai_name()} passes – your turn  [{self._fmt_score()}]"
        else:
            self.state  = S_AI
            self.status = f"{info}  |  {self._ai_name()} thinking…"
            self._start_ai_move()

    def _finish_ai_move(self, action):
        if action is None:
            self.state  = S_SELECT
            self.status = f"{self._ai_name()} passes  [{self._fmt_score()}]  |  Your turn"
            return

        ci   = action // N_DIRS
        di   = action % N_DIRS
        dest = int(DEST_TABLE[ci, di])
        q1, r1 = CELLS[ci]
        q2, r2 = CELLS[dest]
        label  = "jump" if di >= 6 else "clone"

        self._apply_move(action)
        if self.state == S_OVER:
            return

        info = f"{self._ai_name()}: ({q1},{r1})→({q2},{r2}) [{label}]"
        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = f"{info}  |  Your turn  [{self._fmt_score()}]"
        else:
            # Replay mode or AI-vs-AI: chain straight into the next move
            self.state  = S_AI
            self._start_ai_move()

    def _apply_move(self, action):
        """Apply action, advance turn, check 3-fold repetition, terminal, and forced pass."""
        yellow = self.turn
        self.board = apply_move(self.board, action, yellow)
        self.selected    = None
        self.valid_dests = {}
        self._last_pass  = None
        self.turn        = not yellow

        # 3-fold repetition
        key = self.board.tobytes() + bytes([self.turn])
        self.board_hist[key] = self.board_hist.get(key, 0) + 1
        if self.board_hist[key] >= 3:
            yc, rc = count_cells(self.board)
            winner = "Yellow" if yc > rc else ("Red" if rc > yc else "nobody (draw)")
            self._end_game(f"3-fold repetition – {winner} wins on count")
            return

        is_terminal, terminal_value = check_terminal(self.board, self.turn)
        if is_terminal:
            assert terminal_value is not None
            yellow_val = terminal_value if self.turn else -terminal_value
            if yellow_val > 0:
                self._end_game(f"Yellow wins!  ({self._fmt_score()})")
            elif yellow_val < 0:
                self._end_game(f"Red wins!  ({self._fmt_score()})")
            else:
                self._end_game(f"Draw  ({self._fmt_score()})")
            return

    def _end_game(self, msg):
        self.state  = S_OVER
        self.status = f"GAME OVER – {msg}  |  [R] to restart"

    # ── AI background thread ──────────────────────────────────────────────────

    def _start_ai_move(self):
        # ── Replay mode ───────────────────────────────────────────────────────
        if self._replay and self._solution is not None:
            step = self._solution_step
            if step >= len(self._solution):
                # Solution exhausted; game should already be terminal
                return

            pair  = self._solution[step]
            total = len(self._solution)

            if self.turn:   # Yellow's replay move
                y_from, y_to = pair["yellow"]
                action = _move_to_action(y_from, y_to)
                self.status = (f"Yellow plays {CELLS[y_from]}→{CELLS[y_to]}  "
                               f"[ply {step + 1}/{total}]  [{self._fmt_score()}]")
            else:            # Stauf's stored response
                s_from, s_to = pair["stauf"]
                if s_from == -1:
                    action = None   # Stauf had no individual move (board fill)
                else:
                    action = _move_to_action(s_from, s_to)
                    self.status = (f"Stauf plays {CELLS[s_from]}→{CELLS[s_to]}  "
                                   f"[ply {step + 1}/{total}]  [{self._fmt_score()}]")
                self._solution_step += 1  # advance after Stauf's half-move

            def _replay_apply(_dt):
                self._finish_ai_move(action)

            pyglet.clock.schedule_once(_replay_apply, self.ai_delay)
            return

        # ── Normal AI ─────────────────────────────────────────────────────────
        board_snap = self.board.copy()
        turn_snap  = self.turn     # capture now; AI plays as this colour
        result = [None]
        done   = [False]

        def compute():
            if self.opponent == "minimax":
                from lib.beehive_minimax import beehive_best_move
                action = beehive_best_move(board_snap, as_yellow=turn_snap,
                                           time_ms=self.ai_time_ms)
                result[0] = action if action >= 0 else None
            else:
                legal = np.where(action_masks(board_snap, turn_snap))[0]
                if len(legal) > 0:
                    result[0] = int(np.random.choice(legal))
            done[0] = True

        def start(_dt):
            threading.Thread(target=compute, daemon=True).start()
            pyglet.clock.schedule_interval(poll, 0.05)

        def poll(dt):
            if done[0]:
                pyglet.clock.unschedule(poll)
                self._finish_ai_move(result[0])

        pyglet.clock.schedule_once(start, self.ai_delay)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self):
        self.clear()
        batch = pyglet.graphics.Batch()
        _keep = []

        _keep.append(shapes.Rectangle(0, 0, WIN_W, WIN_H, color=BG, batch=batch))

        for ci, verts in enumerate(CELL_VERTS):
            if self.board[ci, RED]:
                base = C_RED
            elif self.board[ci, YELLOW]:
                base = C_YELLOW
            else:
                base = C_EMPTY

            _keep.append(shapes.Polygon(*verts, color=base, batch=batch))

            if ci == self.selected:
                ov = shapes.Polygon(*verts, color=C_SEL, batch=batch)
                ov.opacity = 180
                _keep.append(ov)
            elif self.state == S_DEST and ci in self.valid_dests:
                _, is_jump = self.valid_dests[ci]
                ov = shapes.Polygon(*verts, color=C_JUMP if is_jump else C_CLONE, batch=batch)
                ov.opacity = 150
                _keep.append(ov)

        batch.draw()

        # Status bar
        bar_color = C_BAR_EDIT if self.state == S_EDIT else C_BAR
        shapes.Rectangle(0, 0, WIN_W, INFO_H, color=bar_color).draw()
        pyglet.text.Label(
            self.status,
            font_name="Arial", font_size=11,
            x=12, y=INFO_H // 2,
            anchor_x="left", anchor_y="center",
            color=C_STATUS,
            width=WIN_W - 50, multiline=False,
        ).draw()

        # Turn indicator dot
        if self.state != S_OVER:
            dot = C_YELLOW if self.turn else C_RED
            shapes.Circle(WIN_W - 22, INFO_H // 2, 9, color=dot).draw()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play The Beehive Puzzle")
    parser.add_argument("--opponent", choices=["random", "minimax"], default="minimax",
                        help="Opponent type (default: minimax)")
    parser.add_argument("--human-color", choices=["yellow", "red"], default="yellow",
                        help="Color you play as (default: yellow)")
    parser.add_argument("--ai-delay", type=float, default=0.4,
                        help="Visual pause before each move in seconds (default: 0.4)")
    parser.add_argument("--ai-time", type=int, default=1000,
                        help="Minimax search budget in ms (default: 1000)")
    parser.add_argument("--solution", metavar="FILE",
                        help="Replay a solver solution JSON (both sides auto-play)")
    args = parser.parse_args()

    solution = None
    if args.solution:
        with open(args.solution) as f:
            data = json.load(f)
        solution = data["moves"]
        yp = data.get("yellow_plies", len(solution))
        ys = data.get("yellow_score", "?")
        rs = data.get("red_score", "?")
        print(f"Replaying solution: {yp} Yellow plies  Yellow {ys} – Red {rs}")
        print(f"Controls: R = restart  |  Esc = quit")
    else:
        human_yellow = (args.human_color == "yellow")
        ai_color     = "Red" if human_yellow else "Yellow"
        print(f"Beehive Puzzle  –  You: {args.human_color.capitalize()}  |  {ai_color}: {args.opponent}")
        if args.opponent == "minimax":
            print(f"Minimax search time: {args.ai_time} ms/move")
        print(f"Window: {WIN_W} × {WIN_H}")
        print("Controls: click piece → destination  |  E = edit mode  |  R = restart  |  Esc = quit")

    human_yellow = (args.human_color == "yellow") if not args.solution else True
    BeehiveApp(opponent=args.opponent, ai_delay=args.ai_delay, ai_time_ms=args.ai_time,
               human_yellow=human_yellow, solution=solution)
    pyglet.app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
