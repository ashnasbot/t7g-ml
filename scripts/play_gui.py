"""
Interactive Pyglet GUI to play Microscope against a trained MCTS model or minimax.

Controls:
    Click piece → click destination    (clone = 1 step, jump = 2 steps)
    E   – toggle edit mode (click cells to cycle blue/green/empty; T swaps active player)
    R   – restart
    Esc – quit

Usage:
    python scripts/play_gui.py --checkpoint models/mcts/iter_0040.pt
    python scripts/play_gui.py --checkpoint models/mcts/iter_0040.pt --simulations 50
    python scripts/play_gui.py --opponent minimax --depth 3
    python scripts/play_gui.py --opponent stauf --depth 5
    python scripts/play_gui.py --checkpoint models/mcts/iter_0040.pt --human-color green
"""
import argparse
import os
import pathlib
import sys
import threading

import numpy as np
import pyglet
import pyglet.shapes as shapes

import torch
from lib.dual_network import DualHeadNetwork
from lib.mcgs import MCGS
from lib.t7g import new_board, apply_move, check_terminal
from lib.t7g import find_best_move, count_cells, action_masks, action_to_move
from lib.t7g import BLUE, GREEN


# ── Layout ────────────────────────────────────────────────────────────────────

CELL    = 80        # pixels per board cell
PAD     = 24        # outer padding
LABEL   = 20        # space for row/col labels
INFO_H  = 60        # status bar height at bottom

BOARD_X = PAD + LABEL               # screen-x of board left edge
BOARD_Y = INFO_H + LABEL            # screen-y of board bottom edge
WIN_W   = 2 * PAD + LABEL + 7 * CELL
WIN_H   = INFO_H + LABEL + 7 * CELL + LABEL + PAD


# ── Colors (R, G, B) ──────────────────────────────────────────────────────────

BG          = (22, 22, 22)
C_EMPTY     = (62, 52, 42)
C_BLUE      = (55, 120, 215)
C_GREEN     = (50, 175, 70)
C_SEL       = (230, 185, 30)        # selected piece
C_CLONE     = (100, 155, 255)       # valid clone destination (1 step)
C_JUMP      = (195, 90, 255)        # valid jump destination (2 steps)
C_GRID      = (12, 12, 12, 255)
C_LABEL     = (150, 150, 150, 255)
C_STATUS    = (200, 200, 200, 255)
C_BAR       = (15, 15, 15)
C_BAR_EDIT  = (38, 28, 0)      # amber tint for edit mode status bar


# ── Game states ───────────────────────────────────────────────────────────────

S_SELECT = "select"     # human choosing a piece
S_DEST   = "dest"       # human choosing a destination
S_AI     = "ai"         # AI thinking (background thread)
S_OVER   = "over"       # game finished
S_EDIT   = "edit"       # board editor


# ── Coordinate helpers ────────────────────────────────────────────────────────

def cell_xy(col, row):
    """Bottom-left screen pixel of board cell (col, row)."""
    return BOARD_X + col * CELL, BOARD_Y + (6 - row) * CELL


def screen_to_cell(mx, my):
    """Mouse screen coords → (col, row), or (None, None) if outside board."""
    col = (mx - BOARD_X) // CELL
    row = 6 - (my - BOARD_Y) // CELL
    if 0 <= col < 7 and 0 <= row < 7:
        return int(col), int(row)
    return None, None


def coords_to_action(fc, fr, tc, tr):
    """(from_col, from_row, to_col, to_row) → 1225-action index, or None."""
    dx, dy = tc - fc, tr - fr
    if abs(dx) > 2 or abs(dy) > 2:
        return None
    piece = fr * 7 + fc
    move  = (dy + 2) * 5 + (dx + 2)
    return piece * 25 + move


# ── Main window ───────────────────────────────────────────────────────────────

class MicroscopeApp(pyglet.window.Window):

    def __init__(self, mcts_agent=None, opponent="mcts", minimax_depth=2,
                 human_color="blue", engine="minimax"):
        super().__init__(WIN_W, WIN_H, caption="Microscope", resizable=False)
        self.mcts_agent   = mcts_agent
        self.opponent     = opponent
        self.mm_depth     = minimax_depth
        self.engine       = engine          # 'minimax', 'micro3', or 'stauf'
        self.human_blue   = (human_color == "blue")
        self._reset()

    # ── Game reset ────────────────────────────────────────────────────────────

    def _reset(self):
        self.board       = new_board()
        self.turn        = True          # Blue always goes first
        self.selected    = None          # (col, row) of selected piece
        self.valid_dests = {}            # (col, row) → action
        self.board_hist  = {}
        if self.mcts_agent:
            self.mcts_agent.root = None  # discard old search tree

        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = "Your turn — click a piece"
        else:
            self.state  = S_AI
            self.status = "AI thinking…"
            self._start_ai_move()

    def _is_human_turn(self):
        return self.turn == self.human_blue

    def _human_cell(self):
        return BLUE if self.human_blue else GREEN

    def _fmt_score(self):
        b, g = count_cells(self.board)
        return f"Blue {b} – Green {g}"

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.R:
            self._reset()
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()
        elif symbol == pyglet.window.key.E:
            if self.state == S_EDIT:
                self._exit_edit()
            elif self.state != S_AI:   # don't interrupt AI thread
                self._enter_edit()
        elif symbol == pyglet.window.key.T and self.state == S_EDIT:
            self.turn = not self.turn
            self._edit_status()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def on_mouse_press(self, mx, my, button, modifiers):
        col, row = screen_to_cell(mx, my)
        if col is None:
            return

        if self.state == S_EDIT:
            self._edit_cycle_cell(col, row)
            return

        if self.state not in (S_SELECT, S_DEST):
            return

        if self.state == S_SELECT:
            self._try_select(col, row)

        elif self.state == S_DEST:
            if (col, row) == self.selected:
                # Click same piece again → deselect
                self._deselect()
            elif (col, row) in self.valid_dests:
                self._human_move(self.valid_dests[(col, row)])
            elif np.array_equal(self.board[row, col], self._human_cell()):
                # Click a different own piece → reselect
                self._try_select(col, row)

    def _try_select(self, col, row):
        if not np.array_equal(self.board[row, col], self._human_cell()):
            return
        dests = self._legal_dests(col, row)
        if dests:
            self.selected    = (col, row)
            self.valid_dests = dests
            self.state       = S_DEST
            self.status      = f"({col},{row}) selected  [{self._fmt_score()}]"
        else:
            self.status = f"({col},{row}) has no legal moves  [{self._fmt_score()}]"

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
        player = "Blue" if self.turn else "Green"
        self.status = (f"EDIT — {player} to move  [{self._fmt_score()}]  "
                       f"[click=cycle  T=swap player  E=resume]")

    def _edit_cycle_cell(self, col, row):
        """Cycle cell at (col, row): blue → green → empty → blue."""
        cell = self.board[row, col]
        if np.array_equal(cell, BLUE):
            self.board[row, col] = GREEN
        elif np.array_equal(cell, GREEN):
            self.board[row, col] = [False, False]
        else:
            self.board[row, col] = BLUE
        self._edit_status()

    def _exit_edit(self):
        """Return to play, resetting history and MCTS tree."""
        if self.mcts_agent:
            self.mcts_agent.root = None
        self.board_hist  = {}
        self.selected    = None
        self.valid_dests = {}

        is_terminal, terminal_value = check_terminal(self.board, self.turn)
        if is_terminal:
            blue_val = terminal_value if self.turn else -terminal_value
            if blue_val > 0:
                self._end_game("Blue wins!")
            elif blue_val < 0:
                self._end_game("Green wins!")
            else:
                self._end_game("Draw!")
            return

        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = f"Your turn  [{self._fmt_score()}]"
        else:
            self.state  = S_AI
            self.status = "AI thinking…"
            self._start_ai_move()

    # ── Move legality ─────────────────────────────────────────────────────────

    def _legal_dests(self, fc, fr):
        masks = action_masks(self.board, self.turn)
        dests = {}
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                tc, tr = fc + dx, fr + dy
                if 0 <= tc < 7 and 0 <= tr < 7:
                    action = coords_to_action(fc, fr, tc, tr)
                    if action is not None and masks[action]:
                        dests[(tc, tr)] = action
        return dests

    # ── Move application ──────────────────────────────────────────────────────

    def _human_move(self, action):
        fx, fy, tx, ty, jump = action_to_move(action)
        label = "jump" if jump else "clone"
        self._apply_move(action)
        if self.state == S_OVER:
            return
        self.state  = S_AI
        self.status = f"You: ({fx},{fy})→({tx},{ty}) [{label}]  [{self._fmt_score()}]  |  AI thinking…"
        self._start_ai_move()

    def _finish_ai_move(self, action):
        if action is None:
            # AI reports no moves; check_terminal will confirm
            self._end_game("AI has no legal moves")
            return
        fx, fy, tx, ty, jump = action_to_move(action)
        label    = "jump" if jump else "clone"
        if self.opponent == "mcts":
            ai_label = "MCTS"
        elif self.engine == "stauf":
            ai_label = f"Stauf-{self.mm_depth}"
        elif self.engine == "micro3":
            ai_label = f"Micro3-{self.mm_depth}"
        elif self.engine == "hmcts":
            ai_label = f"HMCTS-{self.mm_depth}"
        else:
            ai_label = f"Micro4-{self.mm_depth}"
        self._apply_move(action)
        if self.state == S_OVER:
            return
        self.state  = S_SELECT
        self.status = f"{ai_label}: ({fx},{fy})→({tx},{ty}) [{label}]  [{self._fmt_score()}]  |  Your turn"

    def _apply_move(self, action):
        """Apply action, flip turn, check repetition + terminal."""
        self.board = apply_move(self.board, action, self.turn)
        if self.mcts_agent:
            self.mcts_agent.advance_tree(action)
        self.selected    = None
        self.valid_dests = {}
        self.turn        = not self.turn

        # 3-fold repetition
        key = self.board.tobytes() + bytes([self.turn])
        self.board_hist[key] = self.board_hist.get(key, 0) + 1
        if self.board_hist[key] >= 3:
            b, g = count_cells(self.board)
            winner = "Blue" if b > g else ("Green" if g > b else "nobody (draw)")
            self._end_game(f"3-fold repetition — {winner} wins by cell count")
            return

        # Terminal state
        is_terminal, terminal_value = check_terminal(self.board, self.turn)
        if is_terminal:
            # terminal_value is from current player's perspective
            blue_val = terminal_value if self.turn else -terminal_value
            if blue_val > 0:
                self._end_game("Blue wins!")
            elif blue_val < 0:
                self._end_game("Green wins!")
            else:
                self._end_game("Draw!")

    def _end_game(self, msg):
        self.state  = S_OVER
        self.status = f"GAME OVER — {msg}  ({self._fmt_score()})  |  [R] to restart"

    # ── AI background thread ──────────────────────────────────────────────────

    def _start_ai_move(self):
        board_snap = self.board.copy()
        turn_snap  = self.turn
        result = [None]
        done   = [False]

        def compute():
            if self.opponent == "mcts" and self.mcts_agent:
                probs  = self.mcts_agent.search(board_snap, turn_snap)
                action = self.mcts_agent.select_action(probs, temperature=0)
            else:
                board_bytes = board_snap.tobytes()
                action = find_best_move(board_bytes, self.mm_depth, turn_snap,
                                        self.engine)
                if action in (-1, 1225):
                    action = None
            result[0] = action
            done[0]   = True

        threading.Thread(target=compute, daemon=True).start()

        def poll(dt):
            if done[0]:
                pyglet.clock.unschedule(poll)
                self._finish_ai_move(result[0])

        pyglet.clock.schedule_interval(poll, 0.05)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def on_draw(self):
        self.clear()
        batch  = pyglet.graphics.Batch()
        _keep  = []   # prevent GC of shapes before batch.draw()

        # Background
        _keep.append(shapes.Rectangle(0, 0, WIN_W, WIN_H, color=BG, batch=batch))

        # Board cells
        for row in range(7):
            for col in range(7):
                x, y   = cell_xy(col, row)
                cell   = self.board[row, col]

                if   np.array_equal(cell, BLUE):  base = C_BLUE
                elif np.array_equal(cell, GREEN): base = C_GREEN
                else:                             base = C_EMPTY

                r = shapes.Rectangle(x + 1, y + 1, CELL - 2, CELL - 2,
                                     color=base, batch=batch)
                _keep.append(r)

                # Selected piece highlight
                if self.selected == (col, row):
                    ov = shapes.Rectangle(x + 1, y + 1, CELL - 2, CELL - 2,
                                          color=C_SEL, batch=batch)
                    ov.opacity = 150
                    _keep.append(ov)

                # Valid destination highlights
                elif self.state == S_DEST and (col, row) in self.valid_dests:
                    sc, sr = self.selected
                    dist   = max(abs(col - sc), abs(row - sr))
                    hint   = C_CLONE if dist == 1 else C_JUMP
                    ov     = shapes.Rectangle(x + 1, y + 1, CELL - 2, CELL - 2,
                                              color=hint, batch=batch)
                    ov.opacity = 130
                    _keep.append(ov)

        # Grid lines
        for i in range(8):
            _keep.append(shapes.Line(
                BOARD_X + i * CELL, BOARD_Y,
                BOARD_X + i * CELL, BOARD_Y + 7 * CELL,
                color=C_GRID, batch=batch))
            _keep.append(shapes.Line(
                BOARD_X,            BOARD_Y + i * CELL,
                BOARD_X + 7 * CELL, BOARD_Y + i * CELL,
                color=C_GRID, batch=batch))

        batch.draw()

        # Row labels  (0–6, left of board)
        for row in range(7):
            _, y = cell_xy(0, row)
            pyglet.text.Label(
                str(row),
                font_name="Arial", font_size=10,
                x=BOARD_X - 10, y=y + CELL // 2,
                anchor_x="center", anchor_y="center",
                color=C_LABEL,
            ).draw()

        # Column labels  (0–6, below board)
        for col in range(7):
            x, _ = cell_xy(col, 6)
            pyglet.text.Label(
                str(col),
                font_name="Arial", font_size=10,
                x=x + CELL // 2, y=BOARD_Y - 10,
                anchor_x="center", anchor_y="center",
                color=C_LABEL,
            ).draw()

        # Status bar
        bar_color = C_BAR_EDIT if self.state == S_EDIT else C_BAR
        shapes.Rectangle(0, 0, WIN_W, INFO_H, color=bar_color).draw()
        pyglet.text.Label(
            self.status,
            font_name="Arial", font_size=11,
            x=10, y=INFO_H // 2,
            anchor_x="left", anchor_y="center",
            color=C_STATUS,
            width=WIN_W - 20, multiline=False,
        ).draw()

        # Turn indicator dot in status bar
        dot_color = C_BLUE if self.turn else C_GREEN
        if self.state != S_OVER:
            shapes.Circle(WIN_W - 18, INFO_H // 2, 7, color=dot_color).draw()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play Microscope against AI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to MCTS checkpoint (.pt)")
    parser.add_argument("--opponent", choices=["mcts", "micro3", "micro4", "stauf", "hmcts"], default="mcts",
                        help="Opponent type (default: mcts)")
    parser.add_argument("--depth", type=int, default=2,
                        help="micro4/Stauf search depth (default: 2)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move (default: 100)")
    parser.add_argument("--human-color", choices=["blue", "green"], default="blue",
                        help="Color you play as (default: blue)")
    args = parser.parse_args()

    mcts_agent = None

    if args.opponent == "mcts":
        checkpoint = args.checkpoint
        if not checkpoint:
            # Fall back to bundled model shipped with the package.
            bundled = pathlib.Path(__file__).parent.parent / "lib" / "best.pt"
            if bundled.exists():
                checkpoint = str(bundled)
            else:
                print("Error: no --checkpoint given and no bundled model found at lib/best.pt")
                return 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {checkpoint}  (device: {device})")
        network = DualHeadNetwork(num_actions=1225).to(device)
        ckpt    = torch.load(checkpoint, weights_only=False)
        network.load_state_dict(ckpt["network"])
        network.eval()
        mcts_agent = MCGS(network, num_simulations=args.simulations)
        print(f"MCTS agent ready  ({args.simulations} sims/move)")
    elif args.opponent == "stauf":
        print(f"Stauf (cell_dll) depth-{args.depth} opponent")
    else:
        print(f"{args.opponent} depth-{args.depth} opponent")

    human_label = args.human_color.capitalize()
    if args.opponent == "mcts":
        ai_label = "MCTS"
    elif args.opponent == "hmcts":
        ai_label = f"hmcts-{args.depth}"
    elif args.opponent == "stauf":
        ai_label = f"Stauf-{args.depth}"
    else:
        ai_label = f"{args.opponent}-{args.depth}"
    print(f"You play as {human_label},  opponent: {ai_label}")
    print("Controls: click to select & move  |  R = restart  |  Esc = quit")

    if args.opponent == "stauf":
        engine = "stauf"
    elif args.opponent == "micro3":
        engine = "micro3"
    elif args.opponent == "hmcts":
        engine = "hmcts"
    else:
        engine = "minimax"

    MicroscopeApp(
        mcts_agent=mcts_agent,
        opponent=args.opponent,
        minimax_depth=args.depth,
        human_color=args.human_color,
        engine=engine,
    )
    pyglet.app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
