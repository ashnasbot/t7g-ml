"""
Interactive Pyglet GUI to play Microscope against a trained MCTS model or minimax.

Controls:
    Click piece -> click destination    (clone = 1 step, jump = 2 steps)
    E   – toggle edit mode (click cells to cycle blue/green/empty; T swaps active player)
    R   – restart
    Esc – quit

Usage:
    python scripts/play_gui.py --checkpoint models/mcts/iter_0040.pt
    python scripts/play_gui.py --checkpoint models/mcts/iter_0040.pt --simulations 50
    python scripts/play_gui.py --opponent minimax --depth 3
    python scripts/play_gui.py --opponent stauf --depth 5
    python scripts/play_gui.py --checkpoint models/mcts/iter_0040.pt --human-color green

    # AI vs AI
    python scripts/play_gui.py --player mcts --player-checkpoint models/mcts/iter_0040.pt --opponent stauf --depth 5
    python scripts/play_gui.py --player minimax --player-depth 3 --opponent micro3 --depth 3 --ai-delay 1.0
"""
import argparse
import pathlib
import random
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pyglet
import pyglet.shapes as shapes

import torch
from lib import eval_db as edb
from lib import paths
from lib.mcgs import MCGS
from lib.t7g import new_board, apply_move, check_terminal
from lib.t7g import find_best_move, count_cells, action_masks, action_to_move
from lib.t7g import BLUE, GREEN, tick_clock, CLOCK_LIMIT


#  Layout

CELL    = 80        # pixels per board cell
PAD     = 24        # outer padding
LABEL   = 20        # space for row/col labels
INFO_H  = 60        # status bar height at bottom

BOARD_X = PAD + LABEL               # screen-x of board left edge
BOARD_Y = INFO_H + LABEL            # screen-y of board bottom edge
WIN_W   = 2 * PAD + LABEL + 7 * CELL
WIN_H   = INFO_H + LABEL + 7 * CELL + LABEL + PAD


#  Colors (R, G, B)

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


#  Game states

S_SELECT = "select"     # human choosing a piece
S_DEST   = "dest"       # human choosing a destination
S_AI     = "ai"         # AI thinking (background thread)
S_OVER   = "over"       # game finished
S_EDIT   = "edit"       # board editor


#  Coordinate helpers

def cell_xy(col, row):
    """Bottom-left screen pixel of board cell (col, row)."""
    return BOARD_X + col * CELL, BOARD_Y + (6 - row) * CELL


def screen_to_cell(mx, my):
    """Mouse screen coords -> (col, row), or (None, None) if outside board."""
    col = (mx - BOARD_X) // CELL
    row = 6 - (my - BOARD_Y) // CELL
    if 0 <= col < 7 and 0 <= row < 7:
        return int(col), int(row)
    return None, None


def coords_to_action(fc, fr, tc, tr):
    """(from_col, from_row, to_col, to_row) -> 1225-action index, or None."""
    dx, dy = tc - fc, tr - fr
    if abs(dx) > 2 or abs(dy) > 2:
        return None
    piece = fr * 7 + fc
    move  = (dy + 2) * 5 + (dx + 2)
    return piece * 25 + move


#  Rating mode (eval-DB integration)
#
#  A human is just another player node in the offline eval DB (lib/eval_db.py):
#  every finished game is appended as a match row against whatever rated
#  opponent they faced, tagged with the canonical net-rating config_hash so it
#  fits alongside the whole ladder.  Opponents are chosen adaptively (nearest
#  rated Elo to the running estimate) so a tight rating falls out of few games.

_NET_CACHE: dict = {}          # path -> MCGS agent (avoid reloading on revisit)


def _load_network(checkpoint, device):
    """Load a checkpoint into the right architecture, auto-detected from the
    state dict (t7g-net2 or the legacy DualHeadNetwork) -- so the bundled
    ``best.pt`` and any historical net both load.  CPU inference: no compile."""
    from lib.device_utils import load_compiled_network
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    state = blob["network"] if isinstance(blob, dict) and "network" in blob else blob
    net, _ = load_compiled_network(state, device, num_actions=1225, compile_net=False)
    return net


def _rate_config_hash():
    """The canonical net-rating config the whole ladder is fitted under."""
    return edb.config_hash({})


def _load_rate_agent(path, device):
    """MCGS opponent built at the canonical eval-DB config (so games are
    config-comparable with `eval_db add`).  Cached per checkpoint path."""
    if path in _NET_CACHE:
        return _NET_CACHE[path]
    cfg = edb.DEFAULT_CONFIG
    network = _load_network(path, device)
    agent = MCGS(network,
                 num_simulations=cfg["sims"], c_puct=cfg["c_puct"],
                 gumbel_k=cfg["gumbel_k"], sigma_scale=cfg["sigma_scale"],
                 completion_n0=cfg["completion_n0"])
    _NET_CACHE[path] = agent
    return agent


def _fit_anchors(names, reg):
    """Anchor the fit on every pinned (fixed_elo) player, else just stauf.

    In the full DB only stauf is pinned (unchanged behaviour); an exported
    reference bundle pins every opponent at its full-DB Elo so the human is
    rated on the true scale from just a handful of games/models."""
    a = {n: reg[n]["fixed_elo"] for n in names
         if reg.get(n, {}).get("fixed_elo") is not None}
    return a or {edb.STAUF_PLAYER_ID: edb.STAUF_ANCHOR_ELO}


def build_rating_pool(chash):
    """Rated opponents available to play interactively, sorted by Elo.

    Each entry is ``(player_id, kind, payload, elo)`` where kind is
    ``'net'`` (payload=checkpoint path), ``'stauf'`` (payload=depth), or
    ``'mm'`` (payload=minimax depth).  Only players that are (a) rated at the
    canonical config and (b) actually runnable (checkpoint on disk / a
    deterministic engine) qualify -- so the human is always bracketed by
    opponents whose Elo is already known.
    """
    names, counts = edb.load_counts(chash)
    if not names:
        return []
    reg = edb.load_players()
    elo, _ = edb.fit_whr(names, counts, anchors=_fit_anchors(names, reg))
    pool = []
    for k, pid in enumerate(names):
        meta = reg.get(pid, {})
        kind = meta.get("kind")
        e = float(elo[k])
        if pid == edb.STAUF_PLAYER_ID:
            pool.append((pid, "stauf", meta.get("depth", edb.STAUF_DEPTH), e))
        elif kind == "mm":
            pool.append((pid, "mm", meta["depth"], e))
        elif kind == "net":
            path = paths.find_checkpoint(pid)     # search the data bundle
            if path is not None:
                pool.append((pid, "net", str(path), e))
    pool.sort(key=lambda c: c[3])
    return pool


#  Main window

class MicroscopeApp(pyglet.window.Window):

    def __init__(self, mcts_agent=None, opponent="mcts", minimax_depth=2,
                 human_color="blue", engine="minimax",
                 player_mcts=None, player_engine="human", player_depth=2,
                 ai_delay=0.5, rate_id=None, rate_pool=None,
                 rate_first_stauf=True, device=None):
        super().__init__(WIN_W, WIN_H, caption="Microscope", resizable=False)
        # Opponent (Green) config
        self.mcts_agent   = mcts_agent
        self.opponent     = opponent
        self.mm_depth     = minimax_depth
        self.engine       = engine          # 'minimax', 'micro3', 'stauf', etc.
        self.human_blue   = (human_color == "blue")
        # Player (Blue) config — "human" means a person is playing Blue
        self.player_mcts   = player_mcts
        self.player_engine = player_engine  # "human" | "mcts" | "minimax" | ...
        self.player_depth  = player_depth
        self.ai_delay      = ai_delay
        self._last_pass    = None           # set by _apply_move on forced pass
        # Rating mode
        self.rate_id       = rate_id        # None => rating disabled
        self.rate_pool     = rate_pool or []
        self.device        = device or torch.device("cpu")
        self.rate_chash    = _rate_config_hash()
        self.rate_name     = (rate_id or "").split("/")[-1]
        self.rate_n = self.rate_w = self.rate_d = self.rate_l = 0
        self.cur_opp_id    = None
        self.cur_opp_elo   = 0.0
        if self.rate_id:
            elos = [c[3] for c in self.rate_pool]
            self.rate_est = float(np.median(elos)) if elos else 1000.0
            self.rate_ci  = float("inf")
            edb.register_player(self.rate_id, {"kind": "human", "name": self.rate_name})
            first = None
            if rate_first_stauf:
                first = next((c for c in self.rate_pool if c[1] == "stauf"), None)
            self._rate_apply_opponent(first or self._rate_pick_next())
        self._reset()

    @property
    def ai_vs_ai(self):
        return self.player_engine != "human"

    #  Game reset

    def _reset(self):
        self.board       = new_board()
        self.turn        = True          # Blue always goes first
        self.selected    = None          # (col, row) of selected piece
        self.valid_dests = {}            # (col, row) -> action
        self.clock       = 0             # halfmove clock (plies since a clone)
        self._last_pass  = None
        self._blue_result = None         # +1/0/-1 (Blue's perspective) at game end
        self._dirty       = False        # touched by edit mode => not recorded
        self._recorded    = False        # guard against double-recording
        if self.mcts_agent:
            self.mcts_agent.root = None  # discard old search tree
        if self.player_mcts and self.player_mcts is not self.mcts_agent:
            self.player_mcts.root = None

        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = "Your turn - click a piece"
        else:
            self.state  = S_AI
            self.status = "AI thinking…"
            self._start_ai_move()

    def _is_human_turn(self):
        if self.ai_vs_ai:
            return False
        return self.turn == self.human_blue

    def _human_cell(self):
        return BLUE if self.human_blue else GREEN

    def _fmt_score(self):
        b, g = count_cells(self.board)
        return f"Blue {b} – Green {g}"

    #  AI helpers

    def _get_current_ai(self):
        """Returns (mcts_agent, engine, depth) for the player whose turn it is."""
        if self.turn and self.ai_vs_ai:
            return self.player_mcts, self.player_engine, self.player_depth
        return self.mcts_agent, self.engine, self.mm_depth

    def _current_ai_label(self):
        """Label string for the AI whose turn self.turn currently is."""
        if self.turn and self.ai_vs_ai:
            engine, depth = self.player_engine, self.player_depth
            is_mcts = (engine == "mcts")
        else:
            is_mcts = (self.opponent == "mcts")
            engine, depth = self.engine, self.mm_depth

        if is_mcts:
            return "MCTS"
        elif engine == "stauf":
            return f"Stauf-{depth}"
        elif engine == "micro4t":
            return f"Micro4t-{depth}ms"
        elif engine == "micro3":
            return f"Micro3-{depth}"
        elif engine == "hmcts":
            return f"HMCTS-{depth}"
        else:
            return f"Micro4-{depth}"

    #  Keyboard

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

    #  Mouse

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
                # Click same piece again -> deselect
                self._deselect()
            elif (col, row) in self.valid_dests:
                self._human_move(self.valid_dests[(col, row)])
            elif np.array_equal(self.board[row, col], self._human_cell()):
                # Click a different own piece -> reselect
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

    #  Edit mode

    def _enter_edit(self):
        self.selected    = None
        self.valid_dests = {}
        self.state       = S_EDIT
        self._dirty      = True   # hand-edited position: don't record this game
        self._edit_status()

    def _edit_status(self):
        player = "Blue" if self.turn else "Green"
        self.status = (f"EDIT - {player} to move  [{self._fmt_score()}]  "
                       f"[click=cycle  T=swap player  E=resume]")

    def _edit_cycle_cell(self, col, row):
        """Cycle cell at (col, row): blue -> green -> empty -> blue."""
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
        if self.player_mcts and self.player_mcts is not self.mcts_agent:
            self.player_mcts.root = None
        self.clock       = 0
        self.selected    = None
        self.valid_dests = {}
        self._last_pass  = None

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

    #  Move legality

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

    #  Move application

    def _human_move(self, action):
        fx, fy, tx, ty, jump = action_to_move(action)
        label = "jump" if jump else "clone"
        self._apply_move(action)
        if self.state == S_OVER:
            return
        move_info = f"You: ({fx},{fy})->({tx},{ty}) [{label}]  [{self._fmt_score()}]"
        if self._last_pass:
            move_info += f"  ({self._last_pass} passes)"
        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = f"{move_info}  |  Your turn"
        else:
            self.state  = S_AI
            self.status = f"{move_info}  |  AI thinking…"
            self._start_ai_move()

    def _finish_ai_move(self, action):
        if action is None:
            # Forced pass: AI has no legal moves but game is not over
            passer = "Blue" if self.turn else "Green"
            self.turn = not self.turn
            self.clock += 1
            msg = f"{passer} passes (no moves)  [{self._fmt_score()}]"
            if self._is_human_turn():
                self.state  = S_SELECT
                self.status = f"{msg}  |  Your turn"
            else:
                self.state  = S_AI
                self.status = f"{msg}  |  AI thinking…"
                if self.ai_vs_ai:
                    pyglet.clock.schedule_once(lambda dt: self._start_ai_move(), self.ai_delay)
                else:
                    self._start_ai_move()
            return

        fx, fy, tx, ty, jump = action_to_move(action)
        label    = "jump" if jump else "clone"
        ai_label = self._current_ai_label()
        self._apply_move(action)
        if self.state == S_OVER:
            return
        move_info = f"{ai_label}: ({fx},{fy})->({tx},{ty}) [{label}]  [{self._fmt_score()}]"
        if self._last_pass:
            move_info += f"  ({self._last_pass} passes)"
        if self._is_human_turn():
            self.state  = S_SELECT
            self.status = f"{move_info}  |  Your turn"
        else:
            self.state  = S_AI
            self.status = f"{move_info}  |  AI thinking…"
            if self.ai_vs_ai:
                pyglet.clock.schedule_once(lambda dt: self._start_ai_move(), self.ai_delay)
            else:
                self._start_ai_move()

    def _apply_move(self, action):
        """Apply action, flip turn, check clock + terminal + forced pass."""
        self.board = apply_move(self.board, action, self.turn)
        if self.mcts_agent:
            self.mcts_agent.advance_tree(action)
        if self.player_mcts and self.player_mcts is not self.mcts_agent:
            self.player_mcts.advance_tree(action)
        self.selected    = None
        self.valid_dests = {}
        self._last_pass  = None
        self.turn        = not self.turn

        # Halfmove clock (libataxx rule): drawn after CLOCK_LIMIT plies
        # without a clone move.
        self.clock = tick_clock(self.clock, action)
        if self.clock >= CLOCK_LIMIT:
            self._blue_result = 0
            self._end_game(f"{CLOCK_LIMIT} plies without a clone - draw")
            return

        # Terminal state
        is_terminal, terminal_value = check_terminal(self.board, self.turn)
        if is_terminal:
            blue_val = terminal_value if self.turn else -terminal_value
            self._blue_result = 1 if blue_val > 0 else (-1 if blue_val < 0 else 0)
            if blue_val > 0:
                self._end_game("Blue wins!")
            elif blue_val < 0:
                self._end_game("Green wins!")
            else:
                self._end_game("Draw!")
            return

        # Forced pass: current player has no legal moves (game is not terminal,
        # so the *other* player must have moves — they just continue).
        if not np.any(action_masks(self.board, self.turn)):
            self._last_pass = "Blue" if self.turn else "Green"
            self.turn = not self.turn
            self.clock += 1

    def _end_game(self, msg):
        self.state  = S_OVER
        self.status = f"GAME OVER - {msg}  ({self._fmt_score()})  |  [R] to restart"
        if (self.rate_id and not self._dirty and not self._recorded
                and self._blue_result is not None):
            self._recorded = True
            self._rate_record()

    #  Rating orchestration

    def _rate_apply_opponent(self, opp):
        """Switch the Green opponent to ``opp`` (player_id, kind, payload, elo)."""
        oid, kind, payload, elo = opp
        self.cur_opp_id  = oid
        self.cur_opp_elo = elo
        if kind == "net":
            self.mcts_agent = _load_rate_agent(payload, self.device)
            self.opponent = self.engine = "mcts"
        elif kind == "stauf":
            self.mcts_agent = None
            self.opponent = self.engine = "stauf"
            self.mm_depth = payload
        else:                                   # deterministic minimax (MMd)
            self.mcts_agent = None
            self.opponent = self.engine = "micro3"
            self.mm_depth = payload

    def _rate_pick_next(self):
        """Adaptive matchmaking: a rated opponent near the current estimate,
        avoiding an immediate rematch, with a little jitter for game variety."""
        cands = sorted(self.rate_pool, key=lambda c: abs(c[3] - self.rate_est))
        near = cands[:3] or cands
        choices = [c for c in near if c[0] != self.cur_opp_id] or near
        return random.choice(choices)

    def _rate_refit(self):
        """Re-fit the whole DB (cheap, pure-numpy) to update the live estimate."""
        names, counts = edb.load_counts(self.rate_chash)
        if self.rate_id not in names:
            return
        reg = edb.load_players()
        elo, hess = edb.fit_whr(names, counts, anchors=_fit_anchors(names, reg))
        i = names.index(self.rate_id)
        self.rate_est = float(elo[i])
        self.rate_ci  = float(edb.whr_ci95(hess)[i])

    def _rate_record(self):
        """Append the finished game, refit, alternate colour, pick next foe."""
        hr = self._blue_result if self.human_blue else -self._blue_result
        edb.append_matches([{
            "a": self.rate_id, "b": self.cur_opp_id,
            "a_is_blue": self.human_blue, "result": int(hr),
            "config_hash": self.rate_chash,
        }])
        self.rate_n += 1
        self.rate_w += hr > 0
        self.rate_l += hr < 0
        self.rate_d += hr == 0
        self._rate_refit()
        ci = "inf" if not np.isfinite(self.rate_ci) else f"{self.rate_ci:.0f}"
        print(f"game {self.rate_n}: {'W' if hr>0 else 'L' if hr<0 else 'D'} "
              f"vs {self.cur_opp_id} ({self.cur_opp_elo:.0f})  ->  "
              f"est {self.rate_est:.0f} +/-{ci}  ({self.rate_w}-{self.rate_d}-{self.rate_l})",
              flush=True)
        self.human_blue = not self.human_blue      # cancel first-move bias
        self._rate_apply_opponent(self._rate_pick_next())

    def _rate_hud(self):
        ci = "?" if not np.isfinite(self.rate_ci) else f"{self.rate_ci:.0f}"
        you = "Blue" if self.human_blue else "Green"
        return (f"RATE {self.rate_name}:  Elo {self.rate_est:.0f} +/-{ci}   "
                f"n={self.rate_n}  W{self.rate_w} D{self.rate_d} L{self.rate_l}   "
                f"next: {self.cur_opp_id} ({self.cur_opp_elo:.0f})  you={you}")

    def on_close(self):
        if self.rate_id and self.rate_n:
            ci = "inf" if not np.isfinite(self.rate_ci) else f"{self.rate_ci:.0f}"
            print(f"\n=== {self.rate_id}: {self.rate_est:.0f} +/-{ci} Elo over "
                  f"{self.rate_n} games ({self.rate_w}-{self.rate_d}-{self.rate_l}) ===")
            print(f"Refit anytime with: python scripts/eval_db.py fit --anchor "
                  f"{edb.STAUF_PLAYER_ID}={edb.STAUF_ANCHOR_ELO:.0f}")
        super().on_close()

    #  AI background thread

    def _start_ai_move(self):
        board_snap = self.board.copy()
        turn_snap  = self.turn
        clock_snap = self.clock
        cur_mcts, cur_engine, cur_depth = self._get_current_ai()
        result = [None]
        done   = [False]

        def compute():
            if cur_engine == "mcts" and cur_mcts:
                probs  = cur_mcts.search(board_snap, turn_snap, clock=clock_snap)
                action = cur_mcts.select_action(probs, temperature=0,
                                                best_action=cur_mcts.last_best_action)
            else:
                board_bytes = board_snap.tobytes()
                action = find_best_move(board_bytes, cur_depth, turn_snap, cur_engine)
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

    #  Drawing

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

        # Rating HUD (top strip)
        if self.rate_id:
            pyglet.text.Label(
                self._rate_hud(),
                font_name="Arial", font_size=10,
                x=10, y=WIN_H - 12,
                anchor_x="left", anchor_y="center",
                color=(235, 205, 90, 255),
                width=WIN_W - 20, multiline=False,
            ).draw()


#  Entry point

def _build_mcts(checkpoint, simulations, device=None):
    if device is None:
        device = torch.device("cpu")
    print(f"Loading {checkpoint}  (device: {device})")
    network = _load_network(checkpoint, device)
    agent = MCGS(network, num_simulations=simulations)
    print(f"MCTS agent ready  ({simulations} sims/move)")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Play Microscope against AI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to opponent MCTS checkpoint (.pt)")
    parser.add_argument("--opponent", choices=["mcts", "micro3", "micro4t", "stauf", "hmcts"], default="mcts",
                        help="Opponent (Green) type (default: mcts)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Opponent search depth (default: 2)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="Opponent MCTS simulations per move (default: 100)")
    parser.add_argument("--human-color", choices=["blue", "green"], default="blue",
                        help="Color you play as — ignored when --player is set (default: blue)")
    # Player (Blue) AI — when set, enables AI vs AI mode
    parser.add_argument("--player", choices=["mcts", "micro3", "micro4t", "stauf", "hmcts"],
                        default=None,
                        help="Player (Blue) AI type; omit to play as human")
    parser.add_argument("--player-checkpoint", type=str, default=None,
                        help="MCTS checkpoint for the Blue player")
    parser.add_argument("--player-simulations", type=int, default=100,
                        help="Blue player MCTS simulations per move (default: 100)")
    parser.add_argument("--player-depth", type=int, default=2,
                        help="Blue player search depth (default: 2)")
    parser.add_argument("--ai-delay", type=float, default=0.5,
                        help="Seconds to pause between AI moves in AI vs AI mode (default: 0.5)")
    # Rating mode
    parser.add_argument("--rate", type=str, default=None, metavar="NAME",
                        help="Rate a human player: record games to the eval DB as "
                             "player 'human/NAME' and show a live Elo estimate")
    parser.add_argument("--rate-first-stauf", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Play game 1 vs the stauf gauge for a direct anchor pin "
                             "(default on)")
    parser.add_argument("--data-dir", type=str, default=None, metavar="DIR",
                        help="Root of the models+eval-DB bundle shipped separately "
                             "from the wheel (else searched via T7G_DATA_DIR / cwd)")
    args = parser.parse_args()

    if args.data_dir:
        paths.set_data_root(args.data_dir)

    device = torch.device("cpu")

    # --- Rating mode: adaptive matchmaking over the rated eval-DB ladder ---
    if args.rate:
        chash = _rate_config_hash()
        pool = build_rating_pool(chash)
        if not pool:
            print(f"Error: no rated opponents at config {chash}. "
                  f"Run `python scripts/eval_db.py add ...` first.")
            return 1
        rate_id = f"human/{args.rate}"
        print(f"Data: {paths.describe()}")
        print(f"Rating {rate_id}: {len(pool)} rated opponents, "
              f"{pool[0][3]:.0f}..{pool[-1][3]:.0f} Elo  (config {chash})")
        print("Play a handful of games; opponents adapt toward your level. "
              "Esc when done.")
        MicroscopeApp(
            human_color="blue", rate_id=rate_id, rate_pool=pool,
            rate_first_stauf=args.rate_first_stauf, device=device,
        )
        pyglet.app.run()
        return 0

    # --- Build opponent (Green) ---
    mcts_agent = None
    if args.opponent == "mcts":
        checkpoint = args.checkpoint
        if not checkpoint:
            bundled = pathlib.Path(__file__).parent.parent / "lib" / "best.pt"
            if bundled.exists():
                checkpoint = str(bundled)
            else:
                print("Error: no --checkpoint given and no bundled model found at lib/best.pt")
                return 1
        mcts_agent = _build_mcts(checkpoint, args.simulations, device)

    if args.opponent == "stauf":
        engine = "stauf"
    elif args.opponent == "micro3":
        engine = "micro3"
    elif args.opponent == "hmcts":
        engine = "hmcts"
    elif args.opponent == "micro4t":
        engine = "micro4t"
        args.depth = 1000
    else:
        engine = "minimax"

    # --- Build player (Blue) ---
    player_mcts   = None
    player_engine = "human"
    player_depth  = args.player_depth

    if args.player is not None:
        player_engine = args.player
        if args.player == "mcts":
            pcheckpoint = args.player_checkpoint
            if not pcheckpoint:
                bundled = pathlib.Path(__file__).parent.parent / "lib" / "best.pt"
                if bundled.exists():
                    pcheckpoint = str(bundled)
                else:
                    print("Error: --player mcts requires --player-checkpoint (no bundled model found)")
                    return 1
            player_mcts = _build_mcts(pcheckpoint, args.player_simulations, device)
        elif args.player == "micro4t":
            player_depth = 1000

    # --- Announce ---
    if args.player:
        if args.player == "mcts":
            blue_label = f"MCTS ({args.player_simulations} sims)"
        elif args.player == "micro4t":
            blue_label = "Micro4t-1000ms"
        else:
            blue_label = f"{args.player}-{args.player_depth}"
        if args.opponent == "mcts":
            green_label = f"MCTS ({args.simulations} sims)"
        elif args.opponent == "micro4t":
            green_label = "Micro4t-1000ms"
        else:
            green_label = f"{args.opponent}-{args.depth}"
        print(f"AI vs AI:  Blue={blue_label}  Green={green_label}  delay={args.ai_delay}s")
    else:
        human_label = args.human_color.capitalize()
        if args.opponent == "mcts":
            ai_label = f"MCTS ({args.simulations} sims)"
        elif args.opponent == "hmcts":
            ai_label = f"hmcts-{args.depth}"
        elif args.opponent == "stauf":
            ai_label = f"Stauf-{args.depth}"
        elif args.opponent == "micro4t":
            ai_label = "Micro4t-1000ms"
        else:
            ai_label = f"{args.opponent}-{args.depth}"
        print(f"You play as {human_label},  opponent: {ai_label}")

    print("Controls: click to select & move  |  R = restart  |  Esc = quit")

    MicroscopeApp(
        mcts_agent=mcts_agent,
        opponent=args.opponent,
        minimax_depth=args.depth,
        human_color=args.human_color,
        engine=engine,
        player_mcts=player_mcts,
        player_engine=player_engine,
        player_depth=player_depth,
        ai_delay=args.ai_delay,
    )
    pyglet.app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
