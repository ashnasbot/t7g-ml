"""Threaded web server to play Microscope (7x7 Ataxx) in a browser.

A browser-hostable twin of ``scripts/play_gui.py``:
the board renders in the page, with all logic reusing the exact same
`lib` functions the GUI does.

The work is blocking and CPU-bound (torch inference / the C engines), so a
threaded server is the right shape: each request runs in its own thread,
a slow `/ai-move` blocks that one thread and nobody else.

Scope: human-vs-AI with rating mode (adaptive-Elo matchmaking, like the GUI's
``--rate``).  This server points at its own eval DB (`webapp/eval_db`).

Run:
    python webapp/server.py                 # http://127.0.0.1:8000
    python webapp/server.py --port 9000 --host 0.0.0.0
"""
import argparse
import json
import os
import pathlib
import random
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBAPP_DIR = pathlib.Path(__file__).resolve().parent
REPO = WEBAPP_DIR.parent
sys.path.insert(0, str(REPO))

# Point lib.eval_db at the webapp's private DB *before* it resolves any path.
# paths.eval_db_dir() reads this env var at call time and requires the dir to
# exist, so we create+seed it below if missing.
os.environ.setdefault("T7G_EVAL_DB", str(WEBAPP_DIR / "eval_db"))

import numpy as np
import torch

from lib import eval_db as edb
from lib import paths
from lib.mcgs import MCGS
from lib.t7g import (BLUE, CLOCK_LIMIT, GREEN, action_masks, action_to_move,
                     apply_move, check_terminal, count_cells, find_best_move,
                     new_board, tick_clock)

DEVICE = torch.device("cpu")

# Deterministic minimax (micro3) rungs seeded to fill gaps in the ladder:
# two below the stauf floor (1000) so a beginner has something to beat, plus a
# mid rung bridging stauf -> MM7 (1392).  (id, search depth, seed Elo).  Seed
# Elos are hand-set and only steer matchmaking until real games refine them --
# they are NOT fixed_elo anchors, so they never distort the human fit.
SEED_OPPONENTS = [
    ("MM1", 1, 500.0),
    ("MM2", 2, 800.0),
    ("MM5", 5, 1200.0),
]

_NET_CACHE: dict = {}      # checkpoint path -> MCGS agent (shared across sessions)
_DB_LOCK = threading.Lock()   # serialise eval-DB appends across session threads


#  Board <-> JSON  (7x7x2 bool array <-> 7x7 grid of 0/1/2)

def board_to_grid(board):
    """(7,7,2) bool board -> 7x7 ints: 0 empty, 1 blue, 2 green."""
    grid = np.zeros((7, 7), dtype=int)
    grid[np.all(board == BLUE, axis=2)] = 1
    grid[np.all(board == GREEN, axis=2)] = 2
    return grid.tolist()


def legal_moves_map(board, turn):
    """{"col,row": [{"to":[c,r], "action":int, "jump":bool}, ...]} for `turn`.

    Keyed by source cell so the browser can highlight destinations on click and
    hand back the exact action index for the chosen (source, dest) pair.
    """
    masks = action_masks(board, turn)
    out: dict = {}
    for action in np.nonzero(masks)[0]:
        fx, fy, tx, ty, jump = action_to_move(int(action))
        out.setdefault(f"{fx},{fy}", []).append(
            {"to": [tx, ty], "action": int(action), "jump": bool(jump)})
    return out


#  Rating helpers (lifted from scripts/play_gui.py, GUI-free)

def _load_network(checkpoint, device):
    from lib.device_utils import load_compiled_network
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    state = blob["network"] if isinstance(blob, dict) and "network" in blob else blob
    net, _ = load_compiled_network(state, device, num_actions=1225, compile_net=False)
    return net


def _rate_config_hash():
    return edb.config_hash({})


def _ensure_seed_opponents():
    """Register the seeded minimax rungs in the webapp DB (idempotent).

    Merges a ``seed_elo`` (and kind/depth) into each; build_rating_pool injects
    them into the pool while they have no games, then defers to the WHR fit."""
    for pid, depth, seed in SEED_OPPONENTS:
        edb.register_player(pid, {"kind": "mm", "depth": depth,
                                  "name": pid, "seed_elo": seed})


def _load_rate_agent(path, device):
    if path in _NET_CACHE:
        return _NET_CACHE[path]
    cfg = edb.DEFAULT_CONFIG
    network = _load_network(path, device)
    agent = MCGS(network, num_simulations=cfg["sims"], c_puct=cfg["c_puct"],
                 gumbel_k=cfg["gumbel_k"], sigma_scale=cfg["sigma_scale"],
                 completion_n0=cfg["completion_n0"])
    _NET_CACHE[path] = agent
    return agent


def _fit_anchors(names, reg):
    a = {n: reg[n]["fixed_elo"] for n in names
         if reg.get(n, {}).get("fixed_elo") is not None}
    return a or {edb.STAUF_PLAYER_ID: edb.STAUF_ANCHOR_ELO}


def build_rating_pool(chash):
    """Rated, runnable opponents at the canonical config, sorted by Elo.

    Entry: (player_id, kind, payload, elo) with kind in {'net','stauf','mm'}.
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
            path = paths.find_checkpoint(pid)
            if path is not None:
                pool.append((pid, "net", str(path), e))

    # Gameless seeded rungs: opponents registered with a ``seed_elo`` but no
    # games yet in the DB fit (e.g. the weak sub-stauf MMs).  Inject them so a
    # fresh human can be matched below the stauf floor; once they accrue games
    # the WHR fit above rates them for real and this branch skips them.
    present = {c[0] for c in pool}
    for pid, meta in reg.items():
        if pid in present or meta.get("seed_elo") is None:
            continue
        seed = float(meta["seed_elo"])
        if pid == edb.STAUF_PLAYER_ID:
            pool.append((pid, "stauf", meta.get("depth", edb.STAUF_DEPTH), seed))
        elif meta.get("kind") == "mm":
            pool.append((pid, "mm", meta["depth"], seed))

    pool.sort(key=lambda c: c[3])
    return pool


#  Game session

class GameSession:
    """One browser's game + rating state.  Mirrors the non-drawing half of
    scripts/play_gui.py's MicroscopeApp: same rules, same matchmaking."""

    def __init__(self, rate_id, rate_pool, rate_first_stauf=True):
        self.lock = threading.Lock()
        self.device = DEVICE
        self.rate_id = rate_id
        self.rate_pool = rate_pool
        self.rate_chash = _rate_config_hash()
        self.rate_name = rate_id.split("/")[-1]

        # Opponent (set by _apply_opponent)
        self.mcts_agent = None
        self.engine = "stauf"
        self.mm_depth = edb.STAUF_DEPTH
        self.cur_opp_id = None
        self.cur_opp_elo = 0.0

        # Rating estimate + record
        elos = [c[3] for c in rate_pool]
        self.rate_est = float(np.median(elos)) if elos else 1000.0
        self.rate_ci = float("inf")
        self.rate_n = self.rate_w = self.rate_d = self.rate_l = 0
        self.human_blue = True

        edb.register_player(rate_id, {"kind": "human", "name": self.rate_name})
        self.rate_career_n = 0
        self._refit()          # continue from the DB if this player has history
        first = None
        if rate_first_stauf:
            first = next((c for c in rate_pool if c[1] == "stauf"), None)
        self._apply_opponent(first or self._pick_next())
        self.reset()

    #  Reset / turn bookkeeping

    def reset(self):
        self.board = new_board()
        self.turn = True                 # Blue always moves first
        self.clock = 0
        self._last_pass = None
        self._blue_result = None
        self._recorded = False
        self.game_over = False
        self.result_msg = ""
        if self.mcts_agent:
            self.mcts_agent.root = None

    def _human_turn(self):
        return (not self.game_over) and (self.turn == self.human_blue)

    def _score(self):
        b, g = count_cells(self.board)
        return {"blue": int(b), "green": int(g)}

    #  Move application (faithful to MicroscopeApp._apply_move)

    def _apply_move(self, action):
        self.board = apply_move(self.board, action, self.turn)
        if self.mcts_agent:
            self.mcts_agent.advance_tree(action)
        self._last_pass = None
        self.turn = not self.turn

        self.clock = tick_clock(self.clock, action)
        if self.clock >= CLOCK_LIMIT:
            self._blue_result = 0
            self._end_game(f"{CLOCK_LIMIT} plies without a clone - draw")
            return

        is_terminal, terminal_value = check_terminal(self.board, self.turn)
        if is_terminal:
            blue_val = terminal_value if self.turn else -terminal_value
            self._blue_result = 1 if blue_val > 0 else (-1 if blue_val < 0 else 0)
            self._end_game("Blue wins!" if blue_val > 0 else
                           "Green wins!" if blue_val < 0 else "Draw!")
            return

        # Forced pass: side to move has no legal moves (game not terminal, so
        # the other side does) -- skip them.
        if not np.any(action_masks(self.board, self.turn)):
            self._last_pass = "Blue" if self.turn else "Green"
            self.turn = not self.turn
            self.clock += 1

    def _end_game(self, msg):
        self.game_over = True
        self.result_msg = msg
        if not self._recorded and self._blue_result is not None:
            self._recorded = True
            self._record()

    #  Human + AI moves

    def human_move(self, action):
        if not self._human_turn():
            return
        legal = action_masks(self.board, self.turn)
        if action < 0 or action >= len(legal) or not legal[action]:
            return
        self._apply_move(action)

    def ai_move(self):
        """Compute + apply one move for the side to move (must be the AI)."""
        if self.game_over or self._human_turn():
            return
        board_snap = self.board.copy()
        turn_snap = self.turn
        clock_snap = self.clock
        if self.engine == "mcts" and self.mcts_agent:
            probs = self.mcts_agent.search(board_snap, turn_snap, clock=clock_snap)
            action = self.mcts_agent.select_action(
                probs, temperature=0, best_action=self.mcts_agent.last_best_action)
        else:
            action = find_best_move(board_snap.tobytes(), self.mm_depth,
                                    turn_snap, self.engine)
            if action in (-1, 1225):
                action = None
        if action is None:
            # AI has no legal move but game isn't over: pass.
            self._last_pass = "Blue" if self.turn else "Green"
            self.turn = not self.turn
            self.clock += 1
            return
        self._apply_move(action)

    #  Rating orchestration (MicroscopeApp._rate_* )

    def _apply_opponent(self, opp):
        oid, kind, payload, elo = opp
        self.cur_opp_id = oid
        self.cur_opp_elo = elo
        if kind == "net":
            self.mcts_agent = _load_rate_agent(payload, self.device)
            self.engine = "mcts"
        elif kind == "stauf":
            self.mcts_agent = None
            self.engine = "stauf"
            self.mm_depth = payload
        else:                                    # deterministic minimax (MMd)
            self.mcts_agent = None
            self.engine = "micro3"
            self.mm_depth = payload

    # Match within the tightest Elo band around the estimate that has a
    # candidate, widening only when empty.  This keeps games close to the
    # player's level -- a random pick among the *nearest few* used to reach far
    # above a player who had just dropped (e.g. straight to a 1600 net after a
    # loss to stauf) whenever the pool was sparse near their rating.
    MATCH_BANDS = (150, 300, 600)

    def _pick_next(self):
        for band in self.MATCH_BANDS:
            cands = [c for c in self.rate_pool
                     if abs(c[3] - self.rate_est) <= band and c[0] != self.cur_opp_id]
            if cands:
                return random.choice(cands)
        # Estimate sits beyond every band edge: take the single nearest
        # opponent (still avoiding an immediate rematch if possible).
        others = [c for c in self.rate_pool if c[0] != self.cur_opp_id] or self.rate_pool
        return min(others, key=lambda c: abs(c[3] - self.rate_est))

    def _refit(self):
        names, counts = edb.load_counts(self.rate_chash)
        if self.rate_id not in names:
            self.rate_career_n = 0          # no games in the DB yet: unrated
            return
        reg = edb.load_players()
        elo, hess = edb.fit_whr(names, counts, anchors=_fit_anchors(names, reg))
        i = names.index(self.rate_id)
        self.rate_est = float(elo[i])
        self.rate_ci = float(edb.whr_ci95(hess)[i])
        # Total games behind this rating (all sessions), so the displayed count
        # matches what the fit is actually based on -- not the session counter.
        self.rate_career_n = int(sum(
            wa + d + wb for (a, b), (wa, d, wb) in counts.items()
            if a == i or b == i))

    def _record(self):
        hr = self._blue_result if self.human_blue else -self._blue_result
        with _DB_LOCK:
            edb.append_matches([{
                "a": self.rate_id, "b": self.cur_opp_id,
                "a_is_blue": self.human_blue, "result": int(hr),
                "config_hash": self.rate_chash,
            }])
            self._refit()
        self.rate_n += 1
        self.rate_w += hr > 0
        self.rate_l += hr < 0
        self.rate_d += hr == 0
        ci = "inf" if not np.isfinite(self.rate_ci) else f"{self.rate_ci:.0f}"
        print(f"[{self.rate_id}] {'W' if hr>0 else 'L' if hr<0 else 'D'} vs "
              f"{self.cur_opp_id} ({self.cur_opp_elo:.0f}) -> est "
              f"{self.rate_est:.0f} +/-{ci}  ({self.rate_career_n} games; "
              f"session {self.rate_w}-{self.rate_d}-{self.rate_l})", flush=True)
        self.human_blue = not self.human_blue          # cancel first-move bias
        self._apply_opponent(self._pick_next())

    #  Serialisation for the client

    def rating(self):
        # A player is "rated" once they have any games in the DB; a brand-new
        # player's rate_est is only the pool-median prior, so flag that so the
        # client shows "unrated" rather than a misleading number.
        rated = self.rate_career_n > 0
        ci = None if not np.isfinite(self.rate_ci) else round(self.rate_ci)
        return {
            "name": self.rate_name, "rated": rated,
            "elo": round(self.rate_est) if rated else None, "ci": ci,
            "games": self.rate_career_n,                     # career games (fit basis)
            "sw": self.rate_w, "sd": self.rate_d, "sl": self.rate_l,  # this session
            "opponent": self.cur_opp_id, "opponent_elo": round(self.cur_opp_elo),
        }

    def state(self):
        human_turn = self._human_turn()
        your_color = "blue" if self.human_blue else "green"
        if self.game_over:
            status = f"Game over - {self.result_msg}"
        elif human_turn:
            status = "Your turn"
        else:
            status = f"{self.cur_opp_id} thinking..."
        if self._last_pass and not self.game_over:
            status = f"{self._last_pass} passed - {status}"
        return {
            "board": board_to_grid(self.board),
            "turn": "blue" if self.turn else "green",
            "your_color": your_color,
            "human_turn": human_turn,
            "game_over": self.game_over,
            "score": self._score(),
            "status": status,
            "result": self.result_msg if self.game_over else None,
            "legal": legal_moves_map(self.board, self.turn) if human_turn else {},
            "rating": self.rating(),
        }


#  HTTP layer

SESSIONS: dict = {}          # session_id -> GameSession
_SESS_LOCK = threading.Lock()
_INDEX_HTML = (WEBAPP_DIR / "static" / "index.html").read_text()
_RATE_POOL_CACHE = None      # built once (WHR fit is the slow part)


def _rate_pool():
    global _RATE_POOL_CACHE
    if _RATE_POOL_CACHE is None:
        _RATE_POOL_CACHE = build_rating_pool(_rate_config_hash())
    return _RATE_POOL_CACHE


class Handler(BaseHTTPRequestHandler):
    server_version = "microscope-web/1.0"

    def log_message(self, *a):        # quieter default logging
        pass

    #  helpers

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            return {}

    def _session(self, sid):
        with _SESS_LOCK:
            return SESSIONS.get(sid)

    #  routing

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(_INDEX_HTML)
        elif self.path == "/api/pool":
            pool = _rate_pool()
            self._send_json({"count": len(pool),
                             "min_elo": round(pool[0][3]) if pool else None,
                             "max_elo": round(pool[-1][3]) if pool else None})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        # /api/new
        if self.path == "/api/new":
            return self._new()
        # /api/<sid>/<action>
        if len(parts) == 3 and parts[0] == "api":
            sid, action = parts[1], parts[2]
            sess = self._session(sid)
            if sess is None:
                return self._send_json({"error": "unknown session"}, 404)
            if action == "move":
                return self._move(sess)
            if action == "ai-move":
                return self._ai_move(sess)
            if action == "reset":
                return self._reset(sess)
        self._send_json({"error": "not found"}, 404)

    #  endpoints

    def _new(self):
        data = self._read_json()
        name = str(data.get("name", "")).strip() or "guest"
        pool = _rate_pool()
        if not pool:
            return self._send_json(
                {"error": "no rated opponents in webapp/eval_db"}, 400)
        rate_id = f"human/{name}"
        sess = GameSession(rate_id, pool,
                           rate_first_stauf=bool(data.get("first_stauf", True)))
        sid = uuid.uuid4().hex
        with _SESS_LOCK:
            SESSIONS[sid] = sess
        with sess.lock:
            self._send_json({"session_id": sid, **sess.state()})

    def _move(self, sess):
        data = self._read_json()
        action = data.get("action")
        with sess.lock:
            if isinstance(action, int):
                sess.human_move(action)
            self._send_json(sess.state())

    def _ai_move(self, sess):
        with sess.lock:                 # blocking: this thread does the thinking
            sess.ai_move()
            self._send_json(sess.state())

    def _reset(self, sess):
        with sess.lock:
            sess.reset()
            self._send_json(sess.state())


def main():
    ap = argparse.ArgumentParser(description="Web server to play Microscope")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # Ensure the private eval DB exists (paths.eval_db_dir needs a real dir).
    pathlib.Path(os.environ["T7G_EVAL_DB"]).mkdir(parents=True, exist_ok=True)
    _ensure_seed_opponents()

    pool = _rate_pool()
    print(f"eval DB: {paths.eval_db_dir()}")
    if pool:
        print(f"{len(pool)} rated opponents, "
              f"{pool[0][3]:.0f}..{pool[-1][3]:.0f} Elo")
    else:
        print("WARNING: no rated opponents found - seed webapp/eval_db first")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
