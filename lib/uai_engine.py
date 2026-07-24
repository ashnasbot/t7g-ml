"""
Subprocess wrapper for external UAI (Universal Ataxx Interface) engines.

Drives a persistent engine child process over the UAI protocol
(uai / isready / position fen ... / go / bestmove), translating between our
(7,7,2) bool board + 1225-action encoding and the engines' FEN / square
notation. See 3rd_party/{autaxx,tiktaxx,Scarletxx} for the vendored engines
and build instructions.

Each call to find_best_move() sends the full current position as a FEN
string rather than an incremental move list, so no game/session state is
shared between calls -- safe to reuse one engine instance across many
games in a worker process.
"""
import pathlib
import queue
import subprocess
import threading
import time

import numpy as np

from lib.t7g import Board, action_masks, action_to_move

_THIRD_PARTY = pathlib.Path(__file__).parent.parent / "3rd_party"


def board_to_fen(board: Board, turn: bool) -> str:
    """Convert our board + turn to a UAI FEN string.

    Board row y=0 is the engine's top FEN rank (rank 7); column x=0 is
    file 'a'. Blue (board plane 1) <-> 'x'; Green (board plane 0) <-> 'o'.
    """
    rows = []
    for y in range(7):
        parts = []
        empty = 0
        for x in range(7):
            if board[y, x, 1]:
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append('x')
            elif board[y, x, 0]:
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append('o')
            else:
                empty += 1
        if empty:
            parts.append(str(empty))
        rows.append("".join(parts))
    side = 'x' if turn else 'o'
    return "/".join(rows) + f" {side} 0 1"


def _square_to_xy(square: str) -> tuple[int, int]:
    x = ord(square[0]) - ord('a')
    y = 7 - int(square[1])
    return x, y


def parse_uai_move(move: str, board: Board, turn: bool) -> int:
    """Convert a UAI move string ('a6' clone, or 'f2e4' jump) to our action index.

    Clone moves only specify the destination (the source piece isn't
    removed), so any legal clone into that square yields the same
    resulting board; we just need *a* matching legal action, not the
    exact one the engine "meant".
    """
    to_x, to_y = _square_to_xy(move[-2:])
    from_xy = _square_to_xy(move[:2]) if len(move) == 4 else None

    legal = np.where(action_masks(board, turn))[0]
    for action in legal:
        fx, fy, tx, ty, jump = action_to_move(int(action))
        if (tx, ty) != (to_x, to_y):
            continue
        if from_xy is not None:
            if (fx, fy) == from_xy:
                return int(action)
        elif not jump:
            return int(action)
    raise ValueError(f"UAI move {move!r} not found among legal actions")


class UAISearchTimeout(RuntimeError):
    """A single go/bestmove round-trip exceeded the search timeout."""


class UAIEngine:
    """Manages one persistent UAI-speaking engine subprocess.

    A pathological position can make some search modes (e.g. autaxx's plain
    alphabeta without NNUE-guided move ordering) blow up to minutes or hours
    at a given depth -- this happened during dev with autaxx-ab at depth 10.
    find_best_move() therefore bounds every search with a wall-clock timeout;
    on timeout the (likely wedged) subprocess is killed and respawned, and
    -1 (no move) is returned so the caller just treats it like a pass.
    """

    SEARCH_TIMEOUT_S = 20.0

    def __init__(self, binary: "str | pathlib.Path", setup_cmds: tuple = (),
                 go_mode: str = "depth"):
        self._binary = pathlib.Path(binary)
        if not self._binary.exists():
            raise FileNotFoundError(
                f"UAI engine binary not found at {self._binary}; build it "
                "first (see 3rd_party/<engine>/ for build instructions)."
            )
        self._setup_cmds = setup_cmds
        self._go_mode = go_mode  # "depth" (alpha-beta) or "movetime" (MCTS/UCT)
        self._lock = threading.Lock()
        self._spawn()

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [str(self._binary)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, cwd=str(self._binary.parent),
        )
        # Dedicated reader thread feeding a queue: a naive select() on the
        # pipe fd is unsafe here because Python's buffered TextIOWrapper can
        # already hold unread lines in its internal buffer while the OS fd
        # itself reports not-readable, causing spurious full-timeout stalls.
        # queue.get(timeout=...) has no such pitfall.
        self._out_q: "queue.Queue[str | None]" = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._send("uai")
        self._read_until("uaiok")
        for cmd in self._setup_cmds:
            self._send(cmd)
        self._send("isready")
        self._read_until("readyok")
        self._send("uainewgame")

    def _reader_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._out_q.put(line.rstrip("\n"))
        self._out_q.put(None)  # sentinel: EOF / process exited

    def _respawn(self) -> None:
        self._proc.kill()
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        self._spawn()

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_until(self, marker: str, timeout: "float | None" = None) -> list[str]:
        deadline = None if timeout is None else time.monotonic() + timeout
        lines = []
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                line = self._out_q.get(timeout=remaining)
            except queue.Empty:
                raise UAISearchTimeout(
                    f"no {marker!r} within {timeout}s, output so far:\n"
                    + "\n".join(lines)
                )
            if line is None:
                raise RuntimeError(
                    "UAI engine process exited unexpectedly, output so far:\n"
                    + "\n".join(lines)
                )
            lines.append(line)
            if line.startswith(marker):
                return lines

    def new_game(self) -> None:
        with self._lock:
            self._send("uainewgame")

    def find_best_move(self, board: Board, budget: int, as_blue: bool) -> int:
        """Return an action index (0-1224), or -1 if there are no legal moves.

        *budget* is plies for depth-based engines, milliseconds for
        movetime-based ones (see go_mode).
        """
        if not np.any(action_masks(board, as_blue)):
            return -1
        with self._lock:
            fen = board_to_fen(board, as_blue)
            self._send(f"position fen {fen}")
            self._send(f"go {self._go_mode} {budget}")
            try:
                lines = self._read_until("bestmove", timeout=self.SEARCH_TIMEOUT_S)
            except UAISearchTimeout:
                self._respawn()
                return -1
        move = lines[-1].split()[1]
        if move == "0000":
            return -1
        return parse_uai_move(move, board, as_blue)

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


# ---------------------------------------------------------------------------
# Known engine specs + per-process singleton cache
# ---------------------------------------------------------------------------

ENGINE_SPECS: dict[str, dict] = {
    # NOTE: autaxx's own default search mode is "tryhard" (see the binary's
    # `option name search ... default tryhard`; the vendored readme.md is stale
    # and predates both tryhard and nnue).  Its native budget is TIME, not
    # depth -- search::Settings defaults to Type::Time and spends clock/30 per
    # move -- so `go movetime` is the honest way to hand it a budget.  Fixed
    # depth is not comparable ACROSS modes: depth 10 is ~100 minutes for
    # alphabeta but ~300 ms for nnue.
    "autaxx-tryhard": dict(  # the engine's own default configuration
        binary=_THIRD_PARTY / "autaxx/build/src/autaxx/autaxx",
        setup_cmds=("setoption name search value tryhard",),
        go_mode="movetime",
    ),
    "autaxx-nnue-mt": dict(  # nnue eval, time-budgeted
        binary=_THIRD_PARTY / "autaxx/build/src/autaxx/autaxx",
        setup_cmds=("setoption name search value nnue",),
        go_mode="movetime",
    ),
    "autaxx": dict(
        binary=_THIRD_PARTY / "autaxx/build/src/autaxx/autaxx",
        setup_cmds=("setoption name search value nnue",),
        go_mode="depth",
    ),
    "autaxx-ab": dict(  # same binary, plain alpha-beta (no NNUE) for comparison
        binary=_THIRD_PARTY / "autaxx/build/src/autaxx/autaxx",
        setup_cmds=("setoption name search value alphabeta",),
        go_mode="depth",
    ),
    "tiktaxx": dict(
        binary=_THIRD_PARTY / "tiktaxx/bin/tiktaxx",
        setup_cmds=(),
        go_mode="depth",
    ),
    "scarlettxx": dict(
        binary=_THIRD_PARTY / "Scarletxx/build/scarlettxx",
        setup_cmds=(),
        go_mode="movetime",  # UCT engine -- depth isn't a meaningful budget
    ),
}

_worker_engines: dict[str, UAIEngine] = {}


def get_worker_engine(name: str) -> UAIEngine:
    """Lazily create/reuse the per-process engine instance for *name*."""
    if name not in _worker_engines:
        _worker_engines[name] = UAIEngine(**ENGINE_SPECS[name])
    return _worker_engines[name]
