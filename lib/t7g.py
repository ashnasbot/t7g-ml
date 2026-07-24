import ctypes
import numpy
import pathlib
import sys
import threading
from PIL import Image
try:
    import warnings
    warnings.filterwarnings("ignore")
    from term_image.image import AutoImage as _AutoImage
    _HAS_TERM_IMAGE = True
except ImportError:
    _HAS_TERM_IMAGE = False
import numpy.typing as npt
from typing import Any, Optional


BLUE = numpy.array([0, 1], dtype=bool)
GREEN = numpy.array([1, 0], dtype=bool)
CLEAR = numpy.array([0, 0], dtype=bool)

BLUE_GRID = numpy.full([7, 7, 2], BLUE)
GREEN_GRID = numpy.full([7, 7, 2], GREEN)

def draw_board(board: "npt.NDArray[numpy.bool_]") -> None:
    """Render the 7×7 board in the terminal using term_image (no-op if not installed)."""
    if not _HAS_TERM_IMAGE:
        return
    cell = 12  # pixels per cell in the rendered image
    img_arr = numpy.zeros((7 * cell, 7 * cell, 3), dtype=numpy.uint8)
    for gy in range(7):
        for gx in range(7):
            r0, r1 = gy * cell + 1, (gy + 1) * cell - 1
            c0, c1 = gx * cell + 1, (gx + 1) * cell - 1
            if board[gy, gx, 1]:          # Blue
                colour = (80, 140, 255)
            elif board[gy, gx, 0]:        # Green
                colour = (80, 210, 80)
            else:                          # Empty
                colour = (50, 40, 35)
            img_arr[r0:r1, c0:c1] = colour
    _AutoImage(Image.fromarray(img_arr, "RGB")).draw()


# Type aliases
Board = npt.NDArray[numpy.bool_]    # shape (7, 7, 2)
Obs = npt.NDArray[numpy.float32]   # shape (7, 7, 4)

# Precomputed index arrays for vectorised action_masks.
# _DEST_Y: (7,1,5,1)  _DEST_X: (1,7,1,5) - broadcast together to (7,7,5,5)
# where axes are [piece_y, piece_x, move_v, move_u].
_YS = numpy.arange(7).reshape(7, 1, 1, 1)
_XS = numpy.arange(7).reshape(1, 7, 1, 1)
_DVS = (numpy.arange(5) - 2).reshape(1, 1, 5, 1)
_DUS = (numpy.arange(5) - 2).reshape(1, 1, 1, 5)
_DEST_Y_RAW = _YS + _DVS          # (7, 1, 5, 1)
_DEST_X_RAW = _XS + _DUS          # (1, 7, 1, 5)
_IN_BOUNDS = (
    (_DEST_Y_RAW >= 0) & (_DEST_Y_RAW < 7) &
    (_DEST_X_RAW >= 0) & (_DEST_X_RAW < 7)
)                                  # (7, 7, 5, 5) after broadcast
_DEST_Y = numpy.clip(_DEST_Y_RAW, 0, 6)   # safe indices for empty_mask lookup
_DEST_X = numpy.clip(_DEST_X_RAW, 0, 6)


def _find_dll(name: str) -> pathlib.Path:
    """Locate a shared library, preferring the native (-march=native) build in the project root.

    Accepts a name with either .dll or .so extension (or no extension).  On
    Linux the .so variant is tried first; on Windows .dll is tried first.
    """
    stem = pathlib.Path(name).stem
    suffixes = [".dll"] if sys.platform == "win32" else [".so"]
    proj = pathlib.Path(__file__).parent.parent
    roots = [
        proj / "build",                            # native (-march=native) build
        pathlib.Path().absolute() / "build",       # native build, cwd-relative
        pathlib.Path(__file__).parent,             # portable build / local libs in lib/
    ]
    for root in roots:
        for suffix in suffixes:
            p = root / (stem + suffix)
            if p.exists():
                return p
    candidates = [r / (stem + s) for r in roots for s in suffixes]
    raise FileNotFoundError(f"Cannot locate {name}; searched: {candidates}")


class _LazyLib:
    """Wraps a C shared library and defers loading until first attribute access.

    Thread-safe: double-checked locking ensures setup() completes before any
    thread can call into the library, even under concurrent first-access.
    """

    def __init__(self, name: str, setup) -> None:
        self._name  = name
        self._setup = setup
        self._lib: 'ctypes.CDLL | None' = None
        self._lock  = threading.Lock()

    def __getattr__(self, attr: str):
        if self._lib is None:
            with self._lock:
                if self._lib is None:          # re-check under lock
                    lib = ctypes.CDLL(str(_find_dll(self._name)))
                    self._setup(lib)           # argtypes/restype set before publish
                    self._lib = lib            # visible to other threads only after setup
        return getattr(self._lib, attr)


def _setup_minimax(lib: ctypes.CDLL) -> None:
    lib.find_best_move.argtypes       = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.find_best_move.restype        = ctypes.c_int
    lib.minimax_score.argtypes        = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.minimax_score.restype         = ctypes.c_float
    lib.score_root_moves.argtypes     = [
        ctypes.POINTER(ctypes.c_bool),   # bool game_board[7][7][2] (98 bytes)
        ctypes.c_int,                    # depth
        ctypes.c_bool,                   # as_blue
        ctypes.POINTER(ctypes.c_float),  # float out_scores[1225]
    ]
    lib.score_root_moves.restype      = None
    lib.find_best_move_timed.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.find_best_move_timed.restype  = ctypes.c_int


def _setup_micro3(lib: ctypes.CDLL) -> None:
    lib.find_best_move.argtypes   = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.find_best_move.restype    = ctypes.c_int
    lib.score_root_moves.argtypes = [
        ctypes.POINTER(ctypes.c_bool),
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.score_root_moves.restype  = None


def _setup_stauf(lib: ctypes.CDLL) -> None:
    lib.find_best_move.restype = ctypes.c_int


def _setup_hmcts(lib: ctypes.CDLL) -> None:
    lib.hmcts_find_best_move.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.hmcts_find_best_move.restype  = ctypes.c_int
    lib.hmcts_init()


def _setup_compy3(lib: ctypes.CDLL) -> None:
    lib.find_best_move_sw.argtypes        = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.find_best_move_sw.restype         = ctypes.c_int
    lib.find_best_move_sw_timed.argtypes  = [ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
    lib.find_best_move_sw_timed.restype   = ctypes.c_int


_minimax_lib = _LazyLib("micro4",               _setup_minimax)
_micro3_lib  = _LazyLib("micro3",               _setup_micro3)
_stauf_lib   = _LazyLib("cell_dll",             _setup_stauf)
_hmcts_lib   = _LazyLib("micro_mcts_heuristic", _setup_hmcts)
_compy3_lib  = _LazyLib("compy3",               _setup_compy3)


def count_cells(board: Board) -> tuple[int, int]:
    return int(numpy.count_nonzero(board[:, :, 1])), int(numpy.count_nonzero(board[:, :, 0]))


def show_board(board: Board) -> None:
    if not _HAS_TERM_IMAGE:
        raise RuntimeError("term_image is not installed; run: pip install term-image")
    img_arr = board.astype(dtype=numpy.uint8)
    img_arr = numpy.dstack((numpy.zeros((7, 7), dtype=numpy.uint8), img_arr))
    img_arr[img_arr == 1] = 255
    img = Image.fromarray(img_arr, 'RGB')
    _AutoImage(img, width=14, height=7).draw(h_align="left", pad_height=-80)


def str_board(board: Board) -> str:
    if not _HAS_TERM_IMAGE:
        raise RuntimeError("term_image is not installed; run: pip install term-image")
    img_arr = board.astype(dtype=numpy.uint8)
    img_arr = numpy.dstack((numpy.zeros((7, 7), dtype=numpy.uint8), img_arr))
    img_arr[img_arr == 1] = 255
    img = Image.fromarray(img_arr, 'RGB')
    return str(_AutoImage(img))


def flip_board(board: Board) -> Board:
    return numpy.rot90(board[:, :, ::-1])


# libataxx halfmove clock (kz04px community rules, adopted 2026-07-18): the
# game is drawn after CLOCK_LIMIT plies without a clone move.  Clones reset
# the clock; jumps and passes tick it.  Replaces threefold repetition, whose
# path-dependence made it invisible to the search and mislabelled 98% of
# "draws" (241/245 repetition games had unequal material, up to 17 pieces).
CLOCK_LIMIT = 100
PASS_ACTION = 1225


def tick_clock(clock: int, action: int) -> int:
    """Advance the halfmove clock for one applied action (or pass)."""
    if action == PASS_ACTION:
        return clock + 1
    mv = action % 25
    is_jump = abs((mv % 5) - 2) == 2 or abs((mv // 5) - 2) == 2
    return clock + 1 if is_jump else 0


def action_to_move(action: int) -> tuple[int, int, int, int, bool]:
    piece = action // 25
    move = action % 25
    from_x = piece % 7
    from_y = piece // 7
    mv_x = (move % 5) - 2
    mv_y = (move // 5) - 2

    to_x = from_x + mv_x
    to_y = from_y + mv_y

    jump = True if abs(mv_x) == 2 or abs(mv_y) == 2 else False

    return from_x, from_y, to_x, to_y, jump


def move_to_action(x: int, y: int, x2: int, y2: int) -> int:
    piece = y * 7 + x
    move = y2 * 5 + x2

    return piece * 25 + move


def calc_reward(board: Board, as_blue: bool = True) -> tuple[float, bool]:
    reward = 0.0
    terminated = False
    new_blue, new_green = count_cells(board)
    as_blue = True

    if as_blue:
        plr = "B:"
        player_cells = new_blue
        opponent_cells = new_green
    else:
        plr = "G:"
        player_cells = new_green
        opponent_cells = new_blue

    end = False
    empty = (49 - new_blue) - new_green
    # Shortcut, assume if we can't move, the other player
    # gets all the remaining cells
    # if *both* are dead, the reward is flat anyway
    if not numpy.any(action_masks(board, as_blue)):
        opponent_cells += empty
        end = True
    if not numpy.any(action_masks(board, not as_blue)):
        player_cells += empty
        end = True

    if end:
        score = player_cells - opponent_cells
        # The game is over at least one player can't move
        if score > 0:
            reward = 1.0
        elif score < 0:
            reward = -1.0
        print(plr, new_blue, new_green, reward)
        return reward, True

    if player_cells == 0:  # We have lost
        reward = -1.0
        terminated = True
    elif opponent_cells == 0:  # We have won!
        reward = 1.0
        terminated = True
    else:
        reward = (pow(player_cells - opponent_cells, 2)) / 100
        if player_cells < opponent_cells:
            reward = -reward
        reward = min(max(reward, -0.8), 0.8)

    return reward, terminated


def action_masks(game_grid: Board, turn: bool) -> npt.NDArray[numpy.bool_]:
    """1225-space of valid turns"""
    # (7,7): True where the current player has a piece
    piece_mask = game_grid[:, :, 1 if turn else 0]

    # (7,7): True where a cell is completely empty
    empty_mask = ~(game_grid[:, :, 0] | game_grid[:, :, 1])

    # empty_mask indexed by precomputed dest coords -> (7,7,5,5)
    dest_empty = empty_mask[_DEST_Y, _DEST_X]

    actions = piece_mask[:, :, numpy.newaxis, numpy.newaxis] & _IN_BOUNDS & dest_empty
    return actions.flatten()


def is_action_valid(board: Board, action: int, as_blue: bool) -> bool:
    if as_blue:
        player_cell = BLUE
    else:
        player_cell = GREEN

    from_x, from_y, to_x, to_y, _ = action_to_move(int(action))
    if numpy.array_equal(board[from_y, from_x], player_cell):
        # We are trying to move our own piece

        if 0 <= to_x < 7 and\
           0 <= to_y < 7:

            if not any(board[to_y, to_x]):
                # The Dest is free
                return True
    return False


def evaluate_position(board: Board, depth: int, as_blue: bool) -> float:
    """Evaluate board from as_blue's perspective; returns value in [-1, +1]."""
    return float(_minimax_lib.minimax_score(board.tobytes(), depth, as_blue))


def find_best_move(board: Any, depth: int, as_blue: bool,
                   engine: str = 'minimax', move_count: int = -1) -> int:
    if engine == 'stauf':
        return _stauf_lib.find_best_move(board, depth, as_blue, move_count)
    if engine == 'micro3':
        return int(_micro3_lib.find_best_move(board, depth, as_blue))
    if engine == 'hmcts':
        # depth repurposed as simulation count for MCTS
        return int(_hmcts_lib.hmcts_find_best_move(board, depth, as_blue))
    if engine == 'micro4t':
        # depth repurposed as millisecond time budget for iterative deepening
        return int(_minimax_lib.find_best_move_timed(board, depth, as_blue))
    if engine == 'compy3':
        return int(_compy3_lib.find_best_move_sw(board, depth, as_blue))
    if engine == 'compy3t':
        # depth repurposed as millisecond time budget
        return int(_compy3_lib.find_best_move_sw_timed(board, depth, as_blue))
    return int(_minimax_lib.find_best_move(board, depth, as_blue))


def find_best_move_timed(board: Any, max_ms: int, as_blue: bool) -> int:
    """Iterative-deepening NegaScout search with a millisecond time budget.

    Uses micro4's bitboard engine with aspiration windows and history heuristic.
    Always returns the best move from the last completed depth.

    Args:
        board:   7×7×2 numpy bool array (or bytes via .tobytes())
        max_ms:  time budget in milliseconds (e.g. 100–1000)
        as_blue: True if the engine plays Blue

    Returns:
        action index in 0..1224, or -1 if no legal moves
    """
    return int(_minimax_lib.find_best_move_timed(board, max_ms, as_blue))


def apply_move(board: Board, action: int, as_blue: bool) -> Board:
    """
    Apply a move to a board copy and return the new state.

    Args:
        board: 7x7x2 numpy bool array
        action: action index in 1225-action space
        as_blue: True if Blue is moving, False if Green

    Returns:
        new_board: 7x7x2 numpy bool array (copy of input with move applied)
    """
    board = board.copy()
    pc = 1 if as_blue else 0   # player channel  (BLUE=[0,1] -> ch1, GREEN=[1,0] -> ch0)
    oc = 1 - pc                # opponent channel

    from_x, from_y, to_x, to_y, jump = action_to_move(action)

    if jump:
        board[from_y, from_x] = False
    board[to_y, to_x, pc] = True
    board[to_y, to_x, oc] = False

    # Capture: flip opponent pieces in the 3x3 neighbourhood of the destination
    # (cells are channel-exclusive, so "opponent present" == opponent channel).
    hood = board[max(0, to_y - 1):to_y + 2, max(0, to_x - 1):to_x + 2]
    hood[:, :, pc] |= hood[:, :, oc]
    hood[:, :, oc] = False

    return board


def encode_action(x: int, y: int, dx: int, dy: int) -> int:
    """
    Convert a piece position and move delta to the 1225-action encoding.

    Args:
        x, y: piece position (0-6)
        dx, dy: move delta (-2 to 2)

    Returns:
        action index in 0-1224 range
    """
    move_idx = (int(dy) + 2) * 5 + (int(dx) + 2)
    return int(y) * 7 * 25 + int(x) * 25 + move_idx


# ---------------------------------------------------------------
# Action encoding
#
# action = piece * 25 + move
#   piece = from_y * 7 + from_x          (source cell, 0-48)
#   move  = (mv_y + 2) * 5 + (mv_x + 2) (delta, 0-24)
#   dest  = to_y * 7 + to_x              (destination cell, 0-48)
#
# ACTION_TO_DEST[a]: destination cell index for action a, or -1 if out-of-bounds
# ---------------------------------------------------------------

def _build_action_dest_map() -> npt.NDArray[numpy.int32]:
    dest_arr = numpy.full(1225, -1, dtype=numpy.int32)
    for action in range(1225):
        piece, move = divmod(action, 25)
        fy, fx = divmod(piece, 7)
        dy = (move // 5) - 2
        dx = (move % 5) - 2
        ty, tx = fy + dy, fx + dx
        if 0 <= ty < 7 and 0 <= tx < 7:
            dest_arr[action] = ty * 7 + tx
    return dest_arr


ACTION_TO_DEST: npt.NDArray[numpy.int32] = _build_action_dest_map()


# ---------------------------------------------------------------
# Board symmetry augmentation (D4 group: 4 rotations × 2 reflections)
# ---------------------------------------------------------------

def _build_symmetry_perms() -> list[npt.NDArray[numpy.int32]]:
    """Precompute action-index permutation tables for all 8 D4 symmetries.

    Each transform is defined by how it maps (y, x, dy, dx) -> (y', x', dy', dx'):
      0: identity          1: rot90 CCW        2: rot180
      3: rot270 CCW        4: flipH (cols)     5: flipV (rows)
      6: flipD1 (main)     7: flipD2 (anti)
    """
    transforms = [
        lambda y, x, dy, dx: (y,     x,     dy,   dx),   # 0 identity
        lambda y, x, dy, dx: (6-x,   y,    -dx,   dy),   # 1 rot90 CCW
        lambda y, x, dy, dx: (6-y,   6-x,  -dy,  -dx),   # 2 rot180
        lambda y, x, dy, dx: (x,     6-y,   dx,  -dy),   # 3 rot270 CCW
        lambda y, x, dy, dx: (y,     6-x,   dy,  -dx),   # 4 flipH
        lambda y, x, dy, dx: (6-y,   x,    -dy,   dx),   # 5 flipV
        lambda y, x, dy, dx: (x,     y,     dx,   dy),   # 6 flipD1 (main diagonal)
        lambda y, x, dy, dx: (6-x,   6-y,  -dx,  -dy),  # 7 flipD2 (anti-diagonal)
    ]
    perms = []
    for tf in transforms:
        perm = numpy.empty(1225, dtype=numpy.int32)
        for action in range(1225):
            piece, move = divmod(action, 25)
            y, x = divmod(piece, 7)
            dy = (move // 5) - 2
            dx = (move % 5) - 2
            ny, nx, ndy, ndx = tf(y, x, dy, dx)
            perm[action] = (ny * 7 + nx) * 25 + (ndy + 2) * 5 + (ndx + 2)
        perms.append(perm)
    return perms


# Forward permutation:  SYMMETRY_PERMS[k][action] = transformed action under symmetry k.
# Inverse permutation:  SYMMETRY_INV_PERMS[k][j] = original action that maps to j under k.
# Use SYMMETRY_INV_PERMS for vectorised policy augmentation: new_policy = old_policy[:, inv_perm]
SYMMETRY_PERMS: list[npt.NDArray[numpy.int32]] = _build_symmetry_perms()
SYMMETRY_INV_PERMS: list[npt.NDArray[numpy.int32]] = [
    numpy.argsort(p).astype(numpy.int32) for p in SYMMETRY_PERMS
]


def _build_symmetry_perms_49() -> list[npt.NDArray[numpy.int32]]:
    """Precompute destination-cell permutation tables for all 8 D4 symmetries.

    Same group as _build_symmetry_perms but operating on 49 destination cell
    indices (y*7+x) only - no delta component.
    """
    transforms_pos = [
        lambda y, x: (y,   x),    # 0 identity
        lambda y, x: (6-x, y),    # 1 rot90 CCW
        lambda y, x: (6-y, 6-x),  # 2 rot180
        lambda y, x: (x,   6-y),  # 3 rot270 CCW
        lambda y, x: (y,   6-x),  # 4 flipH
        lambda y, x: (6-y, x),    # 5 flipV
        lambda y, x: (x,   y),    # 6 flipD1 (main diagonal)
        lambda y, x: (6-x, 6-y),  # 7 flipD2 (anti-diagonal)
    ]
    perms = []
    for tf in transforms_pos:
        perm = numpy.empty(49, dtype=numpy.int32)
        for cell in range(49):
            y, x = divmod(cell, 7)
            ny, nx = tf(y, x)
            perm[cell] = ny * 7 + nx
        perms.append(perm)
    return perms


# 49-dim destination permutation tables (for D4 symmetry augmentation on dest indices).
SYMMETRY_PERMS_49: list[npt.NDArray[numpy.int32]] = _build_symmetry_perms_49()
SYMMETRY_INV_PERMS_49: list[npt.NDArray[numpy.int32]] = [
    numpy.argsort(p).astype(numpy.int32) for p in SYMMETRY_PERMS_49
]


def apply_obs_symmetry(
    obs_batch: npt.NDArray[numpy.float32], k: int,
) -> npt.NDArray[numpy.float32]:
    """Apply the k-th D4 symmetry to a batch of obs arrays of shape (N, 7, 7, 4).

    Returns a view or copy - always call np.ascontiguousarray before torch.from_numpy.
    """
    if k == 0:
        return obs_batch
    if k == 1:
        return numpy.rot90(obs_batch, k=1, axes=(1, 2))  # type: ignore[return-value]
    if k == 2:
        return numpy.rot90(obs_batch, k=2, axes=(1, 2))  # type: ignore[return-value]
    if k == 3:
        return numpy.rot90(obs_batch, k=3, axes=(1, 2))  # type: ignore[return-value]
    if k == 4:
        return obs_batch[:, :, ::-1, :]
    if k == 5:
        return obs_batch[:, ::-1, :, :]
    if k == 6:
        return obs_batch.transpose(0, 2, 1, 3)
    # k == 7: anti-diagonal flip
    return obs_batch[:, ::-1, ::-1, :].transpose(0, 2, 1, 3)


def new_board() -> Board:
    """Create the canonical 4-corner starting position."""
    board = numpy.zeros((7, 7, 2), dtype=numpy.bool_)
    board[0, 0] = BLUE
    board[0, 6] = GREEN
    board[6, 0] = GREEN
    board[6, 6] = BLUE
    return board


def board_to_obs(board: Board, turn: bool, clock: int = 0) -> Obs:
    """
    Convert a 7x7x2 board + turn flag to a 7x7x4 float32 observation.

    Always current-player-relative:
      channel 0 = opponent pieces
      channel 1 = my pieces (current player)
      channel 2 = 1.0 (constant; kept for architecture compatibility)
      channel 3 = halfmove clock / CLOCK_LIMIT (0 for legacy nets, which
                  were trained with this plane all-zero)

    This prevents the value head from shortcutting on the turn indicator
    and forces it to learn from actual piece positions.
    """
    obs = numpy.zeros((7, 7, 4), dtype=numpy.float32)
    if turn:  # Blue's turn: blue=mine(ch1), green=opponent(ch0)
        obs[:, :, 0] = board[:, :, 0].astype(numpy.float32)
        obs[:, :, 1] = board[:, :, 1].astype(numpy.float32)
    else:     # Green's turn: green=mine(ch1), blue=opponent(ch0)
        obs[:, :, 0] = board[:, :, 1].astype(numpy.float32)
        obs[:, :, 1] = board[:, :, 0].astype(numpy.float32)
    obs[:, :, 2] = 1.0
    if clock:
        obs[:, :, 3] = float(clock) / float(CLOCK_LIMIT)
    return obs


def check_terminal(board: Board, turn: bool) -> tuple[bool, Optional[float]]:
    """
    Check if the game is over from the perspective of the player whose turn it is.

    The game ends only when BOTH players have no legal moves (or a side has
    zero pieces). If only the current player is stuck they pass and the other
    continues - that is NOT a terminal state. Empty cells are never allocated;
    the final score is purely blue_count - green_count.

    Returns:
        (is_terminal, value) where value is +1.0 win, -1.0 loss, 0.0 draw, or None if not terminal.
    """
    blue_count, green_count = count_cells(board)

    if blue_count == 0:
        return True, (-1.0 if turn else 1.0)
    if green_count == 0:
        return True, (1.0 if turn else -1.0)

    # Only terminal when BOTH players are stuck simultaneously
    if numpy.any(action_masks(board, turn)) or numpy.any(action_masks(board, not turn)):
        return False, None

    score = (blue_count - green_count) if turn else (green_count - blue_count)
    return True, 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
