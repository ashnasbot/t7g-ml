"""
Core game logic for The Beehive Puzzle (The 11th Hour).

Hexagonal Ataxx variant played on a regular hexagonal board (side length 5):
  - 61 valid cells in axial coordinates (q, r): max(|q|, |r|, |q+r|) <= 4
  - Clone: move to any of 6 adjacent cells (distance-1); source piece stays
  - Jump:  move to any of 12 cells at hex-distance 2; source piece vacates
  - After landing, all 6 adjacent opponent pieces are converted to the mover's colour
  - Two players: Yellow (first) and Red
  - Starting: 3 pieces each at alternating corners of the hexagonal board

Action encoding:
  action = cell_idx * N_DIRS + dir_idx   (0-1097, N_DIRS=18)
  dir_idx 0-5:  clone directions (CLONE_DIRS)
  dir_idx 6-17: jump directions  (JUMP_DIRS)
"""
import numpy
import numpy.typing as npt
from typing import Optional

# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------

RADIUS = 4   # hex board radius; side length = RADIUS + 1 = 5

# All 61 valid cells in axial (q, r) coordinates.
# Constraint: max(|q|, |r|, |q+r|) <= RADIUS
CELLS: list[tuple[int, int]] = [
    (q, r)
    for r in range(-RADIUS, RADIUS + 1)
    for q in range(-RADIUS, RADIUS + 1)
    if abs(q + r) <= RADIUS
]
CELL_TO_IDX: dict[tuple[int, int], int] = {c: i for i, c in enumerate(CELLS)}
N_CELLS: int = len(CELLS)   # 61

# Clone (distance-1) hex neighbours
CLONE_DIRS: list[tuple[int, int]] = [
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
]
# Jump (distance-2) hex cells — exactly the 12 cells reachable in two steps
# that are not reachable in one step
JUMP_DIRS: list[tuple[int, int]] = [
    (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2),
    (1, 1), (-1, -1), (1, -2), (-1, 2), (2, -1), (-2, 1),
]
ALL_DIRS: list[tuple[int, int]] = CLONE_DIRS + JUMP_DIRS
N_DIRS: int = len(ALL_DIRS)   # 18
N_ACTIONS: int = N_CELLS * N_DIRS   # 1098

# DEST_TABLE[ci, di] = destination cell index, or -1 if off-board
def _build_dest_table() -> npt.NDArray[numpy.int16]:
    table = numpy.full((N_CELLS, N_DIRS), -1, dtype=numpy.int16)
    for ci, (q, r) in enumerate(CELLS):
        for di, (dq, dr) in enumerate(ALL_DIRS):
            dest = (q + dq, r + dr)
            if dest in CELL_TO_IDX:
                table[ci, di] = CELL_TO_IDX[dest]
    return table

DEST_TABLE: npt.NDArray[numpy.int16] = _build_dest_table()

# NEIGHBOR_TABLE[ci, k] = index of k-th clone-neighbor of cell ci, or -1
NEIGHBOR_TABLE: npt.NDArray[numpy.int16] = DEST_TABLE[:, :6].copy()

# ---------------------------------------------------------------------------
# Player channels
# ---------------------------------------------------------------------------

YELLOW = 1   # channel index; Yellow moves first (analogous to Blue in T7G)
RED    = 0   # channel index; Red moves second (analogous to Green in T7G)

# Starting positions: 3 pieces each at alternating corners.
# In pointy-top display (screen_row = 2r+q), CW from top:
#   top(0,-4), upper-right(4,-4), lower-right(4,0), bottom(0,4), lower-left(-4,4), upper-left(-4,0)
# Red holds the top / lower-right / lower-left; Yellow the other three.
_CORNERS_YELLOW = [(0, -4), (4,  0), (-4, 4)]   # top, lower-right, lower-left  (ScummVM cells 0,34,56)
_CORNERS_RED    = [(4, -4), (0,  4), (-4, 0)]   # upper-right, bottom, upper-left (ScummVM cells 4,26,60)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Board = npt.NDArray[numpy.bool_]   # shape (N_CELLS, 2)
Obs   = npt.NDArray[numpy.float32] # shape (N_CELLS, 4)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def new_board() -> Board:
    """Starting position: 3 Yellow at alternating corners, 3 Red at the other three."""
    board = numpy.zeros((N_CELLS, 2), dtype=numpy.bool_)
    for q, r in _CORNERS_YELLOW:
        board[CELL_TO_IDX[(q, r)], YELLOW] = True
    for q, r in _CORNERS_RED:
        board[CELL_TO_IDX[(q, r)], RED] = True
    return board


def count_cells(board: Board) -> tuple[int, int]:
    """Returns (yellow_count, red_count)."""
    return int(numpy.count_nonzero(board[:, YELLOW])), int(numpy.count_nonzero(board[:, RED]))


def action_masks(board: Board, yellow_turn: bool) -> npt.NDArray[numpy.bool_]:
    """N_ACTIONS-length bool array; True for each currently legal action."""
    my_ch  = YELLOW if yellow_turn else RED
    opp_ch = RED    if yellow_turn else YELLOW

    my_cells = board[:, my_ch]                         # (N_CELLS,) bool
    opp_cells = board[:, opp_ch]
    empty    = ~(board[:, 0] | board[:, 1])            # (N_CELLS,) bool

    dest_valid = DEST_TABLE >= 0                        # (N_CELLS, N_DIRS)
    dest_idx   = numpy.maximum(DEST_TABLE, 0)           # safe indices (clip -1 → 0)
    dest_empty = empty[dest_idx]                        # (N_CELLS, N_DIRS)

    mask = my_cells[:, numpy.newaxis] & dest_valid & dest_empty  # (N_CELLS, N_DIRS)

    # Jump restriction (di >= 6): dest must have an opponent neighbour
    # and must NOT have a own-piece neighbour (clone would dominate).
    # Precompute per-cell: has_opp_neighbour and has_own_neighbour.
    has_opp_adj = numpy.zeros(N_CELLS, dtype=numpy.bool_)
    has_own_adj = numpy.zeros(N_CELLS, dtype=numpy.bool_)
    clone_dest_idx = numpy.maximum(DEST_TABLE[:, :6], 0)      # (N_CELLS, 6)
    clone_dest_valid = DEST_TABLE[:, :6] >= 0
    # For each cell d: does d have an opp/own neighbour via the clone table?
    # We accumulate: any source s with DEST_TABLE[s, di]==d means d and s are adjacent.
    # Easier: for each dest cell d, its adjacency is exactly DEST_TABLE rows that include d.
    # Use the precomputed neighbour table implicitly via broadcast.
    for di in range(6):
        nbrs = DEST_TABLE[:, di]          # nbrs[ci] = clone-neighbour di of ci (or -1)
        valid = nbrs >= 0
        # nbrs[ci] = d means d is adjacent to ci
        # → cell d has opp-adj if opp_cells[ci] and nbrs[ci]==d
        # Accumulate: has_opp_adj[d] |= opp_cells[ci] for each ci where nbrs[ci]==d
        numpy.add.at(has_opp_adj, nbrs[valid], opp_cells[valid])
        numpy.add.at(has_own_adj, nbrs[valid], my_cells[valid])
    has_opp_adj = has_opp_adj.astype(bool)
    has_own_adj = has_own_adj.astype(bool)

    jump_mask = mask[:, 6:]  # (N_CELLS, 12)
    jump_dest_idx  = dest_idx[:, 6:]
    jump_opp_ok    = has_opp_adj[jump_dest_idx]   # dest has opp neighbour
    jump_own_block = has_own_adj[jump_dest_idx]   # dest has own neighbour → blocked
    mask[:, 6:] = jump_mask & jump_opp_ok & ~jump_own_block

    # Source-side jump condition (ScummVM sub11): a jump from source ci in
    # direction j is only valid if source is isolated (no own-colour neighbours)
    # OR every opponent neighbour of source lies in the jump direction family.
    # JUMP_ALLOWED_BITS[j] is a 6-bit mask of clone-dirs "facing" jump dir j:
    #   j 0-5 (straight 2× clone): single bit = the same clone dir
    #   j 6-11 (diagonal): two clone-dir bits (the pair that combine to reach it)
    #   j0=(2,0)→[0]  j1=(-2,0)→[1]  j2=(0,2)→[2]  j3=(0,-2)→[3]
    #   j4=(2,-2)→[4] j5=(-2,2)→[5]
    #   j6=(1,1)→[0,2]  j7=(-1,-1)→[1,3]  j8=(1,-2)→[3,4]
    #   j9=(-1,2)→[2,5] j10=(2,-1)→[0,4]  j11=(-2,1)→[1,5]
    _JUMP_ALLOWED = numpy.array(
        [1, 2, 4, 8, 16, 32, 5, 10, 24, 36, 17, 34], dtype=numpy.uint8
    )  # shape (12,)

    # Compute 6-bit own/opp neighbour bitmasks for each source cell
    source_own_bits = numpy.zeros(N_CELLS, dtype=numpy.uint8)
    source_opp_bits = numpy.zeros(N_CELLS, dtype=numpy.uint8)
    for cd in range(6):
        nbr = DEST_TABLE[:, cd]
        valid_cd = nbr >= 0
        safe_nbr = numpy.where(valid_cd, nbr, 0)
        bit = numpy.uint8(1 << cd)
        source_own_bits |= numpy.where(valid_cd & my_cells[safe_nbr],  bit, numpy.uint8(0))
        source_opp_bits |= numpy.where(valid_cd & opp_cells[safe_nbr], bit, numpy.uint8(0))

    source_isolated = (source_own_bits == 0)                          # (N_CELLS,)
    not_allowed = (~_JUMP_ALLOWED).astype(numpy.uint8)                # (12,)
    opp_outside = (source_opp_bits[:, None] & not_allowed[None, :])   # (N_CELLS, 12)
    source_ok = source_isolated[:, None] | (opp_outside == 0)         # (N_CELLS, 12)
    mask[:, 6:] &= source_ok

    return mask.flatten()


def is_action_valid(board: Board, action: int, yellow_turn: bool) -> bool:
    masks = action_masks(board, yellow_turn)
    return bool(masks[action])


def apply_move(board: Board, action: int, yellow_turn: bool) -> Board:
    """Apply a move (clone or jump) and return the resulting board copy."""
    board = board.copy()
    my_ch  = YELLOW if yellow_turn else RED
    opp_ch = RED    if yellow_turn else YELLOW

    ci = action // N_DIRS
    di = action % N_DIRS
    dest = int(DEST_TABLE[ci, di])

    if di >= 6:   # jump: source vacates
        board[ci, my_ch] = False

    board[dest, my_ch] = True

    # Capture: flip adjacent opponent pieces (vectorized)
    nbrs  = NEIGHBOR_TABLE[dest]          # shape (6,) int16, -1 = off-board
    valid = nbrs >= 0
    safe  = numpy.where(valid, nbrs, 0)   # clip -1 so indexing is safe
    hits  = valid & board[safe, opp_ch]
    board[safe[hits], opp_ch] = False
    board[safe[hits], my_ch]  = True

    return board


def legal_actions_simple(board: Board, yellow_turn: bool) -> npt.NDArray[numpy.intp]:
    """All legal actions using game-rule move legality (not sub11 AI restrictions).

    Faster than action_masks: just my_cell → empty dest via any of the 18 directions.
    Use this for rollout move selection; use action_masks for tree expansion.
    """
    my_ch    = YELLOW if yellow_turn else RED
    my_cells = board[:, my_ch]
    empty    = ~(board[:, 0] | board[:, 1])
    dest_valid = DEST_TABLE >= 0
    dest_idx   = numpy.maximum(DEST_TABLE, 0)
    dest_empty = empty[dest_idx]
    mask = my_cells[:, numpy.newaxis] & dest_valid & dest_empty   # (N_CELLS, N_DIRS)
    return numpy.flatnonzero(mask)


def can_move_simple(board: Board, yellow_turn: bool) -> bool:
    """Return True if the current player has at least one reachable empty cell.

    This matches ScummVM's sub02 / selectSourceHexagon logic: any empty cell
    within hex-distance 1 (clone) or 2 (jump) counts as a valid move.  The
    AI's additional restrictions (destination must have an opponent neighbour,
    source-isolation condition) are search pruning only, not game rules.
    """
    my_ch   = YELLOW if yellow_turn else RED
    my_cells = board[:, my_ch]                       # (N_CELLS,) bool
    empty    = ~(board[:, 0] | board[:, 1])
    dest_idx = numpy.maximum(DEST_TABLE, 0)          # (N_CELLS, N_DIRS), -1 clipped to 0
    dest_valid = DEST_TABLE >= 0
    dest_empty = empty[dest_idx]                     # (N_CELLS, N_DIRS)
    return bool(numpy.any(my_cells[:, None] & dest_valid & dest_empty))


def check_terminal(board: Board, yellow_turn: bool) -> tuple[bool, Optional[float]]:
    """
    Check if the game is over from the perspective of the player whose turn it is.

    The game ends when the current player has no legal moves OR a side has 0 pieces.
    Per the original 11th Hour rule: when a player is stuck, the opponent claims all
    remaining empty cells, then the side with more pieces wins.

    Move legality uses the actual game rule (sub02 semantics): any empty cell
    within hex-distance 1 or 2 is reachable.  The AI's sub11 restrictions are
    search optimisations, not game rules.

    Returns (is_terminal, value) where value is +1.0 win / -1.0 loss / 0.0 draw
    from the perspective of the yellow_turn player, or None if not terminal.
    """
    yc, rc = count_cells(board)

    if yc == 0:
        return True, (-1.0 if yellow_turn else 1.0)
    if rc == 0:
        return True, (1.0 if yellow_turn else -1.0)

    if can_move_simple(board, yellow_turn):
        return False, None   # current player has moves — game continues

    # Current player is stuck: opponent claims all empty cells, then count.
    empty = int(N_CELLS - yc - rc)
    if yellow_turn:
        # Yellow stuck → Red gets empties
        yc_f, rc_f = yc, rc + empty
    else:
        # Red stuck → Yellow gets empties
        yc_f, rc_f = yc + empty, rc

    score = yc_f - rc_f
    raw = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
    return True, (raw if yellow_turn else -raw)


def board_to_obs(board: Board, yellow_turn: bool) -> Obs:
    """
    Convert board + turn to a (N_CELLS, 4) float32 observation.

    Always current-player-relative so the value head learns from positions, not identities:
      channel 0 = opponent pieces
      channel 1 = my pieces
      channel 2 = 1.0 (constant; kept for architecture compatibility with T7G)
      channel 3 = 0.0 (unused)
    """
    obs = numpy.zeros((N_CELLS, 4), dtype=numpy.float32)
    if yellow_turn:
        obs[:, 0] = board[:, RED].astype(numpy.float32)
        obs[:, 1] = board[:, YELLOW].astype(numpy.float32)
    else:
        obs[:, 0] = board[:, YELLOW].astype(numpy.float32)
        obs[:, 1] = board[:, RED].astype(numpy.float32)
    obs[:, 2] = 1.0
    return obs


def draw_board(board: Board) -> None:
    """Print the board in pointy-top hex orientation.

    Mapping: screen_row = 2*r + q + 2*RADIUS  (range 0..4*RADIUS)
             screen_col = (q + RADIUS) * 2     (range 0..4*RADIUS)

    Adjacent columns are staggered by one screen row, giving the classic
    pointy-top hex look.  R = Red, Y = Yellow, . = empty.
    """
    cell_char: dict[tuple[int, int], str] = {}
    for i, (q, r) in enumerate(CELLS):
        if board[i, RED]:
            cell_char[(q, r)] = 'R'
        elif board[i, YELLOW]:
            cell_char[(q, r)] = 'Y'
        else:
            cell_char[(q, r)] = '.'

    # screen_row = 2*r + q + 2*RADIUS  →  range [0, 4*RADIUS]
    # screen_col = (q + RADIUS) * 2   →  range [0, 4*RADIUS]
    n = 4 * RADIUS   # = 16; grid is (n+1) × (n+1) = 17×17
    row_off = 2 * RADIUS   # = 8
    grid = [[' '] * (n + 1) for _ in range(n + 1)]

    for q, r in CELLS:
        sr = 2 * r + q + row_off
        sc = (q + RADIUS) * 2
        grid[sr][sc] = cell_char[(q, r)]

    for row in grid:
        line = ''.join(row)
        if line.strip():
            print(line)


# ---------------------------------------------------------------------------
# Action/cell utilities
# ---------------------------------------------------------------------------

def action_to_move(action: int) -> tuple[int, int, bool]:
    """Decode action into (from_cell_idx, to_cell_idx, is_jump)."""
    ci = action // N_DIRS
    di = action % N_DIRS
    dest = int(DEST_TABLE[ci, di])
    return ci, dest, di >= 6


def encode_action(ci: int, di: int) -> int:
    """Encode (cell_index, direction_index) into a flat action integer."""
    return ci * N_DIRS + di
