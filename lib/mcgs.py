"""
Gumbel AlphaZero MCGS backed by micro_mcts.c (C extension via ctypes).

The C extension manages the search tree and transposition table in C heap
memory (malloc/free), eliminating the Python arena allocator leak that
accumulates ~8 MB per game when the Python-object MCGS graph is used.

Network inference still happens in Python; the step-wise interface lets the
game pool batch leaf evaluations across all concurrent games into one GPU pass.

Public surface (same as the old pure-Python version):
  MCGS          — search driver; owns one C MCGSInstance (one TT)
  MCGSSearch    — step-wise search handle returned by MCGS.start_search()
  MCGSNode      — kept for import compatibility (type-hint only)
"""
from __future__ import annotations

import ctypes
import pathlib
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import (
    Board,
    count_cells, ACTION_TO_DEST,
)


# ── Search heuristics ────────────────────────────────────────────────────────

def _heuristic_value(board: Board, turn: bool) -> float:
    """
    Raw cell-count value from the current player's perspective, normalised to
    [-1, 1].  Pure numpy — no C minimax call, negligible overhead per leaf.
    """
    blue, green = count_cells(board)
    total = blue + green
    if total == 0:
        return 0.0
    return (blue - green) / total if turn else (green - blue) / total


def _capture_heuristic_49(board: Board, turn: bool) -> npt.NDArray[np.float32]:
    """
    49-dim destination prior: captures (adjacent opponent pieces) plus a clone
    reachability bonus.  Cloning adds a piece unconditionally so clone-reachable
    destinations are preferred over jump-only destinations even with zero captures.
    A small floor (1e-2) keeps all destinations in the distribution.
    """
    own = board[:, :, 1 if turn else 0].astype(np.float32)
    opp = board[:, :, 0 if turn else 1].astype(np.float32)
    kernel = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if not (dy == 0 and dx == 0)]
    own_pad = np.pad(own, 1, mode='constant')
    opp_pad = np.pad(opp, 1, mode='constant')
    own_neighbors = np.stack(
        [own_pad[1+dy:8+dy, 1+dx:8+dx] for dy, dx in kernel], axis=0).sum(axis=0)
    opp_neighbors = np.stack(
        [opp_pad[1+dy:8+dy, 1+dx:8+dx] for dy, dx in kernel], axis=0).sum(axis=0)
    h = opp_neighbors + 0.3 * (own_neighbors > 0).astype(np.float32) + 1e-2
    return (h / h.sum()).flatten().astype(np.float32)


def _capture_heuristic_1225(board: Board, turn: bool) -> npt.NDArray[np.float32]:
    """1225-dim version: broadcast the 49-dim capture heuristic via ACTION_TO_DEST."""
    h49 = _capture_heuristic_49(board, turn)
    h = np.zeros(1225, dtype=np.float32)
    valid = ACTION_TO_DEST >= 0
    h[valid] = h49[ACTION_TO_DEST[valid]]
    total = h.sum()
    return h / total if total > 0 else h


# ── DLL loading ─────────────────────────────────────────────────────────────

def _find_mcts_dll() -> pathlib.Path:
    candidates = [
        pathlib.Path(__file__).parent / "micro_mcts.dll",       # installed wheel
        pathlib.Path().absolute() / "lib" / "micro_mcts.dll",   # dev: standard build
        pathlib.Path().absolute() / "micro_mcts.dll",           # dev: native build
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot locate micro_mcts.dll; build with: make dll\n"
        f"Searched: {candidates}"
    )


_lib = ctypes.CDLL(str(_find_mcts_dll()))

# mcgs_init / create / clear / destroy
_lib.mcgs_init.argtypes = []
_lib.mcgs_init.restype  = None

_lib.mcgs_create.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int]
_lib.mcgs_create.restype  = ctypes.c_void_p

_lib.mcgs_clear.argtypes = [ctypes.c_void_p]
_lib.mcgs_clear.restype  = None

_lib.mcgs_destroy.argtypes = [ctypes.c_void_p]
_lib.mcgs_destroy.restype  = None

_lib.mcgs_tt_size.argtypes = [ctypes.c_void_p]
_lib.mcgs_tt_size.restype  = ctypes.c_int

# search lifecycle
_lib.mcgs_start_search.argtypes = [
    ctypes.c_void_p,              # MCGSInstance*
    ctypes.POINTER(ctypes.c_bool),  # bool py_board[7][7][2]  (flat 98 bytes)
    ctypes.c_bool,                # bool turn
]
_lib.mcgs_start_search.restype = ctypes.c_void_p  # MCGSSearchState*

_lib.mcgs_search_destroy.argtypes = [ctypes.c_void_p]
_lib.mcgs_search_destroy.restype  = None

_lib.mcgs_pending_count.argtypes = [ctypes.c_void_p]
_lib.mcgs_pending_count.restype  = ctypes.c_int

_lib.mcgs_is_done.argtypes = [ctypes.c_void_p]
_lib.mcgs_is_done.restype  = ctypes.c_int

_lib.mcgs_get_leaf_board.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_bool),
]
_lib.mcgs_get_leaf_board.restype = None

_lib.mcgs_get_leaf_turn.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.mcgs_get_leaf_turn.restype  = ctypes.c_bool

_lib.mcgs_commit_expansion.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),  # float policy[1225]
    ctypes.c_float,                  # float value
]
_lib.mcgs_commit_expansion.restype = None

_lib.mcgs_step.argtypes = [ctypes.c_void_p]
_lib.mcgs_step.restype  = ctypes.c_int

_lib.mcgs_get_result.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.mcgs_get_result.restype  = None

_lib.mcgs_get_pending_boards.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_bool),   # bool boards_out[n * 98]
    ctypes.POINTER(ctypes.c_bool),   # bool turns_out[n]
]
_lib.mcgs_get_pending_boards.restype = ctypes.c_int

_lib.mcgs_commit_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # float policies_flat[n * 1225]
    ctypes.POINTER(ctypes.c_float),  # float values[n]
    ctypes.c_int,
]
_lib.mcgs_commit_batch.restype = None

_lib.mcgs_init()


# ── Pending leaf proxy ───────────────────────────────────────────────────────

class _PendingLeaf:
    """
    Thin proxy for one C pending leaf node.

    Carries the board/turn that _expand_batch needs for inference, plus
    the search-state pointer and index so commit_expansion can be routed back.
    """
    __slots__ = ['board', 'turn', 'is_terminal', 'is_expanded',
                 '_search_ptr', '_idx']

    def __init__(self, board: npt.NDArray[np.bool_], turn: bool,
                 search_ptr: int, idx: int) -> None:
        self.board       = board
        self.turn        = turn
        self.is_terminal = False
        self.is_expanded = False
        self._search_ptr = search_ptr  # ctypes void* as int
        self._idx        = idx


# ── MCGSNode — kept for import / type-hint compatibility ────────────────────

class MCGSNode:
    """Stub kept for import compatibility.  Not used by the C backend."""
    pass


# ── MCGSSearch — step-wise search handle ────────────────────────────────────

class MCGSSearch:
    """
    Step-wise search handle backed by a C MCGSSearchState.

    Lifecycle (same external interface as the old Python MCGSSearch):
      1. Read pending_leaves — build leaf proxies for uncommitted nodes.
      2. Call mcgs._expand_batch(pending_leaves) to run inference & commit.
      3. Call step() to backprop and advance to the next set of leaves.
      4. Repeat until done is True, then read result.
    """

    __slots__ = ['_ptr']

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def __del__(self) -> None:
        if self._ptr:
            _lib.mcgs_search_destroy(self._ptr)
            self._ptr = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return bool(_lib.mcgs_is_done(self._ptr))

    @property
    def pending_leaves(self) -> list[_PendingLeaf]:
        """Build fresh leaf proxies for nodes currently awaiting expansion."""
        n = _lib.mcgs_pending_count(self._ptr)
        leaves: list[_PendingLeaf] = []
        for i in range(n):
            board = np.zeros((7, 7, 2), dtype=np.bool_)
            _lib.mcgs_get_leaf_board(
                self._ptr, i,
                board.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            )
            turn = bool(_lib.mcgs_get_leaf_turn(self._ptr, i))
            leaves.append(_PendingLeaf(board, turn, self._ptr, i))
        return leaves

    @property
    def result(self) -> npt.NDArray[np.float32]:
        out = np.zeros(1225, dtype=np.float32)
        _lib.mcgs_get_result(
            self._ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def step(self) -> None:
        _lib.mcgs_step(self._ptr)


# ── MCGS — search driver ─────────────────────────────────────────────────────

class _TTProxy:
    """Fake mapping whose len() returns the C TT node count for monitoring."""
    __slots__ = ['_ptr']

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def __len__(self) -> int:
        return _lib.mcgs_tt_size(self._ptr)


class MCGS:
    """
    Gumbel AlphaZero MCGS backed by micro_mcts.c.

    One instance owns one C MCGSInstance (one transposition table).
    Network inference is performed in Python; the C side manages the tree.
    """

    def __init__(
        self,
        network: nn.Module,
        num_simulations: int = 100,
        c_puct: float = 0.75,
        gumbel_k: int = 8,
    ) -> None:
        self.network         = network
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.gumbel_k        = gumbel_k
        self._device         = next(network.parameters()).device
        self._ptr            = _lib.mcgs_create(
            num_simulations, ctypes.c_float(c_puct), gumbel_k,
        )

    def __del__(self) -> None:
        ptr = getattr(self, '_ptr', None)
        if ptr:
            _lib.mcgs_destroy(ptr)
            self._ptr = 0

    # ------------------------------------------------------------------
    # Public API (same interface as old Python MCGS)
    # ------------------------------------------------------------------

    @property
    def root(self) -> None:
        """No-op property kept for compatibility (play_net_vs_net_game sets it to None)."""
        return None

    @root.setter
    def root(self, _value: object) -> None:
        pass  # no-op in C version — TT serves the same purpose

    @property
    def transposition_table(self) -> _TTProxy:
        """Monitoring proxy: len() returns C TT node count."""
        return _TTProxy(self._ptr)

    def clear(self) -> None:
        """Free all C nodes and reset the transposition table."""
        _lib.mcgs_clear(self._ptr)

    def advance_tree(self, action: int) -> None:  # noqa: ARG002
        """No-op: the TT persists across moves so Q-values are reused automatically."""

    def start_search(
        self,
        board: Board,
        turn: bool,
        hint_root: object = None,  # ignored in C version
        move_count: int | None = None,
    ) -> MCGSSearch:
        """Start a step-wise search. hint_root is accepted but not used."""
        import warnings
        board_c = np.ascontiguousarray(board, dtype=np.bool_)
        ptr = _lib.mcgs_start_search(
            self._ptr,
            board_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            ctypes.c_bool(turn),
        )
        if not ptr:
            tt_size = _lib.mcgs_tt_size(self._ptr)
            move_info = f" at move {move_count}" if move_count is not None else ""
            warnings.warn(
                f"mcgs_start_search returned NULL (node slab full, {tt_size} nodes){move_info}; "
                "clearing transposition table and retrying.",
                RuntimeWarning, stacklevel=2,
            )
            _lib.mcgs_clear(self._ptr)
            ptr = _lib.mcgs_start_search(
                self._ptr,
                board_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                ctypes.c_bool(turn),
            )
            if not ptr:
                warnings.warn(
                    "mcgs_start_search failed again after clear — "
                    "returning null search; game will recover via uniform-policy fallback.",
                    RuntimeWarning, stacklevel=2,
                )
        return MCGSSearch(ptr)

    def search(self, board: Board, turn: bool) -> npt.NDArray[np.float32]:
        """
        Run Gumbel MCGS from position (single-game, non-batched).

        Returns improved action probability distribution (1225 floats).
        """
        ss = self.start_search(board, turn)
        while not ss.done:
            self._expand_batch([ss])
            ss.step()
        return ss.result

    def _expand_batch(self, searches: 'list[MCGSSearch]') -> None:
        """
        Evaluate all pending leaves across the given searches in one GPU pass.

        Uses mcgs_get_pending_boards / mcgs_commit_batch — one C call per search
        slot instead of one per leaf — eliminating the per-leaf ctypes overhead
        that was stalling the CPU between GPU batches.
        """
        seg_ptrs: list[int] = []
        seg_counts: list[int] = []
        board_segs: list[npt.NDArray] = []
        turn_segs:  list[npt.NDArray] = []

        for ss in searches:
            n = _lib.mcgs_pending_count(ss._ptr)
            if n == 0:
                continue
            boards = np.empty((n, 7, 7, 2), dtype=np.bool_)
            turns  = np.empty(n, dtype=np.bool_)
            _lib.mcgs_get_pending_boards(
                ss._ptr,
                boards.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                turns.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            )
            board_segs.append(boards)
            turn_segs.append(turns)
            seg_ptrs.append(ss._ptr)
            seg_counts.append(n)

        if not board_segs:
            return

        boards_all = np.concatenate(board_segs)   # (N, 7, 7, 2)
        turns_all  = np.concatenate(turn_segs)    # (N,)
        N = len(turns_all)

        t = turns_all[:, None, None]
        obs_batch = np.zeros((N, 7, 7, 4), dtype=np.float32)
        obs_batch[:, :, :, 0] = np.where(t, boards_all[:, :, :, 0], boards_all[:, :, :, 1])
        obs_batch[:, :, :, 1] = np.where(t, boards_all[:, :, :, 1], boards_all[:, :, :, 0])
        obs_batch[:, :, :, 2] = 1.0

        obs_tensor = torch.from_numpy(obs_batch).to(self._device)
        with torch.no_grad():
            policy_logits, values = self.network(obs_tensor)
            policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()  # (N, 1225)
            values_np    = values.cpu().numpy().flatten()                   # (N,)

        # One C call per search slot instead of one per leaf
        offset = 0
        for ptr, n in zip(seg_ptrs, seg_counts):
            _lib.mcgs_commit_batch(
                ptr,
                policy_probs[offset:offset + n].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                values_np[offset:offset + n].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n),
            )
            offset += n

    def select_action(
        self,
        action_probs: npt.NDArray[np.float32],
        board: Board | None = None,  # noqa: ARG002
        turn: bool | None = None,    # noqa: ARG002
        temperature: float = 1.0,
    ) -> int:
        """Select action from MCGS probability distribution (board/turn ignored)."""
        if temperature == 0:
            return int(np.argmax(action_probs))
        probs = action_probs ** (1.0 / temperature)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            return int(np.argmax(action_probs))
        return int(np.random.choice(len(probs), p=probs))
