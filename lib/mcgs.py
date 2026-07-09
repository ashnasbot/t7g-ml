"""
Gumbel AlphaZero MCGS backed by micro_mcts.c (C extension via ctypes).

The C extension manages the search tree and transposition table in C heap
memory (malloc/free), eliminating the Python arena allocator leak that
accumulates ~8 MB per game when the Python-object MCGS graph is used.

Network inference still happens in Python; the step-wise interface lets the
game pool batch leaf evaluations across all concurrent games into one GPU pass.

Public surface:
  MCGS          - search driver; owns one C MCGSInstance (one TT)
  MCGSSearch    - step-wise search handle returned by MCGS.start_search()
  MCGSNode      - kept for import compatibility (type-hint only)
"""
from __future__ import annotations

import ctypes
import pathlib
import sys
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import (
    Board,
    count_cells, ACTION_TO_DEST,
)


#  Search heuristics 

def _heuristic_value(board: Board, turn: bool) -> float:
    """
    Raw cell-count value from the current player's perspective, normalised to
    [-1, 1].  Pure numpy - no C minimax call, negligible overhead per leaf.
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


#  DLL loading 

def _find_mcts_dll() -> pathlib.Path:
    suffixes = [".dll"] if sys.platform == "win32" else [".so"]
    roots = [
        pathlib.Path().absolute(),                # dev: project root build
        pathlib.Path().absolute() / "lib",        # dev: lib/ build
        pathlib.Path(__file__).parent,            # installed wheel
    ]
    for root in roots:
        for suffix in suffixes:
            p = root / ("micro_mcts" + suffix)
            if p.exists():
                return p
    candidates = [r / ("micro_mcts" + s) for r in roots for s in suffixes]
    raise FileNotFoundError(
        "Cannot locate micro_mcts.dll/.so; build with: make dll\n"
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

_lib.mcgs_get_root_value.argtypes = [ctypes.c_void_p]
_lib.mcgs_get_root_value.restype  = ctypes.c_float

_lib.mcgs_apply_root_dirichlet.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
_lib.mcgs_apply_root_dirichlet.restype  = None

_lib.mcgs_get_pending_boards.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,                 # bool boards_out[n * 98] — pass as void* to skip ctypes.cast
    ctypes.c_void_p,                 # bool turns_out[n]       — skip ctypes.cast overhead
]
_lib.mcgs_get_pending_boards.restype = ctypes.c_int

_lib.mcgs_commit_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,                 # float policies_flat[n * 1225] — void* to skip ctypes.cast
    ctypes.c_void_p,                 # float values[n]               — skip ctypes.cast overhead
    ctypes.c_int,
]
_lib.mcgs_commit_batch.restype = None

_lib.mcgs_init()


#  Pending leaf proxy 

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


#  MCGSNode - kept for import / type-hint compatibility 

class MCGSNode:
    """Stub kept for import compatibility.  Not used by the C backend."""
    pass


#  MCGSSearch - step-wise search handle 

class MCGSSearch:
    """
    Step-wise search handle backed by a C MCGSSearchState.

    Lifecycle:
      1. Read pending_leaves - build leaf proxies for uncommitted nodes.
      2. Call mcgs._expand_batch(pending_leaves) to run inference & commit.
      3. Call step() to backprop and advance to the next set of leaves.
      4. Repeat until done is True, then read result.
    """

    __slots__ = ['_ptr', '_root_expanded']

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr
        self._root_expanded = False

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

    @property
    def root_value(self) -> float:
        """Root Q-value from the mover's perspective (value_sum / visit_count)."""
        return float(_lib.mcgs_get_root_value(self._ptr))

    def step(self) -> None:
        _lib.mcgs_step(self._ptr)


#  MCGS - search driver 

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
        dirichlet_alpha: float = 0.0,
        dirichlet_eps: float = 0.0,
    ) -> None:
        self.network         = network
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.gumbel_k        = gumbel_k
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps
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
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> None:
        """No-op property kept for compatibility (play_net_vs_net_game sets it to None)."""
        return None

    @root.setter
    def root(self, _value: object) -> None:
        pass  # no-op in C version - TT serves the same purpose

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
                    "mcgs_start_search failed again after clear - "
                    "returning null search; game will recover via uniform-policy fallback.",
                    RuntimeWarning, stacklevel=2,
                )
        return MCGSSearch(ptr)

    def search(self, board: Board, turn: bool) -> npt.NDArray[np.float32]:
        """
        Run Gumbel MCGS from position (single-game, non-batched).

        Returns improved action probability distribution (1225 floats).
        After returning, self.last_root_value holds the root Q (mover's perspective).
        """
        ss = self.start_search(board, turn)
        while not ss.done:
            self._expand_batch([ss])
            ss.step()
        self.last_root_value: float = ss.root_value
        return ss.result

    def _launch_forward(self, searches: 'list[MCGSSearch]'):
        """
        Gather pending leaves from `searches` and dispatch a GPU forward pass
        asynchronously.  Returns an opaque handle that `_collect_and_commit`
        consumes; returns `None` if nothing was pending across the group.

        This is the "launch" half of the double-buffered pool inference loop:
        H2D copy, forward, softmax, and D2H copy are all queued on the CUDA
        stream, and a CUDA event is recorded after the D2H.  The Python
        thread returns *without* syncing, so the caller can do CPU-side
        work on a different group of searches while the GPU is busy here.
        """
        seg_searches: list[MCGSSearch] = []
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
                boards.ctypes.data,
                turns.ctypes.data,
            )
            board_segs.append(boards)
            turn_segs.append(turns)
            seg_searches.append(ss)
            seg_ptrs.append(ss._ptr)
            seg_counts.append(n)

        if not board_segs:
            return None

        boards_all = np.concatenate(board_segs)   # (N, 7, 7, 2)
        turns_all  = np.concatenate(turn_segs)    # (N,)
        N = len(turns_all)

        # ROCm/gfx1151: MIOpen re-searches conv kernels for every distinct
        # batch shape (no immediate-mode find-db for this arch), which
        # collapses self-play throughput ~17x as the pending-leaf count
        # varies from step to step. Pad the batch up to the next power-of-two
        # bucket so at most ~log2(max) shapes are ever compiled; the padded
        # rows are sliced off again below. No-op on CUDA, where cuDNN/compile
        # handle dynamic shapes without a per-shape penalty.
        N_pad = (1 << (N - 1).bit_length()) if torch.version.hip else N

        t = turns_all[:, None, None]
        obs_batch = np.zeros((N_pad, 7, 7, 4), dtype=np.float32)
        obs_batch[:N, :, :, 0] = np.where(t, boards_all[:, :, :, 0], boards_all[:, :, :, 1])
        obs_batch[:N, :, :, 1] = np.where(t, boards_all[:, :, :, 1], boards_all[:, :, :, 0])
        obs_batch[:N, :, :, 2] = 1.0

        # CUDA fast path: pinned H2D + BF16 autocast forward + async D2H
        # into pinned host buffers, with a CUDA event to signal D2H done.
        # See _collect_and_commit for the matching sync + commit half.
        # BF16 keeps FP32 exponent range; the ~1% mantissa precision loss
        # is invisible to MCTS visit statistics.  Softmax is done in FP32
        # after upcasting to keep the policy distribution numerically clean.
        if self._device.type == 'cuda':
            obs_tensor = (
                torch.from_numpy(obs_batch).pin_memory()
                .to(self._device, non_blocking=True)
            )
            # BF16 on NVIDIA; FP16 on ROCm/gfx1151. On the Strix Halo iGPU
            # MIOpen has no usable bf16 conv kernels (missing find-db +
            # CK grouped-conv lib) so bf16 autocast falls back ~35x slower
            # than fp16. fp16 is fastest there and its narrower exponent
            # range is fine since softmax runs in fp32 just below.
            _ac_dtype = torch.float16 if torch.version.hip else torch.bfloat16
            with torch.no_grad(), torch.autocast(
                device_type='cuda', dtype=_ac_dtype,
            ):
                policy_logits, values, _ = self.network(obs_tensor)
            # Drop the padded rows (see N_pad above) before softmax / D2H so
            # the committed slabs are exactly the N real pending leaves.
            policy_probs_gpu = F.softmax(policy_logits[:N].float(), dim=-1)
            values_gpu       = values[:N].float()
            # Pinned CPU buffers + non_blocking copy gives a true async D2H.
            policy_cpu = torch.empty(
                policy_probs_gpu.shape, dtype=torch.float32, pin_memory=True,
            )
            values_cpu = torch.empty(
                values_gpu.shape, dtype=torch.float32, pin_memory=True,
            )
            policy_cpu.copy_(policy_probs_gpu, non_blocking=True)
            values_cpu.copy_(values_gpu, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            obs_tensor = torch.from_numpy(obs_batch).to(self._device)
            with torch.no_grad():
                policy_logits, values, _ = self.network(obs_tensor)
                policy_cpu = F.softmax(policy_logits, dim=-1).cpu()
                values_cpu = values.float().cpu()
            event = None

        return (policy_cpu, values_cpu, event, seg_searches, seg_ptrs, seg_counts)

    def _collect_and_commit(self, handle) -> None:
        """
        Sync the CUDA event produced by `_launch_forward`, then commit each
        slot's (policy, value) slab back to the C side.  Safe to call with
        `handle=None` (no-op) so pool code doesn't need to special-case
        empty launches.
        """
        if handle is None:
            return
        policy_cpu, values_cpu, event, seg_searches, seg_ptrs, seg_counts = handle
        if event is not None:
            event.synchronize()
        policy_probs = policy_cpu.numpy()
        values_np    = values_cpu.numpy().flatten()

        offset = 0
        for ss, ptr, n in zip(seg_searches, seg_ptrs, seg_counts):
            _lib.mcgs_commit_batch(
                ptr,
                policy_probs[offset:offset + n].ctypes.data,
                values_np[offset:offset + n].ctypes.data,
                ctypes.c_int(n),
            )
            if not ss._root_expanded:
                if self.dirichlet_alpha > 0.0:
                    _lib.mcgs_apply_root_dirichlet(
                        ptr,
                        ctypes.c_float(self.dirichlet_alpha),
                        ctypes.c_float(self.dirichlet_eps),
                    )
                ss._root_expanded = True
            offset += n

    def _expand_batch(self, searches: 'list[MCGSSearch]') -> None:
        """
        Synchronous one-shot: launch the forward pass and wait for it before
        returning.  Retained for callers that don't double-buffer (single-game
        `search()`, MM-mix pools) and as a fallback.
        """
        handle = self._launch_forward(searches)
        if handle is not None:
            self._collect_and_commit(handle)

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
