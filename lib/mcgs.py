"""
Gumbel AlphaZero MCGS backed by micro_mcts.c (C extension via ctypes).

The C extension manages the search tree and transposition table on a slab.

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
import warnings
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import Board


#  DLL loading

def _find_mcts_dll() -> pathlib.Path:
    suffixes = [".dll"] if sys.platform == "win32" else [".so"]
    roots = [
        pathlib.Path(__file__).parent.parent / "build",  # dev: native build
        pathlib.Path().absolute() / "build",             # dev: native build, cwd-relative
        pathlib.Path(__file__).parent,                   # portable build / installed wheel
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

# Arena-sized constructor: the TT persists across a game's moves, so a high
# simulation budget needs a much larger node/edge arena than the default (see
# the slab-capacity comment in src/micro_mcts.c).
_lib.mcgs_create_ex.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int]
_lib.mcgs_create_ex.restype  = ctypes.c_void_p

_lib.mcgs_clear.argtypes = [ctypes.c_void_p]
_lib.mcgs_clear.restype  = None

# All exports below are REQUIRED; stale binaries fail loudly here rather than
# silently degrading - rebuild via `make dll dll-native`.
_lib.mcgs_set_sigma_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.mcgs_set_sigma_scale.restype  = None

_lib.mcgs_get_best_action.argtypes = [ctypes.c_void_p]
_lib.mcgs_get_best_action.restype  = ctypes.c_int

_lib.mcgs_set_completion_n0.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.mcgs_set_completion_n0.restype  = None

_lib.mcgs_set_num_simulations.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.mcgs_set_num_simulations.restype  = None

_lib.mcgs_destroy.argtypes = [ctypes.c_void_p]
_lib.mcgs_destroy.restype  = None

_lib.mcgs_tt_size.argtypes = [ctypes.c_void_p]
_lib.mcgs_tt_size.restype  = ctypes.c_int
_lib.mcgs_edge_used.argtypes = [ctypes.c_void_p]
_lib.mcgs_edge_used.restype  = ctypes.c_int

# search lifecycle
_lib.mcgs_start_search.argtypes = [
    ctypes.c_void_p,              # MCGSInstance*
    ctypes.c_void_p,              # bool py_board[7][7][2] (flat 98 bytes) — void* to skip cast
    ctypes.c_bool,                # bool turn
    ctypes.c_int,                 # int clock (halfmove clock at root)
]
_lib.mcgs_start_search.restype = ctypes.c_void_p  # MCGSSearchState*

_lib.mcgs_set_clock_obs.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.mcgs_set_clock_obs.restype  = None

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

# obs-assembly fold: writes perspective-flipped float32 obs straight into the
# batch buffer, skipping the bool-board + np.concatenate + np.where CPU work.
_lib.mcgs_get_pending_obs.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,                 # float obs_out[n * 196] — void* to skip ctypes.cast
]
_lib.mcgs_get_pending_obs.restype = ctypes.c_int

_lib.mcgs_commit_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,                 # float policies_flat[n * 1225] — void* to skip ctypes.cast
    ctypes.c_void_p,                 # float values[n]               — skip ctypes.cast overhead
    ctypes.c_int,
]
_lib.mcgs_commit_batch.restype = None

# Multi-search batch API: the pool makes ONE ctypes call per group per phase
# instead of one per slot (step/done/pending/obs/commit were ~4M ctypes
# crossings per minute at pool 512 - ~10% of self-play wall clock).
# All take a uintp array of MCGSSearchState* as void*.
_lib.mcgs_step_many.argtypes = [
    ctypes.c_void_p,                 # MCGSSearchState *ss_arr[n]
    ctypes.c_int,
    ctypes.c_void_p,                 # int32 done_out[n]
]
_lib.mcgs_step_many.restype = None

_lib.mcgs_pending_counts.argtypes = [
    ctypes.c_void_p,                 # MCGSSearchState *ss_arr[n]
    ctypes.c_int,
    ctypes.c_void_p,                 # int32 counts_out[n]
]
_lib.mcgs_pending_counts.restype = None

_lib.mcgs_get_pending_obs_many.argtypes = [
    ctypes.c_void_p,                 # MCGSSearchState *ss_arr[n]
    ctypes.c_int,
    ctypes.c_void_p,                 # float obs_out[total * 196]
]
_lib.mcgs_get_pending_obs_many.restype = ctypes.c_int

_lib.mcgs_commit_batch_many.argtypes = [
    ctypes.c_void_p,                 # MCGSSearchState *ss_arr[n]
    ctypes.c_void_p,                 # int32 counts[n]
    ctypes.c_int,
    ctypes.c_void_p,                 # float policies_flat[total * 1225]
    ctypes.c_void_p,                 # float values[total]
]
_lib.mcgs_commit_batch_many.restype = None

_lib.mcgs_init()


# Pad inference batches up to the next power-of-two bucket.  Defaults on for
# ROCm, where MIOpen re-searches conv kernels for every distinct batch shape
# (no immediate-mode find-db on gfx1151) - unpadded, self-play throughput
# collapses ~17x as the pending-leaf count varies step to step.  Also required
# (and set by train_mcts --cudagraphs) when inference is compiled with
# mode="reduce-overhead": CUDA graphs record one graph per input shape, so the
# shape set must stay small.  No-op cost elsewhere: padded rows are sliced off
# after the forward.
PAD_BATCH_POW2 = bool(torch.version.hip)


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

    @property
    def root_value(self) -> float:
        """Root Q-value from the mover's perspective (value_sum / visit_count)."""
        return float(_lib.mcgs_get_root_value(self._ptr))

    @property
    def best_action(self) -> int:
        """Sequential Halving winner of a finished search (-1 if degenerate).

        This is the action the search budget certified; play it at
        temperature 0 instead of argmax(result), whose completed-Q logits
        can be reshuffled by low-visit Q noise.
        """
        return int(_lib.mcgs_get_best_action(self._ptr))

    def step(self) -> None:
        _lib.mcgs_step(self._ptr)


def _search_ptr_array(searches: 'list[MCGSSearch]') -> npt.NDArray[np.uintp]:
    """Pack search-state pointers into a uintp array for the *_many C calls."""
    return np.fromiter(((ss._ptr or 0) for ss in searches),
                       dtype=np.uintp, count=len(searches))


def step_searches(searches: 'list[MCGSSearch]') -> npt.NDArray[np.int32]:
    """Step every search once (one C call) and return per-search done flags."""
    n = len(searches)
    done = np.empty(n, dtype=np.int32)
    if n:
        ptrs = _search_ptr_array(searches)
        _lib.mcgs_step_many(ptrs.ctypes.data, n, done.ctypes.data)
    return done


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
        sigma_scale: float = 1.0,
        completion_n0: float | None = None,
        clock_obs: bool | None = None,
        node_cap: int = 0,
        edge_cap: int = 0,
    ) -> None:
        self.network         = network
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.gumbel_k        = gumbel_k
        self.sigma_scale     = sigma_scale
        self.completion_n0   = completion_n0
        self.last_root_value: float = 0.0
        self.last_best_action: int = -1
        self._device         = next(network.parameters()).device
        self._ptr            = _lib.mcgs_create_ex(
            num_simulations, ctypes.c_float(c_puct), gumbel_k,
            int(node_cap), int(edge_cap),
        )
        if not self._ptr:
            raise MemoryError(
                f"mcgs_create_ex failed (node_cap={node_cap}, edge_cap={edge_cap}); "
                "the arena request was probably too large"
            )
        if sigma_scale != 1.0:
            _lib.mcgs_set_sigma_scale(self._ptr, ctypes.c_float(sigma_scale))
        if completion_n0 is not None:
            _lib.mcgs_set_completion_n0(self._ptr, ctypes.c_float(completion_n0))
        if clock_obs is None:
            # Auto: net2 nets train with the halfmove clock in obs ch3; legacy
            # nets trained with an all-zero plane and must keep seeing zeros.
            # (Unwrap torch.compile's OptimizedModule before the arch check.)
            from lib.net2 import Net2
            clock_obs = isinstance(getattr(network, '_orig_mod', network), Net2)
        self.clock_obs = bool(clock_obs)
        if self.clock_obs:
            _lib.mcgs_set_clock_obs(self._ptr, 1)

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
    def edge_used(self) -> int:
        """Edge-slab high-water mark - the binding arena constraint (see C side)."""
        return _lib.mcgs_edge_used(self._ptr)

    @property
    def transposition_table(self) -> _TTProxy:
        """Monitoring proxy: len() returns C TT node count."""
        return _TTProxy(self._ptr)

    def clear(self) -> None:
        """Free all C nodes and reset the transposition table."""
        _lib.mcgs_clear(self._ptr)

    def set_num_simulations(self, n: int) -> None:
        """Change the per-search simulation budget (takes effect at the next
        start_search; in-flight searches keep the N they were started with)."""
        if not hasattr(_lib, "mcgs_set_num_simulations"):
            raise RuntimeError(
                "per-move simulation budgets require a micro_mcts build with "
                "mcgs_set_num_simulations (rebuild via `make dll`)")
        self.num_simulations = n
        _lib.mcgs_set_num_simulations(self._ptr, n)

    def advance_tree(self, action: int) -> None:  # noqa: ARG002
        """No-op: the TT persists across moves so Q-values are reused automatically."""

    def set_clock_obs(self, enable: bool) -> None:
        """Expose the halfmove clock as obs channel 3 (clock/100).  Keep off
        for legacy nets, which were trained with an all-zero channel 3."""
        _lib.mcgs_set_clock_obs(self._ptr, 1 if enable else 0)

    def start_search(
        self,
        board: Board,
        turn: bool,
        hint_root: object = None,  # ignored in C version
        move_count: int | None = None,
        clock: int = 0,
    ) -> MCGSSearch:
        """Start a step-wise search. hint_root is accepted but not used.

        clock is the halfmove clock at the root (plies since the last clone
        move, counting jumps and passes); search walks tick/reset it per edge
        and value clock-expired continuations as draws."""
        board_c = np.ascontiguousarray(board, dtype=np.bool_)
        ptr = _lib.mcgs_start_search(
            self._ptr,
            board_c.ctypes.data,
            ctypes.c_bool(turn),
            ctypes.c_int(clock),
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
                board_c.ctypes.data,
                ctypes.c_bool(turn),
                ctypes.c_int(clock),
            )
            if not ptr:
                warnings.warn(
                    "mcgs_start_search failed again after clear - "
                    "returning null search; game will recover via uniform-policy fallback.",
                    RuntimeWarning, stacklevel=2,
                )
        return MCGSSearch(ptr)

    def search(self, board: Board, turn: bool,
               clock: int = 0) -> npt.NDArray[np.float32]:
        """
        Run Gumbel MCGS from position (single-game, non-batched).

        Returns improved action probability distribution (1225 floats).
        After returning, self.last_root_value holds the root Q (mover's perspective).
        """
        ss = self.start_search(board, turn, clock=clock)
        while not ss.done:
            self._expand_batch([ss])
            ss.step()
        self.last_root_value: float = ss.root_value
        self.last_best_action: int = ss.best_action
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
        # Pass 1: collect pending counts per slot (one C call for the whole
        # group). The C pending set is stable until mcgs_step/commit, so it's
        # safe to read counts now and fetch the obs in a second pass below.
        n_searches = len(searches)
        if n_searches == 0:
            return None
        seg_ptrs = _search_ptr_array(searches)
        seg_counts = np.empty(n_searches, dtype=np.int32)
        _lib.mcgs_pending_counts(seg_ptrs.ctypes.data, n_searches,
                                 seg_counts.ctypes.data)
        N = int(seg_counts.sum())
        if N == 0:
            return None

        # Pow2 batch-shape bucketing - see PAD_BATCH_POW2 above; the padded
        # rows are sliced off again below.
        N_pad = (1 << (N - 1).bit_length()) if PAD_BATCH_POW2 else N

        # C writes the perspective-flipped float32 obs for every slot straight
        # into the batch buffer in one call (board_to_obs folded into the
        # bitboard unpack). No bool boards, no concatenate, no np.where.
        # Only the ROCm padding rows need explicit zeroing.
        obs_batch = np.empty((N_pad, 7, 7, 4), dtype=np.float32)
        if N_pad > N:
            obs_batch[N:] = 0.0
        _lib.mcgs_get_pending_obs_many(seg_ptrs.ctypes.data, n_searches,
                                       obs_batch.ctypes.data)

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

        return (policy_cpu, values_cpu, event, seg_ptrs, seg_counts, n_searches)

    def _collect_and_commit(self, handle) -> None:
        """
        Sync the CUDA event produced by `_launch_forward`, then commit the
        whole (policy, value) slab back to the C side in one call.  Safe to
        call with `handle=None` (no-op) so pool code doesn't need to
        special-case empty launches.
        """
        if handle is None:
            return
        policy_cpu, values_cpu, event, seg_ptrs, seg_counts, n_searches = handle
        if event is not None:
            event.synchronize()
        policy_probs = np.ascontiguousarray(policy_cpu.numpy())
        values_np    = np.ascontiguousarray(values_cpu.numpy()).reshape(-1)
        _lib.mcgs_commit_batch_many(
            seg_ptrs.ctypes.data, seg_counts.ctypes.data, n_searches,
            policy_probs.ctypes.data, values_np.ctypes.data,
        )

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
        best_action: int | None = None,
    ) -> int:
        """Select action from MCGS probability distribution (board/turn ignored).

        At temperature 0, plays `best_action` (the Sequential Halving winner)
        when the caller provides one; falling back to argmax(result) is only
        for callers without a live search (e.g. uniform recovery paths).
        """
        if temperature == 0:
            if best_action is not None and best_action >= 0:
                return int(best_action)
            return int(np.argmax(action_probs))
        if temperature == 1.0:
            probs = action_probs.copy()   # x**1.0 == x bit-exactly; skip the pow
        else:
            probs = action_probs ** (1.0 / temperature)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            return int(np.argmax(action_probs))
        return int(np.random.choice(len(probs), p=probs))
