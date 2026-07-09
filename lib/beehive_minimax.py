"""
ctypes wrapper around beehive_4.so / beehive_4.dll.

Usage:
    from lib.beehive_minimax import beehive_best_move
    action = beehive_best_move(board, as_yellow=False, time_ms=1000)
"""
import ctypes
import pathlib

import numpy as np

from lib.beehive import N_CELLS

_lib = None


def _find_dll() -> str:
    import sys
    ext = "dll" if sys.platform == "win32" else "so"
    root = pathlib.Path(__file__).resolve().parent.parent
    candidates = [
        root / f"beehive4.{ext}",
        root / "lib" / f"beehive4.{ext}",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"beehive4.{ext} not found.\n"
        "Build it with:  make beehive4   (or: make beehive4-native)"
    )


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_find_dll())
        _lib.init_tt()

        _lib.find_best_move_timed.restype  = ctypes.c_int
        _lib.find_best_move_timed.argtypes = [
            ctypes.POINTER(ctypes.c_bool),  # game_board[N_CELLS][2]
            ctypes.c_int,                   # max_ms
            ctypes.c_bool,                  # as_yellow
        ]

        _lib.find_best_move.restype  = ctypes.c_int
        _lib.find_best_move.argtypes = [
            ctypes.POINTER(ctypes.c_bool),
            ctypes.c_int,   # depth
            ctypes.c_bool,  # as_yellow
        ]

        _lib.minimax_score.restype  = ctypes.c_float
        _lib.minimax_score.argtypes = [
            ctypes.POINTER(ctypes.c_bool),
            ctypes.c_int,
            ctypes.c_bool,
        ]
    return _lib


def beehive_score(board: np.ndarray, depth: int, as_yellow: bool) -> float:
    """Return the minimax score at the given depth (positive = good for mover)."""
    lib = _get_lib()
    arr = np.ascontiguousarray(board, dtype=np.bool_)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
    return float(lib.minimax_score(ptr, ctypes.c_int(depth), ctypes.c_bool(as_yellow)))


def beehive_best_move(board: np.ndarray, as_yellow: bool,
                      time_ms: int = 1000) -> int:
    """Return the best action index (0-1097) for the given player.

    board:     (N_CELLS, 2) bool_ array; [:, 0] = Red, [:, 1] = Yellow
    as_yellow: True if the caller is Yellow, False if Red
    time_ms:   search time budget in milliseconds
    """
    lib = _get_lib()
    # board is already (61, 2) bool_; ensure C-contiguous and correct dtype
    arr = np.ascontiguousarray(board, dtype=np.bool_)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
    return lib.find_best_move_timed(ptr, ctypes.c_int(time_ms),
                                    ctypes.c_bool(as_yellow))
