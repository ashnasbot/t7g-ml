"""Runtime discovery of the *data bundle* (models + eval DB) that ships
alongside -- but separately from -- the installed wheel.

The wheel is code + the C shared libraries (and, optionally, ``lib/best.pt``).
The trained checkpoints (``models/``, ~GBs) and the evaluation database
(``matches.jsonl`` + ``players.json``) are large and change independently, so
they are distributed as a separate bundle the user drops next to wherever they
run the tool.  These helpers locate that bundle *at run time* -- so an installed
``play-t7g`` finds models/DB relative to the **invocation** (cwd), an explicit
env var, or a ``--data-dir`` flag, never baked into the wheel.

Search order (first existing hit wins), for both the models dir and the DB dir:

  1. an explicit override  -- ``set_data_root()`` / ``T7G_DATA_DIR``, or the
     narrow ``T7G_MODELS_DIR`` / ``T7G_EVAL_DB``
  2. the current working directory   -- ``./models``, ``./eval_db`` or
     ``./debug/eval_db``  (the "local to invocation" case)
  3. the source checkout that contains this file  -- the dev-tree fallback

This module is deliberately dependency-free (os + pathlib) so it can be
imported from the torch/mp-free :mod:`lib.eval_db` without pulling anything in.
"""
from __future__ import annotations

import os
import pathlib

# The source checkout (…/lib/paths.py -> repo root).  Present only in a dev
# tree; harmless when installed (the branch simply never matches).
_REPO = pathlib.Path(__file__).resolve().parent.parent

# Process-wide override set by --data-dir before any path is resolved.  Takes
# precedence over the environment so a CLI flag always wins.
_DATA_ROOT_OVERRIDE: pathlib.Path | None = None


def set_data_root(path: str | os.PathLike | None) -> None:
    """Pin the data-bundle root for this process (e.g. from ``--data-dir``).

    ``None`` clears it, falling back to the environment / cwd search.
    """
    global _DATA_ROOT_OVERRIDE
    _DATA_ROOT_OVERRIDE = pathlib.Path(path).expanduser() if path else None


def _data_root() -> pathlib.Path | None:
    """The bundle root from the flag override or ``T7G_DATA_DIR`` (may not exist)."""
    if _DATA_ROOT_OVERRIDE is not None:
        return _DATA_ROOT_OVERRIDE
    env = os.environ.get("T7G_DATA_DIR")
    return pathlib.Path(env).expanduser() if env else None


def _first_existing(cands, default: pathlib.Path) -> pathlib.Path:
    for c in cands:
        if c is not None and pathlib.Path(c).is_dir():
            return pathlib.Path(c)
    return default


def models_dir() -> pathlib.Path:
    """Directory holding ``<run>/<iter_NNNN>.pt`` checkpoints.

    Returns the first existing candidate; if none exists yet, ``./models``
    (a sensible writable default under the invocation dir).
    """
    root = _data_root()
    env = os.environ.get("T7G_MODELS_DIR")
    cwd = pathlib.Path.cwd()
    return _first_existing(
        [pathlib.Path(env).expanduser() if env else None,
         (root / "models") if root else None,
         cwd / "models",
         _REPO / "models"],
        default=cwd / "models",
    )


def eval_db_dir() -> pathlib.Path:
    """Directory holding ``matches.jsonl`` + ``players.json``.

    Accepts either a clean ``eval_db/`` bundle layout or the repo's
    ``debug/eval_db/``.  Falls back to ``./debug/eval_db`` (created on write).
    """
    root = _data_root()
    env = os.environ.get("T7G_EVAL_DB")
    cwd = pathlib.Path.cwd()
    return _first_existing(
        [pathlib.Path(env).expanduser() if env else None,
         (root / "eval_db") if root else None,
         (root / "debug" / "eval_db") if root else None,
         cwd / "eval_db",
         cwd / "debug" / "eval_db",
         _REPO / "debug" / "eval_db"],
        default=cwd / "debug" / "eval_db",
    )


def find_checkpoint(pid_or_path: str) -> pathlib.Path | None:
    """Resolve a checkpoint from a player id (``run/iter_0170``) or a path.

    Tries the argument as-is, then ``<models_dir>/<pid>.pt``.  Returns ``None``
    if nothing is found (caller decides whether that is fatal).
    """
    p = pathlib.Path(pid_or_path).expanduser()
    if p.suffix == ".pt" and p.is_file():
        return p
    cand = models_dir() / (pid_or_path + ".pt")
    return cand if cand.is_file() else None


def bundled_model() -> pathlib.Path | None:
    """The optional default checkpoint shipped inside the wheel (``lib/best.pt``)."""
    p = pathlib.Path(__file__).resolve().parent / "best.pt"
    return p if p.is_file() else None


def describe() -> str:
    """One-line summary of where data is being resolved from (for --data-dir echo)."""
    return f"models={models_dir()}  eval_db={eval_db_dir()}"
