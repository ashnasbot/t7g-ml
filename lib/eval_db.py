"""
Offline evaluation database + rating fitters (WHR / Bradley-Terry).

This is the *out-of-band* rating layer described in
``docs/offline_eval_db_design.md``.  The training loop keeps a cheap,
non-saturating self-anchored gate in the hot path (see the promotion block in
``scripts/train_mcts.py``); this module is the rigorous, low-variance rating
that runs offline over a persisted match database and produces the calibrated
Elo-vs-training-time curve across checkpoints and across runs.

Design invariants (do not violate):

  * **Each checkpoint is a distinct, immutable player.**  You cannot model "the
    network" as one time-varying player because iterA-vs-iterB would be a
    self-match and Bradley-Terry gives 0.5 identically.  Temporal smoothing
    instead enters as a Gauss-Markov (Wiener) prior linking *consecutive*
    checkpoints of the same run (see ``fit_whr``).

  * **Ratings are search-config dependent.**  Only ever aggregate/fit games
    played at the *same* config (sims, c_puct, gumbel_k, sigma_scale,
    completion_n0, engine).  Every stored match carries a ``config_hash`` and
    ``load_counts`` refuses to mix hashes.

This module is deliberately free of torch / multiprocessing so the fits and
persistence can be unit-tested in milliseconds.  The expensive game-playing
lives in ``scripts/eval_db.py``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Iterable

import numpy as np

from lib import paths as _paths

# ---------------------------------------------------------------------------
# Paths / canonical config
# ---------------------------------------------------------------------------

# The DB location is resolved *lazily* (per call) from the invocation dir / env
# / --data-dir override (see lib/paths.py), so a shipped tool finds a DB bundle
# placed next to where it runs -- even when the override is set after import.
# In the dev tree this lands on repo/debug/eval_db, unchanged.  Every function
# still accepts an explicit ``path=`` (tests pass their own tmp paths); the
# module-level names ``DB_DIR`` / ``MATCHES_PATH`` / ``PLAYERS_PATH`` remain
# readable (dynamically) via the module ``__getattr__`` below.

def _db_dir() -> str:
    return str(_paths.eval_db_dir())


def matches_path() -> str:
    return os.path.join(_db_dir(), "matches.jsonl")


def players_path() -> str:
    return os.path.join(_db_dir(), "players.json")


def __getattr__(name: str):
    if name == "DB_DIR":
        return _db_dir()
    if name == "MATCHES_PATH":
        return matches_path()
    if name == "PLAYERS_PATH":
        return players_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# The live-gauntlet search config (scripts/train_mcts.py).  This is the one that
# reproduces the 2026-07-13 rating scale; `add`/`fit` default to it so numbers
# stay comparable with pool.json and the memory record.  NOTE the sigma/n0
# fields: the old archive rating script omitted them and rated at the wrong
# config -- folding them in here is the canonical fix.
DEFAULT_CONFIG = {
    "sims": 500,
    "c_puct": 1.3,
    "gumbel_k": 16,
    "sigma_scale": 1.0,
    "completion_n0": 50.0,
    "engine": "micro3",   # minimax engine used for MM anchors
}

# The definitive gauge of the ladder.  ``stauf`` is the original 7th Guest
# microscope AI (extracted from ScummVM's Groovie engine, exposed via
# lib/cell_dll.so); beating it is the whole point of the project, so it -- not a
# minimax proxy -- pins the scale.  It is the SOLE fixed anchor: everything else
# (MM depths, nets) floats and is rated *relative to stauf*.  1000 is an
# arbitrary reference offset (only differences are meaningful).
#
# MM depths are deterministic, so once their Elo has been derived on this gauge
# they can be frozen as extra fixed anchors (`eval_db pin MM7=<derived>`), which
# is consistent by construction -- they were measured against stauf, not
# assumed from the retired MM3=1000 scale.
STAUF_PLAYER_ID = "stauf"
STAUF_ANCHOR_ELO = 1000.0
# Canonical original-game Stauf: depth 6 with cumulative move-count cycling
# (empirically matched to real-game recordings -- see archive identify_stauf.py /
# find_stauf_line.py).  depth 6 -> depths[12..14] = {3,2,2} plies by move index.
STAUF_DEPTH = 6


def eve_config_hash(stauf_depth: int = STAUF_DEPTH, opening_plies: int = 4,
                    mm_engine: str = "micro3") -> str:
    """Config hash for the deterministic engine-vs-engine low-end ladder.

    These games have no net search config; what defines them is the Stauf depth,
    the minimax engine, and how many random opening plies seed game variety.
    Kept in its own hash bucket so ladder games never mix with net-rating games.
    """
    canon = {"kind": "engine_v_engine", "stauf_depth": int(stauf_depth),
             "opening_plies": int(opening_plies), "mm_engine": mm_engine}
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return "eve" + hashlib.sha1(blob.encode()).hexdigest()[:6]

# Natural-log Elo scale factor: gamma = 10**(elo/400) = exp(C_LN10_400 * elo).
_C = math.log(10.0) / 400.0


def _sigmoid(x: float) -> float:
    """Overflow-safe logistic (ratings can transiently blow up mid-sweep)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def config_hash(config: dict) -> str:
    """Short, order-independent hash of a search-config dict.

    Only keys present in ``DEFAULT_CONFIG`` participate, filled from the
    default when missing, so callers can pass partial overrides and still land
    on a stable hash.  Floats are normalised so ``500`` and ``500.0`` agree.
    """
    merged = {**DEFAULT_CONFIG, **(config or {})}
    canon = {k: merged[k] for k in sorted(DEFAULT_CONFIG)}
    # Normalise numeric types to one canonical form so 500 == 500.0 and
    # 1.30 == 1.3 regardless of whether they arrived as int or float.
    for k, v in canon.items():
        if isinstance(v, (int, float)):
            fv = float(v)
            canon[k] = int(fv) if fv.is_integer() else round(fv, 6)
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Player registry
# ---------------------------------------------------------------------------

def load_players(path: str | None = None) -> dict:
    """Return the ``{player_id: meta}`` registry (empty dict if none yet)."""
    if path is None:
        path = players_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _write_players(reg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def register_player(player_id: str, meta: dict, path: str | None = None) -> dict:
    """Insert/refresh one player's metadata (merge); returns the full registry.

    ``meta`` keys: ``kind`` ("net"|"mm"|"stauf"), ``path`` or ``depth``, ``run``,
    ``iteration``, ``arch``, and ``fixed_elo`` for pinned anchors.
    """
    if path is None:
        path = players_path()
    reg = load_players(path)
    reg[player_id] = {**reg.get(player_id, {}), **meta}
    _write_players(reg, path)
    return reg


def set_fixed_elo(player_id: str, elo: float | None, path: str | None = None) -> dict:
    """Pin ``player_id`` at ``elo`` (freeze a derived deterministic anchor), or
    unpin it when ``elo`` is None.  Returns the full registry."""
    if path is None:
        path = players_path()
    reg = load_players(path)
    entry = reg.setdefault(player_id, {})
    if elo is None:
        entry.pop("fixed_elo", None)
    else:
        entry["fixed_elo"] = float(elo)
    _write_players(reg, path)
    return reg


# ---------------------------------------------------------------------------
# Match store (append-only jsonl)
# ---------------------------------------------------------------------------

def append_matches(rows: Iterable[dict], path: str | None = None) -> int:
    """Append match rows; returns the number written.

    Each row: ``{"a", "b", "a_is_blue", "result", "config_hash", "ts"}`` with
    ``result`` in {+1, 0, -1} from a's perspective.  ``ts`` is filled if
    absent.
    """
    if path is None:
        path = matches_path()
    rows = list(rows)
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a") as f:
        for r in rows:
            r = {**r}
            r.setdefault("ts", now)
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


def load_matches(config_hash_filter: str | None = None,
                 path: str | None = None) -> list[dict]:
    """Load raw match rows, optionally filtered to a single ``config_hash``."""
    if path is None:
        path = matches_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if config_hash_filter is None or row.get("config_hash") == config_hash_filter:
                out.append(row)
    return out


def load_counts(config_hash_filter: str,
                path: str | None = None) -> tuple[list[str], dict]:
    """Aggregate stored matches for one config into pairwise W/D/L counts.

    Returns ``(names, counts)`` where ``names`` is the sorted list of player
    ids that appear, and ``counts[(i, j)] = [wins_i, draws, wins_j]`` for
    ``i < j`` (index into ``names``).  Colour is folded away -- only the
    result matters for BT/WHR.
    """
    rows = load_matches(config_hash_filter, path)
    names = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    idx = {n: k for k, n in enumerate(names)}
    counts: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        ia, ib = idx[r["a"]], idx[r["b"]]
        res = r["result"]          # from a's perspective
        i, j = (ia, ib) if ia < ib else (ib, ia)
        if ia > ib:
            res = -res             # re-orient to i's perspective
        cell = counts.setdefault((i, j), [0, 0, 0])
        cell[0 if res > 0 else (2 if res < 0 else 1)] += 1
    return names, counts


def pair_game_count(config_hash_filter: str, a: str, b: str,
                    path: str | None = None) -> int:
    """How many games between players ``a`` and ``b`` already exist at a config.

    Used by ``add`` to skip pairs already at their target game count
    (idempotent accumulation -- re-running ``add`` plays nothing new).
    """
    rows = load_matches(config_hash_filter, path)
    pair = {a, b}
    return sum(1 for r in rows if {r["a"], r["b"]} == pair)


# ---------------------------------------------------------------------------
# Static Bradley-Terry fit (Hunter's MM) -- the Phase-1 fitter and the
# w -> inf validation target for WHR.
# ---------------------------------------------------------------------------

def fit_bradley_terry(names, counts, virtual_draws=2.0, iters=2000):
    """Fit BT strengths from pairwise counts ``{(i, j): [w_i, d, w_j]}``.

    Returns Elo ratings (mean-normalised; caller re-anchors).  Draws count half
    a win each; ``virtual_draws`` pseudo-games per pair keep a 100-0 pairing
    finite.  This is the fit that produced the 2026-07-13 scale.
    """
    n = len(names)
    score = np.zeros((n, n))
    games = np.zeros((n, n))
    for (i, j), (w, d, loss) in counts.items():
        score[i, j] += w + 0.5 * d + 0.5 * virtual_draws
        score[j, i] += loss + 0.5 * d + 0.5 * virtual_draws
        games[i, j] += w + d + loss + virtual_draws
        games[j, i] += w + d + loss + virtual_draws

    gamma = np.ones(n)
    wins_total = score.sum(axis=1)
    for _ in range(iters):
        denom = (games / (gamma[:, None] + gamma[None, :])).sum(axis=1)
        gamma = wins_total / np.maximum(denom, 1e-12)
        gamma /= gamma.mean()
    return 400.0 * np.log10(np.maximum(gamma, 1e-12))


def bootstrap_ci(names, counts, reps=200, virtual_draws=2.0, seed=0):
    """95% CI half-width per player via multinomial resampling of each pairing."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        resampled = {}
        for pair, (w, d, loss) in counts.items():
            total = w + d + loss
            if total == 0:
                continue
            rw, rd, rl = rng.multinomial(total, [w / total, d / total, loss / total])
            resampled[pair] = [rw, rd, rl]
        samples.append(fit_bradley_terry(names, resampled, virtual_draws))
    arr = np.array(samples)
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    return (hi - lo) / 2.0


# ---------------------------------------------------------------------------
# Whole-History Rating (Coulom coordinate-Newton with a Wiener prior)
# ---------------------------------------------------------------------------

def _anchor_indices(names, anchors):
    """Map ``{player_id: elo}`` anchors to ``{index: elo}``; ignore absentees."""
    idx = {n: k for k, n in enumerate(names)}
    return {idx[n]: float(e) for n, e in (anchors or {}).items() if n in idx}


def fit_whr(names, counts, chains=None, w=None, anchors=None,
            virtual_draws=2.0, max_iters=1000, tol=1e-4, damping=1.0):
    """Whole-History Rating via Coulom's coordinate-Newton sweep.

    Parameters
    ----------
    names, counts
        As in :func:`fit_bradley_terry` -- ``counts[(i, j)] = [w_i, d, w_j]``.
    chains : list[list[int]] | None
        Temporal chains: each inner list is player indices in *iteration order*
        for one run.  Consecutive entries are linked by the Wiener prior.  An
        entry may be ``(index, iteration)`` to encode uneven iteration gaps;
        a bare ``index`` assumes unit spacing.
    w : float | None
        Wiener drift in **Elo per sqrt(iteration)**: consecutive checkpoints
        satisfy ``r_{N+1} - r_N ~ N(0, w**2 * delta_iters)``.  ``None`` or
        ``inf`` disables the prior, recovering independent Bradley-Terry (the
        built-in validation target).
    anchors : dict | None
        ``{player_id: elo}`` players pinned at fixed Elo (gauge fix).  If none
        given, the fit is mean-normalised like ``fit_bradley_terry``.

    Returns
    -------
    (elo, hess) : (np.ndarray, np.ndarray)
        Elo per player and the per-player observed-information (positive) whose
        reciprocal square root is a cheap 1-D standard error.
    """
    n = len(names)
    r = np.zeros(n)                       # current Elo estimate
    anchor = _anchor_indices(names, anchors)
    for i, e in anchor.items():
        r[i] = e

    use_prior = w is not None and math.isfinite(w) and w > 0
    prior_prec = 1.0 / (w * w) if use_prior else 0.0   # precision per unit iter-gap

    # Pre-index each player's opponents + the ordered points it scored.
    #   games_from[i] = list of (j, n_ij, S_ij)  where S_ij is i's points.
    games_from: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
    for (i, j), (wi, d, wj) in counts.items():
        ni = wi + wj + d + virtual_draws
        si = wi + 0.5 * d + 0.5 * virtual_draws        # i's points off j
        games_from[i].append((j, ni, si))
        games_from[j].append((i, ni, ni - si))

    # Temporal neighbours: for each player, the (neighbour_index, delta_iters).
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if use_prior and chains:
        for chain in chains:
            norm = [(c if isinstance(c, (tuple, list)) else (c, k))
                    for k, c in enumerate(chain)]
            for (ia, ta), (ib, tb) in zip(norm, norm[1:]):
                delta = max(abs(float(tb) - float(ta)), 1e-6)
                neighbours[ia].append((ib, delta))
                neighbours[ib].append((ia, delta))

    free = [i for i in range(n) if i not in anchor]
    for _ in range(max_iters):
        max_step = 0.0
        for i in free:
            # --- likelihood gradient / curvature (concave, in Elo units) ---
            g = 0.0
            h = 0.0            # positive curvature magnitude ( = -d2/dr2 )
            for j, nij, sij in games_from[i]:
                p = _sigmoid(_C * (r[i] - r[j]))
                g += _C * (sij - nij * p)
                h += _C * _C * nij * p * (1.0 - p)
            # --- Wiener prior linking consecutive checkpoints ---
            for j, delta in neighbours[i]:
                prec = prior_prec / delta
                g += -prec * (r[i] - r[j])
                h += prec
            if h <= 0:
                continue
            # Clamp the Newton step: a far-away anchor on the first sweep can
            # otherwise overshoot into a numerical blow-up before converging.
            step = max(-400.0, min(400.0, damping * g / h))
            r[i] += step
            max_step = max(max_step, abs(step))
        if max_step < tol:
            break

    # Gauge: anchors pin the scale; otherwise mean-normalise like BT.
    if not anchor:
        r = r - r.mean()

    # Per-player observed information (for cheap 1-D CIs).
    hess = np.zeros(n)
    for i in range(n):
        h = 0.0
        for j, nij, _sij in games_from[i]:
            p = _sigmoid(_C * (r[i] - r[j]))
            h += _C * _C * nij * p * (1.0 - p)
        for j, delta in neighbours[i]:
            h += prior_prec / delta
        hess[i] = h
    return r, hess


def whr_ci95(hess):
    """95% CI half-width from per-player observed information (1.96 / sqrt(H))."""
    return np.where(hess > 0, 1.96 / np.sqrt(np.maximum(hess, 1e-12)), np.inf)


def reanchor(names, elo, anchor_name, anchor_elo=1000.0):
    """Shift an Elo vector so ``anchor_name`` sits at ``anchor_elo``."""
    k = names.index(anchor_name)
    return np.asarray(elo) - elo[k] + anchor_elo
