"""
Offline evaluation DB CLI -- schedule/play rating matches, fit WHR, plot curves.

This is the out-of-band rating benchmark from ``docs/offline_eval_db_design.md``.
It complements the in-loop self-anchored gate (which prevents saturation) with a
calibrated, low-variance Elo-vs-training-time curve across all checkpoints and
runs -- the thing you want for "which checkpoint is actually best", "did run B
beat run A", and clean progress plots.

    # play games: each net vs its temporal neighbours + the MM7 anchor
    python scripts/eval_db.py add models/run_selfgate/iter_00*.pt --mm 7 \\
        --vs neighbors --games 12 --pool-size 64

    # fit ratings (default: independent Bradley-Terry; pass --w to smooth)
    python scripts/eval_db.py fit --anchor MM7=1235
    python scripts/eval_db.py fit --w 40 --anchor MM7=1235   # Wiener-smoothed WHR

    # emit an Elo-vs-iteration curve for one run
    python scripts/eval_db.py curve run_selfgate

Ratings are search-config dependent: only games played at the *same* config are
ever fitted together (every match carries a ``config_hash``).  ``add`` defaults
to the live-gauntlet config (500 sims, c_puct 1.3, K16, sigma 1.0, n0 50), which
reproduces the 2026-07-13 rating scale.

Design notes: each checkpoint is a distinct immutable player; temporal
smoothing enters only as a Wiener prior linking consecutive checkpoints of the
same run (``--w`` Elo/sqrt(iter); ``w -> inf`` recovers independent BT).  See
lib/eval_db.py for the fitters and the reasoning.
"""
import argparse
import itertools
import multiprocessing
import os
import random
import re
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")

from lib import eval_db as edb                                     # noqa: E402
from lib.train_workers import (play_engine_vs_engine,              # noqa: E402
                               tournament_pool)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Agent construction for tournament_pool.
#
# `add` plays every scheduled game in ONE in-process batched pool rather than
# one-game-per-worker across a spawn Pool.  A 500-sim search costs the same
# number of *sequential* forwards however many games are in flight, and a
# forward carrying 200 leaves costs about what one carrying 10 does (it is
# latency-bound), so concurrency is nearly free -- see the tournament_pool
# docstring in lib/train_workers.py.  Search semantics per game are unchanged,
# so ratings stay on the same scale as previously banked games.
# ---------------------------------------------------------------------------

def _pool_agent(pid, reg, engine):
    """tournament_pool agent tuple for a registered player.

    Anchors are played at zero noise, fixed depth -- the config the fixed
    anchor ratings were fitted under.  Anchor-vs-anchor is never scheduled
    (both deterministic; MM depths are placed relative to stauf transitively
    through their shared net opponents).
    """
    kind = reg[pid]["kind"]
    if kind == "mm":
        return ("engine", reg[pid]["depth"], 0.0, engine, False)
    if kind == "stauf":
        return ("engine", reg[pid].get("depth", edb.STAUF_DEPTH), 0.0, "stauf", False)
    return ("net", pid)


# ---------------------------------------------------------------------------
# Deterministic engine-vs-engine ladder workers (no net/torch; C engines load
# lazily per process on first find_best_move call).
# ---------------------------------------------------------------------------

_eve_specs: list = []          # [(engine, depth), ...] indexed like players
_eve_opening_plies: int = 4


def _eve_worker_init(specs, opening_plies):
    global _eve_specs, _eve_opening_plies
    _eve_specs = specs
    _eve_opening_plies = opening_plies


def _eve_worker_game(args):
    """Play one deterministic engine-vs-engine game from a seeded random opening."""
    ia, ib, a_is_blue, seed = args
    rng = np.random.default_rng(seed)
    disc = play_engine_vs_engine(_eve_specs[ia], _eve_specs[ib], a_is_blue,
                                 opening_plies=_eve_opening_plies, rng=rng)
    return ia, ib, disc, a_is_blue


# ---------------------------------------------------------------------------
# Player identity
# ---------------------------------------------------------------------------

_ITER_RE = re.compile(r"iter[_-]?(\d+)")


def player_id_for(path: str) -> tuple[str, dict]:
    """Derive ``run/iter_NNNN`` id + registry metadata from a checkpoint path.

    ``models/run_selfgate/iter_0080.pt`` -> ``run_selfgate/iter_0080``.
    """
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _ITER_RE.search(stem)
    iteration = int(m.group(1)) if m else None
    pid = f"{run}/{stem}"
    return pid, {"kind": "net", "path": path, "run": run, "iteration": iteration}


# ---------------------------------------------------------------------------
# Matchmaking: which (player_a, player_b) pairs to schedule
# ---------------------------------------------------------------------------

def schedule_pairs(net_ids, net_meta, anchor_ids, vs):
    """Return the set of unordered ``{a, b}`` pairs to play.

    ``vs`` in {neighbors, net, all}.  Every net plays every anchor (stauf gauge
    + any MM depths); the net-vs-net structure is:
      neighbors -> consecutive checkpoints within each run (Wiener chain)
      net / all -> full round-robin among the supplied nets
    (``all`` is reserved to grow long-bond / cross-run matchmaking in Phase 3.)
    """
    pairs: set[frozenset] = set()
    # nets vs deterministic anchors (stauf pins the scale; MMs float onto it)
    for a in net_ids:
        for m in anchor_ids:
            pairs.add(frozenset((a, m)))
    if vs == "neighbors":
        by_run: dict[str, list] = {}
        for pid in net_ids:
            by_run.setdefault(net_meta[pid]["run"], []).append(pid)
        for run, ids in by_run.items():
            ids.sort(key=lambda p: (net_meta[p]["iteration"] or 0))
            for a, b in zip(ids, ids[1:]):
                pairs.add(frozenset((a, b)))
    else:  # net / all -> round-robin among nets
        for a, b in itertools.combinations(net_ids, 2):
            pairs.add(frozenset((a, b)))
    return pairs


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def cmd_add(args):
    import torch

    cfg = {"sims": args.sims}
    chash = edb.config_hash(cfg)
    mcts_kwargs = dict(
        num_simulations=args.sims,
        c_puct=edb.DEFAULT_CONFIG["c_puct"],
        gumbel_k=edb.DEFAULT_CONFIG["gumbel_k"],
        sigma_scale=edb.DEFAULT_CONFIG["sigma_scale"],
        completion_n0=edb.DEFAULT_CONFIG["completion_n0"],
    )
    engine = edb.DEFAULT_CONFIG["engine"]

    # Register net players + remember paths -> state dicts (loaded lazily below).
    net_ids, net_meta, net_paths = [], {}, {}
    for path in args.checkpoints:
        pid, meta = player_id_for(path)
        net_ids.append(pid)
        net_meta[pid] = meta
        net_paths[pid] = path
        edb.register_player(pid, meta)

    anchor_ids = []
    # stauf: the sole fixed gauge (original-game AI).  Included by default -- it
    # is the reference the whole ladder is measured against.
    if args.stauf:
        edb.register_player(edb.STAUF_PLAYER_ID,
                            {"kind": "stauf", "depth": edb.STAUF_DEPTH,
                             "fixed_elo": edb.STAUF_ANCHOR_ELO})
        anchor_ids.append(edb.STAUF_PLAYER_ID)
    # MM depths: float and get rated *relative to stauf* (no fixed_elo -- pinning
    # them at the retired MM3=1000 scale would dual-anchor inconsistently; freeze
    # their derived Elos later with `eval_db pin` once measured on this gauge).
    mm_depths = [int(d) for d in args.mm.split(",") if d.strip()]
    for d in sorted(mm_depths):
        pid = f"MM{d}"
        anchor_ids.append(pid)
        edb.register_player(pid, {"kind": "mm", "depth": d})

    pairs = schedule_pairs(net_ids, net_meta, anchor_ids, args.vs)

    # Dedup: only play up to the target game count per pair for this config.
    to_play = []            # list of (a_id, b_id, n_new)
    for pair in sorted(pairs, key=lambda s: sorted(s)):
        a, b = sorted(pair)
        have = edb.pair_game_count(chash, a, b)
        need = max(0, args.games - have)
        if need:
            to_play.append((a, b, need))
    if not to_play:
        print(f"config {chash}: all {len(pairs)} pairs already at "
              f">= {args.games} games -- nothing to play.")
        return

    # Instantiate the players actually needed.  Nets are loaded ONCE, in this
    # process, and shared by every pool slot that plays them.
    needed = sorted({p for a, b, _ in to_play for p in (a, b)})
    reg = edb.load_players()
    from lib.device_utils import get_device, load_compiled_network
    device = get_device()
    nets, agents = {}, {}
    for pid in needed:
        agents[pid] = _pool_agent(pid, reg, engine)
        if agents[pid][0] == "net":
            blob = torch.load(net_paths[pid], map_location="cpu", weights_only=True)
            state = blob["network"] if isinstance(blob, dict) and "network" in blob else blob
            net, _ = load_compiled_network(state, device,
                                           num_actions=args.num_actions, compile_net=False)
            net.eval()
            nets[pid] = net

    # Build the game list (alternate colours) and shuffle so a partial run
    # covers all pairs.  Net-vs-net randomises who moves first; games against
    # an engine anchor always start with Blue to move -- both are part of what
    # the anchors were rated under, so keep them distinct (as rate_vs_pool does).
    rng = random.Random(0)
    games = []
    for a, b, need in to_play:
        for g in range(need):
            a_is_blue = g % 2 == 0
            blue, green = (a, b) if a_is_blue else (b, a)
            both_nets = agents[a][0] == "net" and agents[b][0] == "net"
            first_turn = bool(rng.getrandbits(1)) if both_nets else True
            games.append(((a, b, a_is_blue), agents[blue], agents[green], first_turn))
    rng.shuffle(games)

    n_pairs = len(to_play)
    total_new = sum(n for _, _, n in to_play)
    pool_size = min(args.pool_size, len(games))
    print(f"config {chash}: {n_pairs} pairs need games, {total_new} to play, "
          f"{args.sims} sims/move, {pool_size} games in flight", flush=True)

    rows, t0, done = [], time.time(), 0
    tqdm_disabled = bool(os.environ.get("TQDM_DISABLE"))
    for r in tqdm(tournament_pool(nets, games, mcts_kwargs, pool_size=pool_size),
                  total=len(games), unit="game", desc="Playing"):
        a_id, b_id, a_is_blue = r["tag"]
        # Pool results are Blue-perspective; sign to a, then discretise the
        # material-ratio truncation result to a win/draw/loss.
        result = r["blue_result"] if a_is_blue else -r["blue_result"]
        disc = 1 if result > 1e-9 else (-1 if result < -1e-9 else 0)
        rows.append({
            "a": a_id, "b": b_id,
            "a_is_blue": bool(a_is_blue), "result": int(disc),
            "config_hash": chash,
        })
        done += 1
        if len(rows) >= 20:                      # periodic flush (crash-safe;
            edb.append_matches(rows)             # engine games can be ~1 min
            rows = []                            # each, keep the loss window small)
        if tqdm_disabled and done % 25 == 0:     # heartbeat for redirected logs
            dt = time.time() - t0
            print(f"  {done}/{len(games)} games  {dt/done:.2f}s/game  "
                  f"eta {(len(games)-done)*dt/done/60:.0f}m", flush=True)
    edb.append_matches(rows)
    dt = time.time() - t0
    print(f"Played {total_new} games in {dt:.0f}s ({dt/max(1,total_new):.2f}s/game) "
          f"-> {edb.MATCHES_PATH}")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def _build_chains(names, reg):
    """Temporal chains: for each run, its net checkpoints in iteration order.

    Returns ``[[(index, iteration), ...], ...]`` for the Wiener prior.
    """
    idx = {n: k for k, n in enumerate(names)}
    by_run: dict[str, list] = {}
    for pid in names:
        meta = reg.get(pid, {})
        if meta.get("kind") == "net" and meta.get("iteration") is not None:
            by_run.setdefault(meta["run"], []).append(pid)
    chains = []
    for run, ids in by_run.items():
        ids.sort(key=lambda p: reg[p]["iteration"])
        if len(ids) >= 2:
            chains.append([(idx[p], reg[p]["iteration"]) for p in ids])
    return chains


def _parse_anchors(anchor_args):
    out = {}
    for a in anchor_args or []:
        name, _, elo = a.partition("=")
        out[name] = float(elo)
    return out


def cmd_fit(args):
    chash = args.config or edb.config_hash({"sims": args.sims})
    names, counts = edb.load_counts(chash)
    if not names:
        print(f"No matches for config {chash}. Run `add` first.")
        return
    reg = edb.load_players()

    anchors = _parse_anchors(args.anchor)
    if not anchors:                              # default: pin fixed_elo players
        anchors = {n: reg[n]["fixed_elo"] for n in names
                   if reg.get(n, {}).get("fixed_elo") is not None}

    chains = _build_chains(names, reg)
    w = None if args.w is None else args.w       # None => independent BT
    elo, hess = edb.fit_whr(names, counts, chains=chains, w=w, anchors=anchors)
    ci = edb.whr_ci95(hess)

    # Bootstrap CI as a cross-check when the fit is plain BT (no prior).
    order = np.argsort(-elo)
    smoothing = "independent BT" if w is None or not np.isfinite(w) else f"WHR w={w}"
    print(f"\nconfig {chash} | {len(names)} players | {sum(sum(c) for c in counts.values())} "
          f"games | {smoothing}")
    print(f"{'player':<26}{'Elo':>7}  {'95% CI':>8}  {'W':>5} {'D':>4} {'L':>5}")
    wdl = _player_wdl(names, counts)
    for k in order:
        w_, d_, l_ = wdl[k]
        pin = " *" if names[k] in anchors else ""
        print(f"{names[k]:<26}{elo[k]:>7.0f}  {'+/-' + format(ci[k], '.0f'):>8}  "
              f"{w_:>5} {d_:>4} {l_:>5}{pin}")

    out = args.out or os.path.join(
        edb.DB_DIR, f"ratings_{chash}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    curves = _run_curves(names, elo, reg)
    with open(out, "w") as f:
        json.dump({
            "config_hash": chash,
            "smoothing": smoothing,
            "w": None if w is None else (None if not np.isfinite(w) else w),
            "anchors": anchors,
            "ratings": {names[k]: {"elo": round(float(elo[k]), 1),
                                   "ci95": round(float(ci[k]), 1)}
                        for k in range(len(names))},
            "curves": curves,
        }, f, indent=2)
    print(f"\nSaved {out}")


def _player_wdl(names, counts):
    """Per-player (W, D, L) totals from pairwise counts."""
    wdl = [[0, 0, 0] for _ in names]
    for (i, j), (wi, d, wj) in counts.items():
        wdl[i][0] += wi; wdl[i][1] += d; wdl[i][2] += wj
        wdl[j][0] += wj; wdl[j][1] += d; wdl[j][2] += wi
    return wdl


def _run_curves(names, elo, reg):
    """{run: [[iteration, elo], ...] sorted by iteration} for net players."""
    curves: dict[str, list] = {}
    for k, pid in enumerate(names):
        meta = reg.get(pid, {})
        if meta.get("kind") == "net" and meta.get("iteration") is not None:
            curves.setdefault(meta["run"], []).append([meta["iteration"], round(float(elo[k]), 1)])
    for run in curves:
        curves[run].sort()
    return curves


# ---------------------------------------------------------------------------
# curve
# ---------------------------------------------------------------------------

def cmd_curve(args):
    chash = args.config or edb.config_hash({"sims": args.sims})
    names, counts = edb.load_counts(chash)
    if not names:
        print(f"No matches for config {chash}.")
        return
    reg = edb.load_players()
    anchors = {n: reg[n]["fixed_elo"] for n in names
               if reg.get(n, {}).get("fixed_elo") is not None}
    chains = _build_chains(names, reg)
    w = None if args.w is None else args.w
    elo, _ = edb.fit_whr(names, counts, chains=chains, w=w, anchors=anchors)
    curves = _run_curves(names, elo, reg)
    series = curves.get(args.run)
    if not series:
        print(f"No net checkpoints for run '{args.run}' at config {chash}. "
              f"Known runs: {', '.join(sorted(curves)) or '(none)'}")
        return
    print(f"# Elo vs iteration -- run '{args.run}' (config {chash})")
    print(f"{'iter':>6}  {'elo':>7}")
    for it, e in series:
        print(f"{it:>6}  {e:>7.0f}")


# ---------------------------------------------------------------------------
# pin -- freeze a derived (deterministic) anchor's Elo, or unpin it
# ---------------------------------------------------------------------------

def cmd_pin(args):
    for spec in args.assignments:
        name, sep, elo = spec.partition("=")
        name = name.strip()
        if not sep:
            print(f"skip '{spec}': expected PLAYER=ELO (or PLAYER= to unpin)")
            continue
        if elo.strip() == "":
            edb.set_fixed_elo(name, None)
            print(f"unpinned {name}")
        else:
            edb.set_fixed_elo(name, float(elo))
            print(f"pinned {name} = {float(elo):.0f}")


# ---------------------------------------------------------------------------
# ladder -- deterministic low-end ladder: how much stronger did MM get vs stauf
# ---------------------------------------------------------------------------

def cmd_ladder(args):
    chash = edb.eve_config_hash(stauf_depth=args.stauf_depth,
                                opening_plies=args.opening_plies)
    # Players: canonical stauf gauge + minimax depths (all deterministic).
    edb.register_player(edb.STAUF_PLAYER_ID,
                        {"kind": "stauf", "depth": args.stauf_depth,
                         "fixed_elo": edb.STAUF_ANCHOR_ELO})
    ids = [edb.STAUF_PLAYER_ID]
    specs = [("stauf", args.stauf_depth)]
    for d in sorted(int(x) for x in args.mm.split(",") if x.strip()):
        edb.register_player(f"MM{d}", {"kind": "mm", "depth": d})
        ids.append(f"MM{d}")
        specs.append(("micro3", d))

    # Full round-robin (stauf-vs-MM AND MM-vs-MM keep the ladder connected).
    to_play = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        need = max(0, args.games - edb.pair_game_count(chash, ids[i], ids[j]))
        if need:
            to_play.append((i, j, need))

    if not to_play:
        print(f"config {chash}: ladder already at >= {args.games} games/pair.")
    else:
        master = np.random.default_rng()
        tasks = [(i, j, g % 2 == 0, int(master.integers(2**63)))
                 for i, j, need in to_play for g in range(need)]
        master.shuffle(tasks)
        total = sum(n for _, _, n in to_play)
        print(f"config {chash}: {len(to_play)} pairs, {total} games, "
              f"stauf@{args.stauf_depth}, opening_plies={args.opening_plies}, "
              f"{args.workers} workers")
        rows, t0 = [], time.time()
        with multiprocessing.get_context("spawn").Pool(
                processes=min(args.workers, len(tasks)),
                initializer=_eve_worker_init,
                initargs=(specs, args.opening_plies)) as pool:
            for ia, ib, disc, a_is_blue in tqdm(
                    pool.imap_unordered(_eve_worker_game, tasks),
                    total=len(tasks), unit="game", desc="Ladder"):
                rows.append({"a": ids[ia], "b": ids[ib], "a_is_blue": bool(a_is_blue),
                             "result": int(disc), "config_hash": chash})
                if len(rows) >= 200:
                    edb.append_matches(rows)
                    rows = []
        edb.append_matches(rows)
        print(f"Played {total} games in {time.time() - t0:.0f}s")

    # Fit + report (stauf pinned at 1000; MM depths float above).
    names, counts = edb.load_counts(chash)
    elo, hess = edb.fit_whr(names, counts,
                            anchors={edb.STAUF_PLAYER_ID: edb.STAUF_ANCHOR_ELO})
    ci = edb.whr_ci95(hess)
    wdl = _player_wdl(names, counts)
    order = np.argsort(-elo)
    print(f"\nLow-end ladder -- stauf gauge = {edb.STAUF_ANCHOR_ELO:.0f} "
          f"({sum(sum(c) for c in counts.values())} games)")
    print(f"{'player':<10}{'Elo':>7}  {'95% CI':>8}  {'W':>5} {'D':>4} {'L':>5}")
    for k in order:
        w_, d_, l_ = wdl[k]
        pin = " *" if names[k] == edb.STAUF_PLAYER_ID else ""
        print(f"{names[k]:<10}{elo[k]:>7.0f}  {'+/-' + format(ci[k], '.0f'):>8}  "
              f"{w_:>5} {d_:>4} {l_:>5}{pin}")
    best = names[order[0]]
    print(f"\nStrongest minimax is {best} at {elo[order[0]]:.0f} Elo -- "
          f"{elo[order[0]] - edb.STAUF_ANCHOR_ELO:+.0f} vs the original Stauf.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="schedule + play matches, append to the DB")
    a.add_argument("checkpoints", nargs="+", help="checkpoint .pt files")
    a.add_argument("--stauf", action=argparse.BooleanOptionalAction, default=True,
                   help="play every net vs the stauf gauge (default on)")
    a.add_argument("--mm", default="7",
                   help="comma-separated MM anchor depths, '' for none (default 7)")
    a.add_argument("--vs", default="neighbors", choices=["neighbors", "net", "all"])
    a.add_argument("--games", type=int, default=12, help="target games per pair (default 12)")
    a.add_argument("--sims", type=int, default=edb.DEFAULT_CONFIG["sims"])
    a.add_argument("--pool-size", type=int, default=_env_int("T7G_EVAL_POOL", 64),
                   help="concurrent games batched into each forward pass "
                        "(default 64, or T7G_EVAL_POOL).  Extra concurrency is "
                        "near-free in wall-clock; the cap is arena memory -- "
                        "each slot holds one MCGS per net it plays")
    a.add_argument("--num-actions", type=int, default=1225)
    a.set_defaults(func=cmd_add)

    f = sub.add_parser("fit", help="fit ratings (WHR / BT) and write a snapshot")
    f.add_argument("--config", default=None, help="config_hash (default: from --sims)")
    f.add_argument("--sims", type=int, default=edb.DEFAULT_CONFIG["sims"])
    f.add_argument("--w", type=float, default=None,
                   help="Wiener drift Elo/sqrt(iter); omit for independent BT")
    f.add_argument("--anchor", action="append",
                   help="pin a player, e.g. MM7=1235 (repeatable). "
                        "Default: players with fixed_elo in players.json")
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("curve", help="emit Elo-vs-iteration for one run")
    c.add_argument("run")
    c.add_argument("--config", default=None)
    c.add_argument("--sims", type=int, default=edb.DEFAULT_CONFIG["sims"])
    c.add_argument("--w", type=float, default=None)
    c.set_defaults(func=cmd_curve)

    p = sub.add_parser("pin", help="freeze/unpin fixed_elo anchors in the registry")
    p.add_argument("assignments", nargs="+",
                   help="PLAYER=ELO to pin (e.g. MM7=1418), or PLAYER= to unpin. "
                        "Use once MM depths have been derived on the stauf gauge.")
    p.set_defaults(func=cmd_pin)

    lad = sub.add_parser("ladder",
                         help="deterministic low-end ladder: minimax depths vs the "
                              "original Stauf AI (engine-vs-engine, random openings)")
    lad.add_argument("--mm", default="2,3,4,5,6,7",
                     help="comma-separated minimax depths (default 2..7)")
    lad.add_argument("--stauf-depth", type=int, default=edb.STAUF_DEPTH,
                     help=f"canonical Stauf depth (default {edb.STAUF_DEPTH})")
    lad.add_argument("--games", type=int, default=60,
                     help="target games per pair, split by colour (default 60)")
    lad.add_argument("--opening-plies", type=int, default=4,
                     help="random opening plies for game variety (default 4)")
    lad.add_argument("--workers", type=int, default=6)
    lad.set_defaults(func=cmd_ladder)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
