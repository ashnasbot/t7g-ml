"""
Evaluation functions for AlphaZero MCTS training.

Every routine here plays its games through lib.train_workers.tournament_pool -
one process, all games concurrent, inference batched across games.  The older
multiprocessing-per-game path was ~2.5k sim/s against self-play's 80k+ (a
500-sim search issues ~147 forwards of ~3 leaves, so it was pure launch
latency) and made eval ~38% of run wall-clock; batching removes that without
changing per-game search semantics.
"""
import math

import numpy as np
from tqdm import tqdm

from lib.train_workers import tournament_pool


# ---------------------------------------------------------------------------
# Evaluation vs minimax
# ---------------------------------------------------------------------------

def evaluate_vs_noisy_minimax(
    network,
    minimax_depth: int = 2,
    noise: float = 0.3,
    num_games: int = 20,
    num_simulations: int = 100,
    engine: str = 'minimax',
    vary_depth: bool = False,
    mcts_kwargs: dict | None = None,
    pool_size: int = 32,
) -> tuple[float, dict]:
    """
    Evaluate MCTS agent against a minimax opponent.

    Half the games are played as Blue, half as Green.
    Returns (win_rate, {wins, losses, draws, wr_as_blue, wr_as_green, ...}).
    """
    network.eval()
    mcts_kw = dict(mcts_kwargs or {}, num_simulations=num_simulations)
    opponent = ("engine", minimax_depth, noise, engine, vary_depth)
    subject  = ("net", "cur")
    # Blue always moves first from the standard start (play_eval_game's rule);
    # colour balance comes from alternating which side the subject plays.
    games = [
        ((game_idx % 2 == 0),
         *((subject, opponent) if game_idx % 2 == 0 else (opponent, subject)),
         True)
        for game_idx in range(num_games)
    ]
    engine_label = (
        "Stauf" if engine == 'stauf' else
        engine if engine in ("autaxx", "autaxx-ab", "tiktaxx", "scarlettxx") else
        f"MM-{minimax_depth}"
    )
    wins = losses = draws = 0
    wins_b = losses_b = wins_g = losses_g = 0
    n_terminal = n_clock = n_truncated = 0
    pbar = tqdm(total=num_games, desc=f"Eval vs {engine_label} (noise={noise:.0%})",
                unit="game", leave=False)
    try:
        for r in tournament_pool({"cur": network}, games, mcts_kw,
                                 pool_size=pool_size):
            is_blue = r["tag"]
            result = r["blue_result"] if is_blue else -r["blue_result"]
            end_reason = r["end_reason"]
            if result > 0:
                wins += 1
                if is_blue:
                    wins_b += 1
                else:
                    wins_g += 1
            elif result < 0:
                losses += 1
                if is_blue:
                    losses_b += 1
                else:
                    losses_g += 1
            else:
                draws += 1
            if end_reason == "clock":
                n_clock += 1
            elif end_reason == "truncated":
                n_truncated += 1
            else:
                n_terminal += 1
            pbar.update(1)
            pbar.set_postfix(win_rate=f"{wins / (wins + losses + draws):.0%}")
    finally:
        pbar.close()
    games_b = num_games // 2
    games_g = num_games - games_b
    return wins / num_games, {
        "wins": wins, "losses": losses, "draws": draws,
        "wr_as_blue":   wins_b / games_b if games_b else 0.0,
        "wr_as_green":  wins_g / games_g if games_g else 0.0,
        "n_terminal":  n_terminal,
        "n_clock":     n_clock,
        "n_truncated": n_truncated,
    }


# ---------------------------------------------------------------------------
# Elo rating vs a fixed anchor pool
# ---------------------------------------------------------------------------

def _anchor_agents(pool: list, num_actions: int, device=None) -> tuple[dict, list]:
    """Build {net key: module} and the per-member agent spec for an Elo pool.

    Net anchors arrive as state_dicts (the pool file stores weights, not live
    modules) and are instantiated once per call.
    """
    from lib.device_utils import get_device, load_compiled_network
    dev = device if device is not None else get_device()
    nets: dict = {}
    agents: list = []
    for i, m in enumerate(pool):
        if m["kind"] == "net":
            net, _ = load_compiled_network(m["payload"], dev,
                                           num_actions=num_actions, compile_net=False)
            net.eval()
            key = ("anchor", i)
            nets[key] = net
            agents.append(("net", key))
        else:
            # Elo anchors are played at zero noise, fixed depth, micro3 - the
            # config the fixed anchor ratings were fitted under.
            agents.append(("engine", m["payload"], 0.0, "micro3", False))
    return nets, agents


def rate_vs_pool(
    network,
    pool: list,
    games_per_opponent: int = 8,
    num_actions: int = 1225,
    mcts_kwargs: dict | None = None,
    pool_size: int = 32,
    virtual_draws: float = 1.0,
) -> tuple[float, dict]:
    """
    Rate *network* on the Elo scale by playing a fixed, pre-rated anchor pool.

    Anchor ratings are held fixed (they come from a full round-robin fit by
    scripts/rate_checkpoints.py), so only the current net's rating is unknown.
    Its maximum-likelihood value solves  sum_i score_i = sum_i E_i(r)  where
    E_i(r) = n_i / (1 + 10^((elo_i - r)/400)) - the expected score against
    anchor i.  That sum is monotone in r, so bisection finds it.  Draws count
    0.5; *virtual_draws* pseudo-games per opponent keep a clean sweep finite
    (a 100% score has infinite MLE).

    pool: [{"name": str, "kind": "net"|"mm", "payload": state_dict|depth,
            "elo": float}, ...]
    Returns (elo, ci95, {name: [wins, draws, losses]}), where ci95 is the
    95% CI half-width (1.96 / sqrt(observed information)), same convention
    as lib/eval_db.whr_ci95 - treats each anchor's Elo as fixed/known, so
    this understates uncertainty (no anchor-rating error propagated), but
    it's directly comparable across iterations of the same pool.
    """
    network.eval()
    mcts_kw = dict(mcts_kwargs or {})
    anchor_nets, anchor_agents = _anchor_agents(pool, num_actions)
    subject = ("net", "cur")
    games = []
    for i, agent in enumerate(anchor_agents):
        for g in range(games_per_opponent):
            cur_is_blue = g % 2 == 0
            blue, green = (subject, agent) if cur_is_blue else (agent, subject)
            # Net-vs-net games randomise who moves first (play_net_vs_net_game);
            # games against an engine anchor always start with Blue to move
            # (play_eval_game).  Both are part of what the anchors were rated
            # under, so keep them distinct.
            first_turn = bool(np.random.randint(2)) if agent[0] == "net" else True
            games.append(((i, cur_is_blue), blue, green, first_turn))
    results: dict[str, list] = {m["name"]: [0, 0, 0] for m in pool}
    # Game-shape aggregates: how the rating was earned, not just the score.
    # A dominant sweep (big margins, short decisive games) and a marginal one
    # (razor-thin wins) can land the same Elo -- these tell them apart.
    win_margins: list = []
    loss_margins: list = []
    draw_margins: list = []
    all_moves: list = []
    pbar = tqdm(total=len(games), desc="Elo vs pool", unit="game", leave=False)
    try:
        for r in tournament_pool({"cur": network, **anchor_nets}, games, mcts_kw,
                                 pool_size=pool_size):
            opp_idx, cur_is_blue = r["tag"]
            result = r["blue_result"] if cur_is_blue else -r["blue_result"]
            margin = r["blue_margin"] if cur_is_blue else -r["blue_margin"]
            slot = 0 if result > 0 else (2 if result < 0 else 1)
            results[pool[opp_idx]["name"]][slot] += 1
            (win_margins if result > 0 else
             loss_margins if result < 0 else draw_margins).append(margin)
            all_moves.append(r["moves"])
            pbar.update(1)
    finally:
        pbar.close()

    score = games = 0.0
    elos  = []
    for m in pool:
        w, d, _ = results[m["name"]]
        score += w + 0.5 * d + 0.5 * virtual_draws
        games += games_per_opponent + virtual_draws
        elos.append(m["elo"])

    def expected(r: float) -> float:
        return sum((games_per_opponent + virtual_draws)
                   / (1.0 + 10.0 ** ((e - r) / 400.0)) for e in elos)

    lo, hi = min(elos) - 1000.0, max(elos) + 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if expected(mid) < score:
            lo = mid
        else:
            hi = mid
    elo = (lo + hi) / 2.0

    # Observed information at the MLE - same Bradley-Terry Fisher-info form
    # as lib/eval_db.whr_ci95, specialised to a single unknown rating.
    c = math.log(10.0) / 400.0
    n_per = games_per_opponent + virtual_draws
    hess = 0.0
    for e in elos:
        p = 1.0 / (1.0 + 10.0 ** ((e - elo) / 400.0))
        hess += n_per * c * c * p * (1.0 - p)
    ci95 = 1.96 / math.sqrt(hess) if hess > 0 else float("inf")

    _med = lambda xs: float(np.median(xs)) if xs else 0.0
    shape = {
        "moves_med":          _med(all_moves),
        "win_margin_med":     _med(win_margins),
        "loss_margin_med":    _med(loss_margins),
        "draw_margin_absmed": _med([abs(m) for m in draw_margins]),
        "n_win":  len(win_margins),
        "n_loss": len(loss_margins),
        "n_draw": len(draw_margins),
    }
    return elo, ci95, results, shape


# ---------------------------------------------------------------------------
# Ladder calibration (resume only)
# ---------------------------------------------------------------------------

def _calibrate_ladder(
    network,
    mcts_kwargs: dict,
    eval_ladder: list,
    eval_simulations: int,
    pool_size: int = 32,
) -> tuple[int, float]:
    """Quickly find the right ladder starting point on resume.

    Nets resumed from a checkpoint typically clear the whole ladder, so check
    the *top* rung first: a 100% sweep there implies (by difficulty ordering)
    every lower rung is cleared too - one eval instead of scanning all rungs.
    Only when the top rung isn't a clean sweep do we fall back to a bottom-up
    scan to locate the first rung the net can't beat.  Advances only on 100%.
    Returns (eval_level, wr_at_that_level).
    """
    def _eval_rung(level: int) -> float:
        depth, noise, label = eval_ladder[level]
        wr, _ = evaluate_vs_noisy_minimax(
            network, minimax_depth=depth, noise=noise,
            num_games=10, num_simulations=eval_simulations,
            mcts_kwargs=mcts_kwargs, engine='micro3', pool_size=pool_size,
        )
        print(f"  Calibrate  {label}  {wr:.0%}")
        return wr

    print("Calibrating ladder position (10 games/rung, advance at 100%)...")
    # Top-rung short-circuit: sweeping the hardest rung clears the ladder in a
    # single eval - the common case for a resumed checkpoint.
    if _eval_rung(len(eval_ladder) - 1) >= 1.0:
        print("  Calibrate  all rungs cleared!")
        return len(eval_ladder), 0.0
    # Weak resume: scan bottom-up for the first rung the net can't sweep.
    for level in range(len(eval_ladder)):
        wr = _eval_rung(level)
        if wr < 1.0:
            return level, wr
    print("  Calibrate  all rungs cleared!")
    return len(eval_ladder), 0.0
