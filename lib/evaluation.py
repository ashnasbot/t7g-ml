"""
Evaluation workers and functions for AlphaZero MCTS training.

Contains the multiprocessing worker globals, initializers, and evaluation
routines for playing vs minimax and vs the best network.
"""
import multiprocessing

import numpy as np
from tqdm import tqdm

from lib.mcgs import MCGS
from lib.train_workers import play_eval_game, play_net_vs_net_game


# ---------------------------------------------------------------------------
# Per-process globals (multiprocessing spawn requires module-level state)
# ---------------------------------------------------------------------------

_local_network        = None   # compiled model used for inference in workers
_base_network         = None   # uncompiled reference for in-place weight updates
_weight_queue         = None   # receives weight broadcasts from the main process
_worker_mcts_cls      = None   # MCGS subclass - set in _worker_init
_worker_mcts_kwargs: dict = {}


# ---------------------------------------------------------------------------
# Worker initializers
# ---------------------------------------------------------------------------

def _worker_init(state_dict, weight_queue=None, num_actions=1225, mcts_cls=None,
                 mcts_kwargs=None):
    """Initialise a self-play or evaluation worker process."""
    global _local_network, _base_network, _weight_queue, _worker_mcts_cls, _worker_mcts_kwargs
    import torch as _torch
    from lib.device_utils import get_device as _get_device
    try:
        device = _get_device()
    except Exception:
        device = _torch.device("cpu")
    if device.type == "cpu":
        _torch.set_num_threads(1)
    from lib.device_utils import load_compiled_network as _lcn
    _local_network, _base_network = _lcn(
        state_dict, device, num_actions=num_actions, compile_net=False,
    )
    _weight_queue = weight_queue
    _worker_mcts_cls    = mcts_cls if mcts_cls is not None else MCGS
    _worker_mcts_kwargs = mcts_kwargs or {}




# ---------------------------------------------------------------------------
# Worker game functions
# ---------------------------------------------------------------------------

def _worker_eval_game(args):
    """Evaluation worker: play one game vs minimax/stauf."""
    num_simulations, minimax_depth, noise, engine, vary_depth, mcts_is_blue = args
    _local_network.eval()  # type: ignore[union-attr]
    mcts = _worker_mcts_cls(
        _local_network,  # type: ignore[arg-type]
        num_simulations=num_simulations,
        **{k: v for k, v in _worker_mcts_kwargs.items() if k != 'num_simulations'},
    )
    result, end_reason = play_eval_game(
        mcts, minimax_depth, noise, engine, vary_depth, mcts_is_blue,
    )
    return result, mcts_is_blue, end_reason


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
    num_actions: int = 1225,
    mcts_cls=None,
    mcts_kwargs: dict | None = None,
    num_workers: int = 4,
) -> tuple[float, dict]:
    """
    Evaluate MCTS agent against a minimax opponent.

    Half the games are played as Blue, half as Green.
    Returns (win_rate, {wins, losses, draws, wr_as_blue, wr_as_green, ...}).
    """
    _mcts_cls    = mcts_cls or MCGS
    _mcts_kw     = mcts_kwargs or {}
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    task_args = [
        (num_simulations, minimax_depth, noise, engine, vary_depth, game_idx % 2 == 0)
        for game_idx in range(num_games)
    ]
    engine_label = "Stauf" if engine == 'stauf' else f"MM-{minimax_depth}"
    wins = losses = draws = 0
    wins_b = losses_b = wins_g = losses_g = 0
    n_terminal = n_repetition = n_truncated = 0
    pbar = tqdm(total=num_games, desc=f"Eval vs {engine_label} (noise={noise:.0%})",
                unit="game", leave=False)
    try:
        with multiprocessing.Pool(
            processes=min(num_workers, num_games),
            initializer=_worker_init,
            initargs=(state_dict, None, num_actions, _mcts_cls, _mcts_kw),
        ) as pool:
            for result, is_blue, end_reason in pool.imap_unordered(_worker_eval_game, task_args):
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
                if end_reason == "repetition":
                    n_repetition += 1
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
        "n_terminal":   n_terminal,
        "n_repetition": n_repetition,
        "n_truncated":  n_truncated,
    }


# ---------------------------------------------------------------------------
# Elo rating vs a fixed anchor pool
# ---------------------------------------------------------------------------

_pool_players: list = []   # [('net', network) | ('mm', depth), ...] per worker


def _worker_pool_init(cur_state_dict, pool_specs, num_actions=1225, mcts_kwargs=None):
    """Initialise a pool-rating worker: current net + every anchor net."""
    global _local_network, _pool_players, _worker_mcts_cls, _worker_mcts_kwargs
    import torch as _torch
    from lib.device_utils import get_device as _get_device
    try:
        device = _get_device()
    except Exception:
        device = _torch.device("cpu")
    if device.type == "cpu":
        _torch.set_num_threads(1)
    from lib.device_utils import load_compiled_network as _lcn
    _local_network, _ = _lcn(cur_state_dict, device,
                             num_actions=num_actions, compile_net=False)
    _pool_players = []
    for kind, payload in pool_specs:
        if kind == "net":
            net, _ = _lcn(payload, device, num_actions=num_actions, compile_net=False)
            _pool_players.append(("net", net))
        else:
            _pool_players.append(("mm", payload))
    _worker_mcts_cls    = MCGS
    _worker_mcts_kwargs = mcts_kwargs or {}


def _worker_pool_game(args):
    """Play one game: current network vs pool member opp_idx."""
    opp_idx, cur_is_blue = args
    _local_network.eval()  # type: ignore[union-attr]
    kind, payload = _pool_players[opp_idx]
    # Fresh MCGS per game so transposition tables never leak between games.
    mcts_cur = _worker_mcts_cls(_local_network, **_worker_mcts_kwargs)  # type: ignore[call-arg]
    if kind == "mm":
        result, _ = play_eval_game(mcts_cur, payload, 0.0, "micro3", False, cur_is_blue)
    else:
        payload.eval()
        mcts_opp = _worker_mcts_cls(payload, **_worker_mcts_kwargs)  # type: ignore[call-arg]
        result = play_net_vs_net_game(mcts_cur, mcts_opp, cur_is_blue)
    return opp_idx, result


def rate_vs_pool(
    network,
    pool: list,
    games_per_opponent: int = 8,
    num_actions: int = 1225,
    mcts_kwargs: dict | None = None,
    num_workers: int = 4,
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
    Returns (elo, {name: [wins, draws, losses]}).
    """
    cur_state = {k: v.cpu() for k, v in network.state_dict().items()}
    specs = [(m["kind"], m["payload"]) for m in pool]
    task_args = [(i, g % 2 == 0)
                 for i in range(len(pool)) for g in range(games_per_opponent)]
    results: dict[str, list] = {m["name"]: [0, 0, 0] for m in pool}
    pbar = tqdm(total=len(task_args), desc="Elo vs pool", unit="game", leave=False)
    try:
        # Explicit spawn context: the default on Python <=3.13 is fork, which
        # deadlocks when the parent holds torch/CUDA thread state (observed
        # locally on 3.12 - workers hang on futexes mid-tournament).
        with multiprocessing.get_context("spawn").Pool(
            processes=min(num_workers, len(task_args)),
            initializer=_worker_pool_init,
            initargs=(cur_state, specs, num_actions, mcts_kwargs or {}),
        ) as mp_pool:
            for opp_idx, result in mp_pool.imap_unordered(_worker_pool_game, task_args):
                slot = 0 if result > 0 else (2 if result < 0 else 1)
                results[pool[opp_idx]["name"]][slot] += 1
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
    return (lo + hi) / 2.0, results


# ---------------------------------------------------------------------------
# Ladder calibration (resume only)
# ---------------------------------------------------------------------------

def _calibrate_ladder(
    network,
    num_actions: int,
    mcts_cls,
    mcts_kwargs: dict,
    eval_ladder: list,
    eval_simulations: int,
    num_workers: int = 4,
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
            num_actions=num_actions, mcts_cls=mcts_cls, mcts_kwargs=mcts_kwargs,
            engine='micro3', num_workers=num_workers,
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
