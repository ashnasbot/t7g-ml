"""
Batched eval tournament pool (lib/train_workers.tournament_pool).

The pool replaced a multiprocessing-per-game eval path.  Its job is to be a
pure THROUGHPUT change: per-game rules and search semantics must stay identical
to play_eval_game / play_net_vs_net_game, because the Elo anchor pool was rated
under those.  These tests pin the game state machine, the Blue-perspective
result convention, per-net batch separation, and that every queued game is
reported exactly once.
"""
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")
from lib.net2 import Net2
from lib.evaluation import evaluate_vs_noisy_minimax
from lib.mcgs import MCGS
from lib.t7g import CLOCK_LIMIT, new_board
from lib.train_workers import (
    _TourneySlot, _t_apply, _t_engine_move, _t_result, _t_settle,
    play_eval_game, tournament_pool,
)

N_SIMS = 8


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    n = Net2(channels=32, num_blocks=2).to("cpu")
    n.eval()
    return n


def _slot(board=None, turn=True, clock=0):
    s = _TourneySlot()
    s.board = new_board() if board is None else board
    s.turn = turn
    s.clock = clock
    return s


# --- state machine --------------------------------------------------------

def test_settle_passes_through_open_position():
    """A normal start position needs a move: settle must not end or skip it."""
    s = _slot()
    assert _t_settle(s) is False
    assert s.turn is True and s.clock == 0 and s.move_count == 0


def test_settle_detects_terminal_and_signs_result_for_blue():
    """Green wiped out => Blue-perspective result is +1 regardless of mover."""
    board = new_board()
    board[:, :, 0] = False          # remove every Green piece
    board[0, 0, 1] = True           # ensure Blue has material
    for turn in (True, False):
        s = _slot(board=board.copy(), turn=turn)
        assert _t_settle(s) is True
        assert s.end_reason == "terminal"
        assert s.result == pytest.approx(1.0)
        assert _t_result(s)["blue_result"] == pytest.approx(1.0)
        assert _t_result(s)["blue_margin"] > 0


def test_settle_clock_expiry_is_a_draw():
    s = _slot(clock=CLOCK_LIMIT)
    assert _t_settle(s) is True
    assert s.end_reason == "clock"
    assert s.result == 0.0


def _legal(slot):
    from lib.t7g import action_masks
    return action_masks(slot.board, slot.turn)


def _blue_must_pass_board():
    """Blue owns one corner piece walled in by Green; empty cells exist but only
    Green can reach them, so Blue must pass and the game is not over."""
    board = new_board()
    board[:, :, :] = False
    board[0:3, :, 0] = True      # Green fills rows 0-2 (Blue's whole reach)
    board[0, 0, 0] = False
    board[0, 0, 1] = True        # ...except the corner, which is Blue's
    return board


def test_settle_pass_flips_turn_and_ticks_clock_without_a_move():
    """A side with no legal moves passes: turn flips, clock ticks, move_count
    does NOT advance (the convention play_eval_game uses)."""
    s = _slot(board=_blue_must_pass_board(), turn=True)
    assert not np.any(_legal(s)), "test board should leave Blue with no move"
    assert _t_settle(s) is False       # Green can still move: game continues
    assert s.turn is False             # ...as Green, after Blue's pass
    assert s.clock == 1
    assert s.move_count == 0


def test_apply_truncates_past_the_move_cap():
    s = _slot()
    s.move_count = 200
    legal = int(np.where(_legal(s))[0][0])
    assert _t_apply(s, legal) is True
    assert s.end_reason == "truncated"
    assert -1.0 <= s.result <= 1.0


def test_engine_move_with_full_noise_plays_a_legal_move():
    s = _slot()
    agent = ("engine", 1, 1.0, "micro3", False)
    assert _t_engine_move(s, agent) is False
    assert s.move_count == 1


# --- pool bookkeeping -----------------------------------------------------

def _games_vs_engine(n, depth=1, noise=0.0):
    opp = ("engine", depth, noise, "micro3", False)
    me = ("net", "cur")
    out = []
    for i in range(n):
        blue, green = (me, opp) if i % 2 == 0 else (opp, me)
        out.append((i, blue, green, True))
    return out


def test_every_queued_game_is_reported_exactly_once(net):
    games = _games_vs_engine(7)
    results = list(tournament_pool({"cur": net}, games,
                                   dict(num_simulations=N_SIMS, gumbel_k=4),
                                   pool_size=3))
    assert sorted(r["tag"] for r in results) == list(range(7))
    for r in results:
        assert -1.0 <= r["blue_result"] <= 1.0
        assert r["moves"] > 0
        assert r["end_reason"] in ("terminal", "clock", "truncated")


def test_pool_smaller_than_queue_recycles_slots(net):
    """One slot must chew through the whole queue (the _refill path)."""
    games = _games_vs_engine(4)
    results = list(tournament_pool({"cur": net}, games,
                                   dict(num_simulations=N_SIMS, gumbel_k=4),
                                   pool_size=1))
    assert len(results) == 4


def test_engine_vs_engine_game_needs_no_net(net):
    """A game with no net player still completes - it never starts a search."""
    mm1 = ("engine", 1, 0.0, "micro3", False)
    mm2 = ("engine", 2, 0.0, "micro3", False)
    results = list(tournament_pool({}, [("eve", mm1, mm2, True)], {}, pool_size=4))
    assert len(results) == 1
    assert results[0]["tag"] == "eve"


def test_net_vs_net_keeps_each_net_in_its_own_batch(net):
    """Every search in a forward pass must belong to the net doing the pass.

    Mixing them would silently evaluate one player's leaves with the other's
    weights - the failure mode the per-net grouping in _launch exists to
    prevent, and one that would not raise.
    """
    torch.manual_seed(1)
    other = Net2(channels=32, num_blocks=2).to("cpu")
    other.eval()

    owner: dict = {}
    orig_start = MCGS.start_search
    orig_launch = MCGS._launch_forward
    violations = []

    def start_search(self, *a, **kw):
        ss = orig_start(self, *a, **kw)
        owner[id(ss)] = id(self.network)
        return ss

    def launch(self, searches):
        for ss in searches:
            if owner.get(id(ss)) not in (None, id(self.network)):
                violations.append((id(ss), id(self.network)))
        return orig_launch(self, searches)

    MCGS.start_search = start_search
    MCGS._launch_forward = launch
    try:
        games = [(g, ("net", "a"), ("net", "b"), True) for g in range(4)]
        results = list(tournament_pool(
            {"a": net, "b": other}, games,
            dict(num_simulations=N_SIMS, gumbel_k=4), pool_size=4))
    finally:
        MCGS.start_search = orig_start
        MCGS._launch_forward = orig_launch

    assert not violations, f"searches evaluated by the wrong network: {violations}"
    assert len(results) == 4


def test_threaded_solver_fleet_matches_the_inline_solver():
    """Each thread must search on its OWN dlopen'd copy of the solver.

    One shared copy is not reentrant (bb_core.h holds the TT and history
    heuristic in globals), and the corruption would be silent - so pin that a
    fleet call returns exactly what the inline solver returns.
    """
    from lib.t7g import find_best_move
    from lib.train_workers import _get_solver_fleet
    board = new_board()
    depth = 5
    expected = find_best_move(board.tobytes(), depth, True, "micro3")
    fleet = _get_solver_fleet("micro3", 4)
    assert len(fleet) == 4
    for fn in fleet:
        assert fn(board.tobytes(), depth, True) == expected


def test_deep_engine_moves_go_through_the_fleet(net):
    """Depth >= _ENGINE_MIN_DEPTH_THREADED is threaded; games must come out the
    same shape as on the inline path."""
    from lib.train_workers import _ENGINE_MIN_DEPTH_THREADED
    games = _games_vs_engine(2, depth=_ENGINE_MIN_DEPTH_THREADED)
    mcts_kw = dict(num_simulations=N_SIMS, gumbel_k=4)
    threaded = list(tournament_pool({"cur": net}, games, mcts_kw,
                                    pool_size=2, engine_threads=2))
    inline = list(tournament_pool({"cur": net}, games, mcts_kw,
                                  pool_size=2, engine_threads=0))
    assert len(threaded) == len(inline) == 2
    for r in threaded + inline:
        assert r["moves"] > 0
        assert -1.0 <= r["blue_result"] <= 1.0


# --- agreement with the single-game path ----------------------------------

def test_pooled_and_single_game_paths_agree_on_score(net):
    """Same net, same opponent, both paths: scores must agree within noise.

    Not bit-identical (the C search draws its own Gumbel noise), so this is a
    coarse guard against perspective flips and colour mix-ups, which would show
    up as a mirrored score rather than a small deviation.
    """
    n_games = 12
    mcts_kw = dict(num_simulations=N_SIMS, gumbel_k=4)

    single = []
    for i in range(n_games):
        mcts = MCGS(net, **mcts_kw)
        result, _, _, _ = play_eval_game(mcts, 1, 0.0, "micro3", False, i % 2 == 0)
        single.append(result)

    pooled = []
    for r in tournament_pool({"cur": net}, _games_vs_engine(n_games), mcts_kw,
                             pool_size=4):
        pooled.append(r["blue_result"] if r["tag"] % 2 == 0 else -r["blue_result"])

    s_single = np.mean([(x + 1) / 2 for x in single])
    s_pooled = np.mean([(x + 1) / 2 for x in pooled])
    assert abs(s_single - s_pooled) < 0.45, (s_single, s_pooled)


def test_evaluate_vs_noisy_minimax_splits_colours(net):
    """The public wrapper must still balance colours and count every game."""
    wr, res = evaluate_vs_noisy_minimax(
        net, minimax_depth=1, noise=1.0, num_games=8, num_simulations=N_SIMS,
        engine='micro3', mcts_kwargs=dict(gumbel_k=4), pool_size=4,
    )
    assert res["wins"] + res["losses"] + res["draws"] == 8
    assert wr == pytest.approx(res["wins"] / 8)
    assert res["n_terminal"] + res["n_clock"] + res["n_truncated"] == 8
    for k in ("wr_as_blue", "wr_as_green"):
        assert 0.0 <= res[k] <= 1.0
