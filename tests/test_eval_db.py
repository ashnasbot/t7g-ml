"""Unit tests for the offline evaluation DB + rating fitters (lib/eval_db.py).

These are pure-logic tests (no torch / no game-play): synthetic match graphs
with analytic Elo, anchor pinning, config_hash isolation, jsonl dedup, and the
w -> inf == static-BT validation from the design doc.
"""
import math

import numpy as np
import pytest

from lib import eval_db as edb


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

def test_config_hash_stable_and_numeric_normalised():
    h1 = edb.config_hash({"sims": 500, "c_puct": 1.3})
    h2 = edb.config_hash({"sims": 500.0, "c_puct": 1.30})   # 500 == 500.0
    h3 = edb.config_hash({})                                # all defaults
    assert h1 == h2 == h3                                   # DEFAULT sims is 500
    assert isinstance(h1, str) and len(h1) == 8


def test_config_hash_isolation():
    assert edb.config_hash({"sims": 500}) != edb.config_hash({"sims": 100})
    assert edb.config_hash({"sigma_scale": 1.0}) != edb.config_hash({"sigma_scale": 0.25})


# ---------------------------------------------------------------------------
# match store: append / load / counts / dedup
# ---------------------------------------------------------------------------

def test_append_load_and_config_filter(tmp_path):
    p = str(tmp_path / "matches.jsonl")
    edb.append_matches([
        {"a": "X", "b": "Y", "a_is_blue": True, "result": 1, "config_hash": "aaa"},
        {"a": "X", "b": "Y", "a_is_blue": False, "result": -1, "config_hash": "bbb"},
    ], path=p)
    assert len(edb.load_matches(path=p)) == 2
    assert len(edb.load_matches("aaa", path=p)) == 1
    # ts auto-filled
    assert all("ts" in r for r in edb.load_matches(path=p))


def test_counts_fold_colour_and_orientation(tmp_path):
    p = str(tmp_path / "m.jsonl")
    # Two games: X beats Y once (a=X), Y beats X once (a=Y, result=+1 => Y won).
    edb.append_matches([
        {"a": "X", "b": "Y", "a_is_blue": True, "result": 1, "config_hash": "h"},
        {"a": "Y", "b": "X", "a_is_blue": True, "result": 1, "config_hash": "h"},
    ], path=p)
    names, counts = edb.load_counts("h", path=p)
    assert names == ["X", "Y"]
    # counts keyed on i<j => (0,1); X won once, Y won once => [1 win_X, 0 draw, 1 win_Y]
    assert counts[(0, 1)] == [1, 0, 1]


def test_pair_game_count_dedup(tmp_path):
    p = str(tmp_path / "m.jsonl")
    edb.append_matches([
        {"a": "X", "b": "Y", "a_is_blue": True, "result": 1, "config_hash": "h"},
        {"a": "Y", "b": "X", "a_is_blue": False, "result": 0, "config_hash": "h"},
        {"a": "X", "b": "Z", "a_is_blue": True, "result": -1, "config_hash": "h"},
        {"a": "X", "b": "Y", "a_is_blue": True, "result": 1, "config_hash": "other"},
    ], path=p)
    assert edb.pair_game_count("h", "X", "Y", path=p) == 2   # colour/order agnostic
    assert edb.pair_game_count("h", "X", "Z", path=p) == 1
    assert edb.pair_game_count("other", "X", "Y", path=p) == 1


# ---------------------------------------------------------------------------
# player registry
# ---------------------------------------------------------------------------

def test_register_player_roundtrip_and_merge(tmp_path):
    p = str(tmp_path / "players.json")
    edb.register_player("run/iter_0010", {"kind": "net", "run": "run", "iteration": 10}, path=p)
    edb.register_player("run/iter_0010", {"arch": "v2"}, path=p)   # merge, not clobber
    reg = edb.load_players(p)
    assert reg["run/iter_0010"]["kind"] == "net"
    assert reg["run/iter_0010"]["arch"] == "v2"
    assert reg["run/iter_0010"]["iteration"] == 10


def test_set_fixed_elo_pin_and_unpin(tmp_path):
    p = str(tmp_path / "players.json")
    edb.register_player("MM7", {"kind": "mm", "depth": 7}, path=p)
    edb.set_fixed_elo("MM7", 1418.0, path=p)          # freeze a derived anchor
    assert edb.load_players(p)["MM7"]["fixed_elo"] == 1418.0
    edb.set_fixed_elo("MM7", None, path=p)            # unpin -> float again
    assert "fixed_elo" not in edb.load_players(p)["MM7"]
    # stauf gauge default
    assert edb.STAUF_ANCHOR_ELO == 1000.0 and edb.STAUF_PLAYER_ID == "stauf"


def test_stauf_sole_gauge_anchoring():
    # stauf pinned at 1000 as the ONLY fixed point; a net that beats stauf ~85%
    # and an MM that beats stauf ~62% both float ABOVE it, MM below the net.
    names = ["stauf", "MM7", "net"]
    counts = {
        (0, 1): [76, 0, 124],   # stauf 76 / MM7 124  -> MM7 above stauf
        (0, 2): [30, 0, 170],   # stauf 30 / net 170  -> net well above stauf
        (1, 2): [60, 0, 140],   # MM7 60 / net 140
    }
    elo, _ = edb.fit_whr(names, counts, anchors={"stauf": 1000.0}, virtual_draws=2.0)
    assert elo[0] == pytest.approx(1000.0, abs=1e-6)   # stauf pinned exactly
    assert elo[2] > elo[1] > elo[0]                    # net > MM7 > stauf, all floating


# ---------------------------------------------------------------------------
# Bradley-Terry analytic sanity
# ---------------------------------------------------------------------------

def test_bt_two_player_analytic_elo():
    # A scores 75% over B  =>  Elo gap = 400*log10(0.75/0.25) = 400*log10(3).
    names = ["A", "B"]
    counts = {(0, 1): [750, 0, 250]}
    elo = edb.fit_bradley_terry(names, counts, virtual_draws=0.0)
    gap = elo[0] - elo[1]
    assert gap == pytest.approx(400 * math.log10(3), abs=3.0)


# ---------------------------------------------------------------------------
# WHR
# ---------------------------------------------------------------------------

def _three_player_counts():
    # Transitive ladder A > B > C, each pair 200 games at fixed win probs.
    # A beats B 70%, B beats C 70%, A beats C ~85% (roughly transitive).
    return ["A", "B", "C"], {
        (0, 1): [140, 0, 60],
        (1, 2): [140, 0, 60],
        (0, 2): [170, 0, 30],
    }


def test_whr_no_prior_matches_bt():
    names, counts = _three_player_counts()
    bt = edb.fit_bradley_terry(names, counts, virtual_draws=2.0)
    whr, _ = edb.fit_whr(names, counts, w=None, virtual_draws=2.0)   # no prior
    # Both mean-normalised; compare after removing the gauge.
    bt = bt - bt.mean()
    whr = whr - whr.mean()
    assert np.allclose(bt, whr, atol=2.0)


def test_whr_w_infinite_equals_bt():
    names, counts = _three_player_counts()
    chains = [[0, 1, 2]]
    bt = edb.fit_bradley_terry(names, counts, virtual_draws=2.0)
    whr, _ = edb.fit_whr(names, counts, chains=chains, w=float("inf"),
                         virtual_draws=2.0)
    bt = bt - bt.mean()
    whr = whr - whr.mean()
    assert np.allclose(bt, whr, atol=2.0)


def test_whr_anchor_pinning():
    names, counts = _three_player_counts()
    whr, _ = edb.fit_whr(names, counts, anchors={"C": 1000.0}, virtual_draws=2.0)
    assert whr[2] == pytest.approx(1000.0, abs=1e-6)   # C pinned exactly
    assert whr[0] > whr[1] > whr[2]                    # ladder preserved


def test_whr_prior_shrinks_undersampled_checkpoint():
    # A(0) and C(2) are well-sampled and far apart; B(1) has ZERO games.
    # Without a prior B is unidentifiable; the Wiener chain A-B-C pulls B to
    # roughly the midpoint of its neighbours instead of leaving it at 0.
    names = ["A", "B", "C"]
    counts = {(0, 2): [100, 0, 100]}          # A == C, only these two play
    chains = [[0, 1, 2]]
    whr, _ = edb.fit_whr(names, counts, chains=chains, w=30.0,
                         anchors={"A": 1000.0, "C": 1000.0}, virtual_draws=2.0)
    assert whr[1] == pytest.approx(1000.0, abs=5.0)   # pulled to neighbours' level


def test_whr_monotone_on_improving_ladder():
    # Synthetic improving run: 5 checkpoints, each ~+80 Elo over the previous,
    # sampled only against temporal neighbours.  WHR should recover a monotone
    # increasing curve.
    true_elo = np.array([1000.0, 1080.0, 1160.0, 1240.0, 1320.0])
    names = [f"iter{k}" for k in range(5)]
    rng = np.random.default_rng(0)
    counts = {}
    for i in range(4):
        j = i + 1
        p = 1.0 / (1.0 + 10 ** ((true_elo[i] - true_elo[j]) / 400.0))  # P(j beats i)
        n = 400
        wj = int(rng.binomial(n, p))
        counts[(i, j)] = [n - wj, 0, wj]     # [wins_i, draw, wins_j]
    chains = [[0, 1, 2, 3, 4]]
    whr, _ = edb.fit_whr(names, counts, chains=chains, w=60.0,
                         anchors={"iter0": 1000.0}, virtual_draws=2.0)
    assert np.all(np.diff(whr) > 0), whr


def test_whr_ci_from_hessian_positive():
    names, counts = _three_player_counts()
    _, hess = edb.fit_whr(names, counts, virtual_draws=2.0)
    ci = edb.whr_ci95(hess)
    assert np.all(ci > 0) and np.all(np.isfinite(ci))


def test_reanchor():
    names = ["A", "B", "C"]
    elo = np.array([50.0, 0.0, -100.0])
    out = edb.reanchor(names, elo, "C", 1000.0)
    assert out[2] == pytest.approx(1000.0)
    assert out[0] == pytest.approx(1150.0)


# ---------------------------------------------------------------------------
# scheduler (scripts/eval_db.py) -- stauf + MM anchors, temporal chains
# ---------------------------------------------------------------------------

def test_schedule_pairs_stauf_and_neighbors():
    import scripts.eval_db as cli
    net_ids = ["r/iter_0010", "r/iter_0020", "r/iter_0030"]
    net_meta = {
        "r/iter_0010": {"run": "r", "iteration": 10},
        "r/iter_0020": {"run": "r", "iteration": 20},
        "r/iter_0030": {"run": "r", "iteration": 30},
    }
    anchor_ids = ["stauf", "MM7"]
    pairs = cli.schedule_pairs(net_ids, net_meta, anchor_ids, vs="neighbors")
    # every net vs every anchor
    for n in net_ids:
        assert frozenset((n, "stauf")) in pairs
        assert frozenset((n, "MM7")) in pairs
    # temporal neighbours only (10-20, 20-30), NOT the 10-30 long bond
    assert frozenset(("r/iter_0010", "r/iter_0020")) in pairs
    assert frozenset(("r/iter_0020", "r/iter_0030")) in pairs
    assert frozenset(("r/iter_0010", "r/iter_0030")) not in pairs
    # anchors never play each other
    assert frozenset(("stauf", "MM7")) not in pairs


def test_schedule_pairs_net_roundrobin():
    import scripts.eval_db as cli
    net_ids = ["r/iter_0010", "r/iter_0030"]
    net_meta = {"r/iter_0010": {"run": "r", "iteration": 10},
                "r/iter_0030": {"run": "r", "iteration": 30}}
    pairs = cli.schedule_pairs(net_ids, net_meta, ["stauf"], vs="net")
    assert frozenset(("r/iter_0010", "r/iter_0030")) in pairs   # full round-robin


# ---------------------------------------------------------------------------
# deterministic engine-vs-engine driver (low-end ladder)
# ---------------------------------------------------------------------------

def test_eve_config_hash_isolated_from_net_config():
    h = edb.eve_config_hash(stauf_depth=6, opening_plies=4)
    assert h.startswith("eve")
    assert h != edb.config_hash({})                          # never collides with net games
    assert h != edb.eve_config_hash(stauf_depth=6, opening_plies=2)   # opening plies matter
    assert edb.STAUF_DEPTH == 6


def test_play_engine_vs_engine_deterministic_and_transitive():
    from lib.train_workers import play_engine_vs_engine
    # Same opening seed + same specs + same colour => identical result (engines
    # are deterministic; only the random opening varies).
    r1 = play_engine_vs_engine(("micro3", 5), ("micro3", 2), True,
                               opening_plies=4, rng=np.random.default_rng(7))
    r2 = play_engine_vs_engine(("micro3", 5), ("micro3", 2), True,
                               opening_plies=4, rng=np.random.default_rng(7))
    assert r1 == r2 and r1 in (-1, 0, 1)
    # Strength ordering: MM5 should dominate MM2 across varied openings.
    rng = np.random.default_rng(0)
    score = sum((play_engine_vs_engine(("micro3", 5), ("micro3", 2), g % 2 == 0,
                                       opening_plies=4, rng=rng) + 1) / 2
                for g in range(20))
    assert score >= 15   # MM5 wins the large majority vs MM2
