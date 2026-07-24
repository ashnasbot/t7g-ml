"""Promotion-gate decision logic (lib.evaluation.gate_decision).

The gate is a relative head-to-head test vs the incumbent generator with
adaptive extension: clear results decide on one 16-game block, close calls
extend up to a cap.  These tests pin the decision zones so retuning the
z-values or margin is a conscious act.
"""
import math

import pytest

from lib.evaluation import gate_decision, h2h_gate, _wilson_bounds

# +25 Elo as a score margin, the value train_mcts.py derives from
# GATE_MARGIN_ELO.
S_MARGIN = 1.0 / (1.0 + 10.0 ** (-25.0 / 400.0))


def dec(w, d, l, **kw):
    return gate_decision(w, d, l, s_margin=S_MARGIN, **kw)


class TestWilson:
    def test_bounds_bracket_score(self):
        lo, hi = _wilson_bounds(0.6, 16, 1.282)
        assert lo < 0.6 < hi

    def test_sweep_is_finite(self):
        lo, hi = _wilson_bounds(1.0, 16, 1.282)
        assert 0.0 < lo < 1.0 and hi <= 1.0 + 1e-9
        assert math.isfinite(lo) and math.isfinite(hi)

    def test_tightens_with_n(self):
        lo16, _ = _wilson_bounds(0.6, 16, 1.282)
        lo96, _ = _wilson_bounds(0.6, 96, 1.282)
        assert lo96 > lo16


class TestFirstBlock:
    """Decisions available after the initial 16 games."""

    def test_clear_win_promotes(self):
        assert dec(11, 0, 5) == "promote"

    def test_sweep_promotes(self):
        assert dec(16, 0, 0) == "promote"

    def test_lucky_looking_win_extends(self):
        # 10-6 is the old-gate failure mode: ~23% likely from a truly equal
        # net.  Must NOT promote on one block.
        assert dec(10, 0, 6) == "extend"

    def test_clear_loss_retains(self):
        assert dec(5, 0, 11) == "retain"

    def test_mild_loss_extends(self):
        # 7-9 from a true +25 net is a plausible bad day: one more block.
        assert dec(7, 0, 9) == "extend"

    def test_even_record_extends(self):
        assert dec(8, 0, 8) == "extend"

    def test_all_draws_extends(self):
        assert dec(0, 16, 0) == "extend"


class TestExtension:
    """Behaviour once blocks have been added."""

    def test_equal_cut_stops_coin_flips(self):
        # Still 50% after 32 games: stop paying for it.
        assert dec(16, 0, 16) == "retain"
        assert dec(0, 32, 0) == "retain"

    def test_equal_cut_respects_threshold(self):
        assert dec(16, 0, 16, equal_cut_games=64) == "extend"

    def test_modest_edge_promotes_at_cap_scale(self):
        # ~57% over 96 games clears both the confidence bound and the margin.
        assert dec(55, 0, 41) == "promote"

    def test_margin_edge_still_ambiguous_at_cap(self):
        # ~55% (= +35 Elo point estimate) is not confidently > 0.5 even at 96
        # games -> "extend"; the caller converts exhaustion into retain.
        assert dec(53, 0, 43) == "extend"

    def test_draws_count_half(self):
        # 48-16-32 -> s = (48+8)/96 = 0.583, same zone as a 56-40 decisive
        # record.
        assert dec(48, 16, 32) == dec(56, 0, 40) == "promote"


class TestSeededGate:
    """h2h_gate must decide from a clear seed record (the pinned incumbent's
    gauntlet games) without spawning workers or touching the networks -
    passing None networks proves no games were played."""

    def test_clear_seed_promotes_free(self):
        decision, rec, s = h2h_gate(None, None, s_margin=S_MARGIN,
                                    seed_record=(11, 0, 5))
        assert decision == "promote" and rec == (11, 0, 5)
        assert s == pytest.approx(11 / 16)

    def test_clear_seed_retains_free(self):
        decision, rec, _ = h2h_gate(None, None, s_margin=S_MARGIN,
                                    seed_record=(5, 0, 11))
        assert decision == "retain" and rec == (5, 0, 11)

    def test_ambiguous_seed_at_cap_retains(self):
        # +35-Elo point estimate but not confidently > 0.5 even with the full
        # budget spent: exhaustion converts extend -> retain, no fresh games.
        decision, rec, _ = h2h_gate(None, None, s_margin=S_MARGIN,
                                    seed_record=(53, 0, 43), max_games=96)
        assert decision == "retain" and rec == (53, 0, 43)


class TestMonotonicity:
    def test_more_wins_never_hurts(self):
        order = {"retain": 0, "extend": 1, "promote": 2}
        prev = -1
        for w in range(17):
            cur = order[dec(w, 0, 16 - w)]
            assert cur >= prev
            prev = cur
