"""
Playout-cap randomization (PCR) contract tests.

Per MOVE, with probability pcr_p_full the search runs the full budget and the
example keeps its policy target; otherwise a cheap pcr_fast_sims search is run
and the example must train value/aux only: policy target zeroed (masked out of
the policy CE by lib/training.py's has_policy check) and the root-Q blend
weight dropped (a low-sim root is too noisy to teach the value head).

pcr_p_full=1.0 must be a strict no-op (the pre-PCR code path, no C calls).
"""
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, ".")
from lib.dual_network import DualHeadNetwork
from lib.mcgs import MCGS
from lib.train_workers import self_play_game_pool

N_SIMS_FULL = 8
N_SIMS_FAST = 4


@pytest.fixture(scope="module")
def mcts():
    torch.manual_seed(0)
    net = DualHeadNetwork(num_actions=1225).to("cpu")
    net.eval()
    return MCGS(net, num_simulations=N_SIMS_FULL, c_puct=1.25, gumbel_k=4)


def _collect(mcts, **pool_kwargs):
    examples = []
    for game_examples, *_ in self_play_game_pool(
        mcts, pool_size=4, target_games=12, **pool_kwargs
    ):
        examples.extend(game_examples)
    assert examples, "No training examples were generated"
    return examples


def test_set_num_simulations():
    """The C-side budget setter must be exported and steer the next search."""
    torch.manual_seed(0)
    net = DualHeadNetwork(num_actions=1225).to("cpu")
    net.eval()
    m = MCGS(net, num_simulations=N_SIMS_FULL, c_puct=1.25, gumbel_k=4)
    m.set_num_simulations(N_SIMS_FAST)
    assert m.num_simulations == N_SIMS_FAST
    from lib.t7g import new_board
    probs = m.search(new_board(), True)
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)


def test_pcr_disabled_all_rows_have_policy(mcts):
    """pcr_p_full=1.0: every recorded example keeps a normalized policy target."""
    np.random.seed(0)
    mcts.set_num_simulations(N_SIMS_FULL)
    for ex in _collect(mcts, pcr_p_full=1.0, pcr_fast_sims=N_SIMS_FAST):
        assert np.any(ex[1]), "zero policy target with PCR disabled"


def test_pcr_fast_rows_masked_and_unblended(mcts):
    """Fast rows: zero policy + zero q_weight (even with blending wide open);
    full rows keep normalized policies.  Split fraction ~ pcr_p_full."""
    np.random.seed(0)
    mcts.set_num_simulations(N_SIMS_FULL)
    examples = _collect(mcts, pcr_p_full=0.5, pcr_fast_sims=N_SIMS_FAST,
                        temp_moves=0, blend_alpha=0.5)
    # 9-tuple layout: 1=policy, 2=z, 7=root_q, 8=q_weight
    fast = [ex for ex in examples if not np.any(ex[1])]
    full = [ex for ex in examples if np.any(ex[1])]
    assert fast, "expected some fast rows at p_full=0.5"
    assert full, "expected some full rows at p_full=0.5"
    for ex in fast:
        assert float(ex[8]) == 0.0, "fast row leaked a nonzero q_weight"
        assert float(ex[7]) == 0.0, "fast row leaked a root_q"
        assert abs(float(ex[2])) == pytest.approx(1.0), "fast row lost its z target"
    for ex in full:
        assert ex[1].sum() == pytest.approx(1.0, abs=1e-4)
    frac_full = len(full) / len(examples)
    assert 0.3 < frac_full < 0.7, (
        f"full-search fraction {frac_full:.2f} far from p_full=0.5 "
        f"({len(full)}/{len(examples)})"
    )
