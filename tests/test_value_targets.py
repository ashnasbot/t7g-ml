"""
Regression test: value targets yielded by self_play_game_pool must be ±1.

Root cause of the bug this guards against
------------------------------------------
PASS is encoded as action 1225, which is out of range for the 1225-element
result array (indices 0–1224).  When a player was forced to pass, all result
entries stayed zero.  The pool misread that as "game over" and closed the
game with a material fraction as the winner — producing small-magnitude value
targets (~0.39 abs-mean) instead of ±1.  The value head then learned to
predict zero (the mean of the distribution) and never moved.

The fix in self_play_game_pool: when action_probs is all-zero but
check_terminal returns False, flip the turn and restart the search instead
of yielding a material-fraction winner.
"""
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, ".")
from lib.dual_network import DualHeadNetwork
from lib.mcgs import MCGS
from lib.train_workers import self_play_game_pool

# Number of complete games to run.  25 gives enough statistical power without
# making the test slow (≈4–6 s with 8 sims/move on CPU).
N_GAMES = 25
N_SIMS  = 8


@pytest.fixture(scope="module")
def mcts():
    torch.manual_seed(0)
    net = DualHeadNetwork(num_actions=1225).to("cpu")
    net.eval()
    return MCGS(net, num_simulations=N_SIMS, c_puct=1.25, gumbel_k=4)


def test_value_targets_are_decisive(mcts):
    """
    Over 99% of value targets in self-play examples must be exactly ±1.0.

    Any non-±1 target indicates a game was terminated early with a material
    fraction rather than played to a true terminal position.
    """
    values: list[float] = []
    for game_examples, _winner, *_ in self_play_game_pool(
        mcts, pool_size=4, target_games=N_GAMES
    ):
        for example in game_examples:
            values.append(float(example[2]))  # index 2 = value_target

    assert len(values) > 0, "No training examples were generated"

    decisive = [v for v in values if abs(abs(v) - 1.0) < 1e-6]
    frac = len(decisive) / len(values)
    assert frac >= 0.99, (
        f"Only {frac:.1%} of value targets are ±1 ({len(decisive)}/{len(values)}). "
        f"Expected ≥99%.  Non-decisive targets suggest forced-pass positions are "
        f"being incorrectly ended with a material fraction."
    )
