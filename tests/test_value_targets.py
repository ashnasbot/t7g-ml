"""
Regression test: value targets yielded by self_play_game_pool must be ±1.

Root cause of the bug this guards against
------------------------------------------
PASS is encoded as action 1225, which is out of range for the 1225-element
result array (indices 0–1224).  When a player was forced to pass, all result
entries stayed zero.  The pool misread that as "game over" and closed the
game with a material fraction as the winner - producing small-magnitude value
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


# ---------------------------------------------------------------------------
# Gated root-Q value blending (VALUE_BLEND_ALPHA < 1)
# ---------------------------------------------------------------------------

def test_q_blend_weight_gating():
    """The gated Q weight must be 0 when disabled/ungated, (1-alpha) at full gate."""
    from lib.train_workers import _q_blend_weight

    onehot  = np.zeros(1225); onehot[0] = 1.0
    uniform = np.ones(1225) / 1225

    # alpha=1.0 disables blending regardless of gates.
    assert _q_blend_weight(50, onehot, blend_alpha=1.0, temp_moves=16) == 0.0
    # Before the temp threshold the phase gate is closed.
    assert _q_blend_weight(10, onehot, blend_alpha=0.5, temp_moves=16) == 0.0
    # Uniform visit distribution -> zero concentration -> gate closed.
    assert _q_blend_weight(50, uniform, blend_alpha=0.5, temp_moves=16) == pytest.approx(0.0)
    # Past the ramp with a one-hot policy both gates are fully open.
    assert _q_blend_weight(50, onehot, blend_alpha=0.5, temp_moves=16) == pytest.approx(0.5)
    assert _q_blend_weight(50, onehot, blend_alpha=0.7, temp_moves=16) == pytest.approx(0.3)


def _train_once(examples):
    """One deterministic training pass over *examples* on a fresh WDL net."""
    from lib.training import train_network, _IterBuffer

    torch.manual_seed(0)
    np.random.seed(0)
    net = DualHeadNetwork(num_actions=1225, wdl=True, ownership=True).to("cpu")
    opt = torch.optim.Adam(net.parameters())
    buf = _IterBuffer(8)
    buf.append_batch(examples)
    return train_network(net, buf, opt, batch_size=len(examples), epochs=1,
                         value_coef=1.0, margin_coef=0.4, ownership_coef=0.15)


def _mk_examples(n=16, q_fields=None):
    """Build n buffer examples; q_fields=(root_q, q_weight) appends the blend fields."""
    from lib.t7g import new_board, board_to_obs

    obs = board_to_obs(new_board(), True)
    pol = np.zeros(1225, dtype=np.float32)
    pol[0] = 1.0
    own = np.full((7, 7), 2, dtype=np.int8)
    out = []
    for i in range(n):
        z = 1.0 if i % 2 == 0 else -1.0
        ex = (obs, pol, z, 0.1, own)
        if q_fields is not None:
            ex = ex + q_fields
        out.append(ex)
    return out


def test_wdl_soft_blend_loss():
    """
    q_weight=0 must be loss-identical to the legacy hard-label path (whether
    or not the fields are present), q_weight>0 must change the value loss and
    nothing else, and sign_acc must always score against the pure outcome z.
    """
    legacy   = _train_once(_mk_examples())                       # 5-tuples
    w_zero   = _train_once(_mk_examples(q_fields=(0.5, 0.0)))    # fields, gate closed
    w_half   = _train_once(_mk_examples(q_fields=(0.5, 0.5)))    # gate open

    assert w_zero["value_loss"] == pytest.approx(legacy["value_loss"], abs=1e-7)
    assert w_half["value_loss"] != pytest.approx(legacy["value_loss"], abs=1e-6)
    # The blend only touches the value target: same net/seed sees the same
    # forward pass, so policy loss and outcome-based sign_acc are unchanged.
    assert w_half["policy_loss"] == pytest.approx(legacy["policy_loss"], abs=1e-7)
    assert w_half["sign_acc"] == pytest.approx(legacy["sign_acc"], abs=1e-7)


def test_wdl_soft_blend_direction():
    """With w=1 and q=+1 the soft target is pure 'win': loss must be lower for
    a net pushed toward win-logits than the hard z=-1 'loss' label would give."""
    import torch.nn.functional as F

    logits = torch.tensor([[2.0, 0.0, -2.0]])  # net says: win
    hard_loss = F.cross_entropy(logits, torch.tensor([2]))          # z = loss
    soft = torch.tensor([[1.0, 0.0, 0.0]])                          # q=+1, w=1
    soft_loss = F.cross_entropy(logits, soft)
    assert soft_loss < hard_loss
