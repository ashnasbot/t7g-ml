"""t7g-net2c architecture tests (lib/net2c.py).

net2c drops the margin and soft-policy heads and rebranches ownership off the
trunk.  These guard the two things that can silently break: the 3-tuple forward
contract now carries margin=None, and the loss path must tolerate that rather
than crash or quietly train a phantom head.
"""
import numpy as np
import pytest
import torch

from lib.net2 import Net2, build_from_state_dict
from lib.net2c import Net2C
from lib.training import ST_LAMBDAS, train_network

from tests.test_net2 import _rand_obs


def test_forward_contract_margin_is_none():
    net = Net2C()
    obs = torch.from_numpy(_rand_obs(3))
    pol, val, marg = net(obs)
    assert pol.shape == (3, 1225)
    assert val.shape == (3, 1)
    assert marg is None, "net2c has no margin head"

    out = net.forward_full(obs)
    assert out["margin"] is None
    assert out["soft_policy_logits"] is None
    assert out["ownership_logits"].shape == (3, 3, 7, 7)
    assert out["st_values"] is None, "net2c has no short-term value heads"
    assert out["value_logits"].shape == (3, 3)


def test_pruned_heads_absent_from_state_dict():
    keys = set(Net2C().state_dict())
    assert not any(k.startswith("margin_fc") for k in keys)
    assert not any(k.startswith("policy_dst_soft") for k in keys)
    assert not any(k.startswith("st_value_fc") for k in keys)
    # ownership is its own branch off the trunk, not a read of the value conv
    assert "own_conv1.weight" in keys
    assert Net2C().own_conv1.weight.shape[1] == Net2C().channels


def test_ownership_head_is_materially_wider_than_net2():
    own2 = sum(p.numel() for k, p in Net2().named_parameters() if k.startswith("own_"))
    ownc = sum(p.numel() for k, p in Net2C().named_parameters() if k.startswith("own_"))
    assert ownc > 20 * own2, (ownc, own2)


def test_checkpoint_roundtrip_and_dispatch(tmp_path):
    net = Net2C(channels=32, num_blocks=2, own_channels=16)
    p = tmp_path / "net2c.pt"
    net.save(str(p))
    sd = torch.load(str(p), weights_only=True)
    rebuilt = build_from_state_dict(sd)
    assert isinstance(rebuilt, Net2C)
    assert rebuilt.own_conv1.weight.shape[0] == 16
    obs = torch.from_numpy(_rand_obs(2))
    net.eval(); rebuilt.eval()
    with torch.no_grad():
        assert torch.allclose(net(obs)[0], rebuilt(obs)[0], atol=1e-6)


def test_net2_still_dispatches_to_net2():
    assert isinstance(build_from_state_dict(Net2().state_dict()), Net2)


def _buffer(n=32, seed=0):
    """Replay-buffer tuples in the layout train_mcts.py:275 actually stores:
    (obs, policy, value, margin, ownership, root_q, q_weight, st_targets)."""
    rng = np.random.default_rng(seed)
    obs = _rand_obs(n, seed)
    out = []
    for i in range(n):
        pol = rng.random(1225).astype(np.float32)
        pol /= pol.sum()
        own = rng.integers(0, 3, size=(7, 7)).astype(np.int8)
        st = (rng.random(len(ST_LAMBDAS)).astype(np.float32) * 2 - 1)
        out.append((obs[i], pol, float(rng.choice([-1.0, 1.0])), 0.3, own,
                    0.2, 0.5, st))
    return out


def test_train_step_runs_without_margin_head():
    """A margin_coef > 0 must not crash a net that has no margin head."""
    net = Net2C(channels=32, num_blocks=2, own_channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    res = train_network(net, _buffer(), opt, batch_size=16, epochs=1,
                        margin_coef=0.4, ownership_coef=0.15, st_value_coef=0.25)
    assert res["margin_loss"] == 0.0, "phantom margin head trained"
    assert np.isfinite(res["total_loss"])
    assert res["ownership_loss"] > 0.0


def test_gradients_reach_the_ownership_branch():
    net = Net2C(channels=32, num_blocks=2, own_channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_network(net, _buffer(), opt, batch_size=16, epochs=1,
                  ownership_coef=1.0, value_coef=0.0)
    assert net.own_conv1.weight.grad is not None
    assert float(net.own_conv1.weight.grad.abs().sum()) > 0.0


# --- lambda-return main value target -----------------------------------------

def test_lambda_return_matches_offline_recursion():
    """The online lambda-return must equal the offline one debug/target_ess.py
    measured, or the run is not testing what the measurement justified."""
    from lib.train_workers import _slot_result

    rng = np.random.default_rng(0)
    n, lam, winner = 12, 0.9375, 1.0

    class _S:
        pass
    slot = _S()
    qs = (rng.random(n) * 2 - 1).astype(np.float32)
    turns = [bool(i % 2 == 0) for i in range(n)]
    slot.examples = [
        (np.zeros((7, 7, 4), np.float32), np.zeros(1225, np.float32), turns[i],
         None, float(qs[i]), i, True)
        for i in range(n)
    ]
    slot.board = np.zeros((7, 7, 2), bool)
    slot.board[0, 0, 1] = True
    slot.move_count, slot.legal_move_counts = n, []
    slot.game_start = 0.0

    ex, *_ = _slot_result(slot, winner, value_lambda=lam, value_q_weight=0.9)

    # Reference: backward recursion in a fixed (blue) frame, seeded with the
    # terminal outcome -- exactly debug/target_ess.py's _lambda_return.
    acc = winner
    ref = np.empty(n, dtype=np.float64)
    for j in range(n - 1, -1, -1):
        q_blue = qs[j] if turns[j] else -qs[j]
        acc = (1.0 - lam) * q_blue + lam * acc
        ref[j] = acc

    for i in range(n):
        want = ref[i] if turns[i] else -ref[i]
        # _slot_result yields the 10-tuple: root_q at 7, q_weight at 8.
        # (The 8-tuple the replay buffer stores drops ex_board/turn -> 5 and 6.)
        assert abs(ex[i][7] - want) < 1e-5, (i, ex[i][7], want)
        assert ex[i][8] == 0.9, "q_weight must be the constant, not the ramp"


def test_lambda_none_is_pure_terminal_target():
    """value_lambda=None: pure z target -- root_q rides along at weight 0."""
    from lib.train_workers import _slot_result

    class _S:
        pass
    slot = _S()
    slot.examples = [
        (np.zeros((7, 7, 4), np.float32), np.ones(1225, np.float32) / 1225, True,
         None, 0.5, 0, True)
    ]
    slot.board = np.zeros((7, 7, 2), bool)
    slot.board[0, 0, 1] = True
    slot.move_count, slot.legal_move_counts = 1, []
    slot.game_start = 0.0
    ex, *_ = _slot_result(slot, 1.0)
    assert ex[0][7] == 0.5, "root_q slot must still carry the 1-step root Q"
    assert ex[0][8] == 0.0, "q_weight must be 0 without a lambda-return"


def test_q_weight_ramp_shape():
    """The ramp must start low, end at target, and stay there on a resume."""
    import importlib.util, pathlib, sys as _sys
    spec = importlib.util.spec_from_file_location(
        "_tm", pathlib.Path(__file__).parent.parent / "scripts" / "train_mcts.py")
    tm = importlib.util.module_from_spec(spec)
    _sys.modules["_tm"] = tm
    spec.loader.exec_module(tm)

    f, start, end = tm._value_q_weight, tm.VALUE_Q_WEIGHT_START, tm.VALUE_Q_WEIGHT
    n = tm.VALUE_Q_WEIGHT_RAMP_ITERS
    assert f(1) == start, "iteration 1 must not trust an untrained net's Q"
    assert abs(f(n + 1) - end) < 1e-9
    assert abs(f(10 * n) - end) < 1e-9, "must clamp, not keep climbing"
    vals = [f(i) for i in range(1, n + 2)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), "ramp must be monotone"
    assert start < f(n // 2) < end


# --- phase output buckets -------------------------------------------------
# See the PHASE OUTPUT BUCKETS section of lib/net2c.py for what these buy and
# why value/own default to 4/8.  These tests pin the CONTRACT, not the gain.

def _obs_with_occupancy(occ, batch=2, seed=0):
    """A batch of observations with exactly `occ` stones on the board."""
    g = torch.Generator().manual_seed(seed)
    o = torch.zeros(batch, 7, 7, 4)
    o[..., 2] = 1.0
    flat = o.reshape(batch, 49, 4)
    for i in range(batch):
        idx = torch.randperm(49, generator=g)[:occ]
        flat[i, idx[:occ // 2], 0] = 1.0
        flat[i, idx[occ // 2:], 1] = 1.0
    return o


def test_bucketing_keeps_output_shapes():
    """Bucketing must be invisible to callers: same shapes, whatever K."""
    from lib.net2c import Net2C
    ob = _obs_with_occupancy(20)
    for vb, obk in ((1, 1), (4, 8), (2, 3)):
        net = Net2C(channels=32, num_blocks=2, own_channels=16,
                    value_buckets=vb, own_buckets=obk).eval()
        pl, val, mg = net(ob)
        assert pl.shape == (2, 1225) and val.shape == (2, 1) and mg is None
        out = net.forward_full(ob)
        assert out["value_logits"].shape == (2, 3)
        assert out["ownership_logits"].shape == (2, 3, 7, 7)


def test_bucket_router_spans_the_board():
    """Occupancy must actually select different buckets across a game."""
    from lib.net2c import Net2C
    net = Net2C(channels=32, num_blocks=2, own_channels=16).eval()
    seen_v, seen_o = set(), set()
    for occ in range(4, 49):
        occv = torch.tensor(float(occ))
        seen_v.add(int(torch.bucketize(occv, net._v_bounds)))
        seen_o.add(int(torch.bucketize(occv, net._o_bounds)))
    assert seen_v == set(range(net.value_buckets)), "value router must use every bucket"
    assert seen_o == set(range(net.own_buckets)), "ownership router must use every bucket"


def test_bucket_pick_matches_manual_indexing():
    """_pick's gather must equal per-row slicing, for both readout ranks."""
    from lib.net2c import Net2C
    torch.manual_seed(0)
    B, K = 6, 4
    bk = torch.randint(0, K, (B,))
    lg = torch.randn(B, K * 3)
    assert torch.allclose(
        Net2C._pick(lg, bk, K),
        torch.stack([lg[i].reshape(K, 3)[bk[i]] for i in range(B)]))
    lg2 = torch.randn(B, K * 3, 7, 7)
    assert torch.allclose(
        Net2C._pick(lg2, bk, K),
        torch.stack([lg2[i].reshape(K, 3, 7, 7)[bk[i]] for i in range(B)]))


def test_infer_arch_reads_buckets_and_legacy_defaults_to_one():
    """A pre-bucketing checkpoint must read back as K=1, not as the new default."""
    from lib.net2c import Net2C
    from lib.net2 import build_from_state_dict
    kw = dict(channels=32, num_blocks=2, own_channels=16)

    net = Net2C(**kw, value_buckets=4, own_buckets=8).eval()
    r = build_from_state_dict(net.state_dict()).eval()
    assert (r.value_buckets, r.own_buckets) == (4, 8)
    ob = _obs_with_occupancy(20)
    assert torch.allclose(r.forward_full(ob)["value_logits"],
                          net.forward_full(ob)["value_logits"])

    legacy = Net2C(**kw, value_buckets=1, own_buckets=1)
    assert Net2C.infer_arch(legacy.state_dict())["value_buckets"] == 1
    assert Net2C.infer_arch(legacy.state_dict())["own_buckets"] == 1


def test_bucket_bounds_are_not_checkpoint_state():
    """Bounds derive from K, which infer_arch reads from the readout shapes;
    persisting them would let the two drift apart."""
    from lib.net2c import Net2C
    keys = set(Net2C(channels=32, num_blocks=2, own_channels=16).state_dict())
    assert not any("_bounds" in k for k in keys)


def test_unknown_bucket_count_is_rejected():
    """No silent fallback: a K with no measured boundary table must raise."""
    from lib.net2c import Net2C
    with pytest.raises(ValueError, match="value_buckets=5"):
        Net2C(channels=32, num_blocks=2, own_channels=16, value_buckets=5)
    with pytest.raises(ValueError, match="own_buckets=7"):
        Net2C(channels=32, num_blocks=2, own_channels=16, own_buckets=7)
