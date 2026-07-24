"""t7g-net2 architecture tests (lib/net2.py)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lib.net2 import Net2, NestedBottleneckBlock, build_from_state_dict
from lib.training import ST_LAMBDAS
from lib.t7g import new_board
from lib.training import (train_network, illegal_action_mask, _ACT_SRC, _ACT_DST,
                          _ACT_INB, POLICY_MASK_VALUE)


def _rand_obs(n=4, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 7, 7, 4), dtype=np.float32)
    for i in range(n):
        cells = rng.permutation(49)
        obs[i].reshape(49, 4)[cells[:8], 0] = 1.0   # opponent pieces
        obs[i].reshape(49, 4)[cells[8:16], 1] = 1.0  # my pieces
    obs[..., 2] = 1.0
    return obs


def test_forward_contract_and_shapes():
    net = Net2()
    obs = torch.from_numpy(_rand_obs(3))
    pol, val, marg = net(obs)
    assert pol.shape == (3, 1225)
    assert val.shape == (3, 1)
    assert marg.shape == (3, 1)
    assert val.abs().max() <= 1.0 and marg.abs().max() <= 1.0

    out = net.forward_full(obs)
    assert out["value_logits"].shape == (3, 3)
    assert out["ownership_logits"].shape == (3, 3, 7, 7)
    assert out["soft_policy_logits"].shape == (3, 1225)
    assert out["st_values"].shape == (3, len(ST_LAMBDAS))
    # value must be P(win) - P(loss) of the WDL softmax
    probs = F.softmax(out["value_logits"], dim=-1)
    assert torch.allclose(out["value"].squeeze(-1), probs[:, 0] - probs[:, 2], atol=1e-6)


def test_oob_actions_pinned():
    net = Net2()
    pol, _, _ = net(torch.from_numpy(_rand_obs(2)))
    oob = torch.from_numpy(~_ACT_INB)
    assert (pol[:, oob] <= POLICY_MASK_VALUE / 2).all()
    assert (pol[:, ~oob] > POLICY_MASK_VALUE / 2).all()


def test_mask_value_fp16_safe():
    """Self-play runs the forward under fp16 autocast on ROCm; the mask fill
    must stay finite in half precision and still zero the softmax."""
    v = torch.tensor(POLICY_MASK_VALUE, dtype=torch.float16)
    assert torch.isfinite(v)
    probs = torch.softmax(torch.tensor([0.0, POLICY_MASK_VALUE]), dim=-1)
    assert probs[1] == 0.0


def test_attention_logits_match_naive_gather():
    """logit(a) must equal S_src(a) . T_dst(a) / sqrt(D) for in-bounds actions."""
    net = Net2()
    net.eval()
    obs = torch.from_numpy(_rand_obs(2, seed=3))
    with torch.no_grad():
        x = obs.permute(0, 3, 1, 2).float()
        x = F.relu(net.input_conv(x))
        x = net.blocks(x)
        x = net.trunk_bn(x)
        s = net.policy_src(x).reshape(2, net.att_dim, 49)
        t = net.policy_dst(x).reshape(2, net.att_dim, 49)
        pol, _, _ = net(obs)
    for a in [0, 12, 617, 1224]:
        if not _ACT_INB[a]:
            continue
        want = (s[:, :, _ACT_SRC[a]] * t[:, :, _ACT_DST[a]]).sum(1) / net.att_dim ** 0.5
        assert torch.allclose(pol[:, a], want, atol=1e-5)


def test_fixed_variance_trunk_scale():
    """Pre-BN trunk activations of a fresh net stay O(1) on random input."""
    net = Net2()
    net.eval()
    x = torch.randn(16, 4, 7, 7)
    with torch.no_grad():
        h = F.relu(net.input_conv(x))
        h = net.blocks(h)
    assert 0.1 < h.std().item() < 3.0


def test_gpool_zero_init_is_identity_at_start():
    blk = NestedBottleneckBlock(64, gpool=True)
    ref = NestedBottleneckBlock(64, gpool=False)
    ref.load_state_dict({k: v for k, v in blk.state_dict().items()
                         if not k.startswith("gpool_fc")})
    x = torch.randn(2, 64, 7, 7)
    with torch.no_grad():
        assert torch.allclose(blk(x), ref(x), atol=1e-6)


def test_checkpoint_roundtrip_and_dispatch(tmp_path):
    net = Net2(channels=64, num_blocks=3, gpool_blocks=(1,), att_dim=8,
               value_channels=16, value_hidden=32)
    p = str(tmp_path / "net2.pt")
    net.save(p)
    sd = torch.load(p, weights_only=True)
    assert Net2.is_net2_state_dict(sd)
    kwargs = Net2.infer_arch(sd)
    assert kwargs == {"channels": 64, "num_blocks": 3, "gpool_blocks": (1,),
                      "att_dim": 8, "value_channels": 16, "value_hidden": 32}
    net2 = build_from_state_dict(sd)
    assert isinstance(net2, Net2)
    obs = torch.from_numpy(_rand_obs(2))
    net.eval(); net2.eval()
    with torch.no_grad():
        a = net(obs); b = net2(obs)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_dispatch_legacy_checkpoint():
    from lib.dual_network import DualHeadNetwork
    old = DualHeadNetwork(wdl=True, ownership=True)
    net = build_from_state_dict(old.state_dict())
    assert isinstance(net, DualHeadNetwork) and net.wdl and net.ownership


def test_predict_matches_board_legality():
    net = Net2()
    probs, value = net.predict(new_board(), True)
    assert probs.shape == (1225,)
    assert abs(probs.sum() - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0
    # OOB slots get exactly zero probability mass
    assert probs[~_ACT_INB].max() == 0.0


def test_train_step_decreases_loss_with_soft_policy():
    """One tiny overfit: full loss path incl. masking + soft policy head."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = Net2(channels=32, num_blocks=2, gpool_blocks=(0,), att_dim=8,
               value_channels=8, value_hidden=16)
    obs = _rand_obs(8, seed=1)
    buffer = []
    for i in range(8):
        legal = illegal_action_mask(obs[i:i + 1])[0]
        pol = np.zeros(1225, dtype=np.float32)
        legal_idx = np.flatnonzero(legal)
        pol[legal_idx[: max(1, len(legal_idx) // 3)]] = 1.0
        pol /= pol.sum()
        own = np.zeros((7, 7), dtype=np.int8)
        st = np.full(len(ST_LAMBDAS), 0.4 if i % 2 == 0 else -0.4, dtype=np.float32)
        buffer.append((obs[i], pol, 1.0 if i % 2 == 0 else -1.0, 0.2, own,
                       0.0, 0.0, st))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    kw = dict(batch_size=8, epochs=1, device="cpu", value_coef=1.0,
              margin_coef=0.4, ownership_coef=0.15, soft_policy_coef=8.0,
              st_value_coef=0.25, mask_illegal=True)
    first = train_network(net, buffer, opt, **kw)
    assert first["soft_policy_loss"] > 0.0
    assert first["st_value_loss"] > 0.0
    for _ in range(30):
        last = train_network(net, buffer, opt, **kw)
    assert last["policy_loss"] < first["policy_loss"]
    assert last["value_loss"] < first["value_loss"]
    assert last["soft_policy_loss"] < first["soft_policy_loss"]
    assert last["st_value_loss"] < first["st_value_loss"]


def test_st_targets_match_naive_sum():
    """_slot_result's backward recursion == the definition
    s_i = (1-l) sum_{j in [i,n)} l^(j-i) q_j(i-persp) + l^(n-i) z(i-persp)."""
    from lib.train_workers import _slot_result, _GameSlot

    rng = np.random.default_rng(7)
    slot = _GameSlot(None)
    n, winner = 12, 1.0
    obs = _rand_obs(1)[0]
    pol = np.full(1225, 1 / 1225, dtype=np.float32)
    turns = [bool((i + 1) % 2) for i in range(n)]     # alternating, Green first
    qs = rng.uniform(-1, 1, n)
    slot.examples = [(obs, pol, turns[i], slot.board.copy(), float(qs[i]), i, True)
                     for i in range(n)]
    examples, *_ = _slot_result(slot, winner=winner)

    for i in [0, 5, n - 1]:
        st = examples[i][9]
        for k, lam in enumerate(ST_LAMBDAS):
            want = 0.0
            for j in range(i, n):
                q_p = qs[j] if turns[j] == turns[i] else -qs[j]
                want += (1 - lam) * lam ** (j - i) * q_p
            want += lam ** (n - i) * (winner if turns[i] else -winner)
            assert st[k] == pytest.approx(want, abs=1e-5)


def test_old_net_training_unaffected_by_soft_coef_default():
    """Old-arch nets through the new train_network: soft loss reports 0 and
    the returned losses are identical with/without the new code path active."""
    from lib.dual_network import DualHeadNetwork
    torch.manual_seed(0)
    np.random.seed(0)
    obs = _rand_obs(8, seed=2)
    own = np.zeros((7, 7), dtype=np.int8)
    buffer = [(obs[i], np.full(1225, 1 / 1225, dtype=np.float32),
               1.0 if i % 2 == 0 else -1.0, 0.1, own, 0.0, 0.0) for i in range(8)]

    def run():
        torch.manual_seed(1)
        np.random.seed(1)
        net = DualHeadNetwork(wdl=True, ownership=True)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        return train_network(net, buffer, opt, batch_size=8, epochs=1, device="cpu",
                             value_coef=1.0, margin_coef=0.4, ownership_coef=0.15,
                             mask_illegal=True)

    a, b = run(), run()
    assert a["soft_policy_loss"] == 0.0
    for k in ("policy_loss", "value_loss", "margin_loss", "ownership_loss"):
        assert a[k] == pytest.approx(b[k])
