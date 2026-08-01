"""Legacy dual-head loader tests (lib/dual_network.py).

The legacy net exists so that pre-net2 checkpoints - most of the frontier in
debug/eval_db, including the all-time champion - stay ratable and playable.
Two things must not drift: every variant on disk still builds from its own
state dict, and the legacy net stays out of the training path.
"""
import pytest
import torch

from lib.device_utils import build_inference_network
from lib.dual_network import DualHeadNetwork, is_legacy_state_dict
from lib.net2 import Net2, build_from_state_dict


def _legacy_sd(aux=True, **kwargs):
    """State dict of a legacy net, with the training-only aux heads faked in.

    Real checkpoints carry ownership/margin tensors; the loader has to drop
    them rather than choke on them.
    """
    sd = DualHeadNetwork(**kwargs).state_dict()
    if aux:
        nf = sd["input_conv.weight"].shape[0]
        sd["own_conv.weight"] = torch.zeros(3, nf, 1, 1)
        sd["own_conv.bias"] = torch.zeros(3)
        sd["margin_fc.weight"] = torch.zeros(1, 256)
        sd["margin_fc.bias"] = torch.zeros(1)
    return sd


@pytest.mark.parametrize("arch", [
    {},                                          # legacy tanh value head
    {"wdl": True},                               # 3-way WDL head
    {"norm": "fixup"},                           # BN-free trunk
    {"wdl": True, "norm": "fixup"},
    {"value_gpool": False},                      # pre-pooling value head
    {"num_filters": 64, "num_blocks": 3},        # width/depth are inferred
])
def test_every_variant_round_trips(arch):
    sd = _legacy_sd(**arch)
    assert is_legacy_state_dict(sd)
    net = build_inference_network(sd)
    assert isinstance(net, DualHeadNetwork)
    # infer_arch must recover what built it, not just something loadable.
    for key, want in arch.items():
        assert DualHeadNetwork.infer_arch(sd)[key] == want


def test_forward_contract_matches_search():
    """lib/mcgs.py unpacks exactly (policy_logits, value, _)."""
    net = build_inference_network(_legacy_sd(wdl=True)).eval()
    with torch.no_grad():
        policy_logits, value, margin = net(torch.rand(4, 7, 7, 4))
    assert policy_logits.shape == (4, 1225)
    assert value.shape == (4, 1)
    assert value.abs().max() <= 1.0
    assert margin is None          # aux head is dropped, not computed


def test_nchw_and_nhwc_agree():
    net = build_inference_network(_legacy_sd()).eval()
    obs = torch.rand(2, 7, 7, 4)
    with torch.no_grad():
        a, va, _ = net(obs)
        b, vb, _ = net(obs.permute(0, 3, 1, 2))
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(va, vb, atol=1e-6)


def test_aux_head_weights_are_dropped_not_ignored():
    """Dropping own_conv/margin_fc must not also mask a real mismatch."""
    sd = _legacy_sd()
    sd["value_fc1.weight"] = torch.zeros(256, 7)      # genuinely wrong shape
    with pytest.raises(RuntimeError):
        build_inference_network(sd)


def test_training_builder_refuses_legacy_checkpoints():
    """The invariant that keeps training clean: only net2/net2c are trainable,
    so no --arch, resume or anchor path can put a legacy net in the optimizer.
    """
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        build_from_state_dict(_legacy_sd(wdl=True))
    # ... while the play-side builder still dispatches net2 correctly.
    assert isinstance(build_inference_network(Net2().state_dict()), Net2)
