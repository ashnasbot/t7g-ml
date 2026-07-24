"""
t7g-net2: KataGo-family network (docs/net_rewrite_brief.md, 2026-07-17).

Replaces the 2017-A0 plain ResNet (lib/dual_network.py DualHeadNetwork) as the
*training* architecture; the old class stays importable so historical
checkpoints remain playable in the eval pool.  ``build_from_state_dict``
dispatches between the two by checkpoint shape.

Design (see the brief for rationale and sources):
- Trunk: nested-bottleneck residual blocks (1x1 down to C/2 -> two inner
  residual 3x3 pairs -> 1x1 up), fixed-variance init (He + 1/sqrt(2) scale
  after every skip-add, same scheme as the validated FixupResidualBlock),
  BN-free except ONE BatchNorm at trunk end before the heads.
- Global-pooling bias in 2 blocks: mean+max pool of the bottleneck features
  between the inner pairs -> zero-init FC -> per-channel bias.  Deviation from
  KataGo (which pools a channel *subset* and biases the rest): we pool and
  bias all C/2 channels — keeps the nested structure's channel count uniform;
  zero-init keeps the fixed-variance property at init.
- Policy: attention head — per-square source/target vectors, logit(src->dst) =
  S_src . T_dst / sqrt(D) gathered onto the flat 1225 = 49x25 action space;
  out-of-bounds action slots are pinned to POLICY_MASK_VALUE (softmax zero;
  finite in fp16 — see lib/training.py).
- Aux soft policy head (shares S, own T): trained on target^(1/4) renormalized
  (KataGoMethods.md: T=4, ~8x loss weight); pruned at inference.
- Value: WDL 3-way CE only (no tanh path).  KataGo-shaped head: 1x1 conv ->
  mean+max gpool -> FC -> heads.  Margin (tanh, final material/49) and
  ownership (3-class/cell off the value conv) kept.
- Short-term value heads: 3 tanh scalars predicting lambda-averaged future
  MCTS root values, horizons ~6/16/40 plies (ST_LAMBDAS).  Present in the
  arch now; targets are plumbed with the first self-play run (loss weight 0
  until then).

forward() keeps the (policy_logits, value, margin) 3-tuple contract of
DualHeadNetwork — search, eval workers, and export code are unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import board_to_obs
from lib.training import (_ACT_SRC, _ACT_DST, _ACT_INB, POLICY_MASK_VALUE,  # noqa: F401
                          ST_HORIZONS, ST_LAMBDAS)

_INV_SQRT2 = 0.7071067811865476


class _InnerPair(nn.Module):
    """Inner residual pair of a nested-bottleneck block:
    conv3x3 -> ReLU -> conv3x3 -> skip -> x 1/sqrt(2) -> ReLU  (fixed-variance)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.conv2(F.relu(self.conv1(x)))
        return F.relu((x + branch) * _INV_SQRT2)


class NestedBottleneckBlock(nn.Module):
    """KataGo nbt block: 1x1 down to C/2 -> inner pair -> [gpool bias] ->
    inner pair -> 1x1 up to C -> outer skip -> x 1/sqrt(2) -> ReLU.

    gpool=True inserts the global-pooling bias between the inner pairs:
    mean+max over the board (2*C/2 features) -> zero-init FC -> per-channel
    bias.  This is where board-wide arithmetic (material, parity, mobility)
    enters mid-trunk instead of only at the value head.
    """

    def __init__(self, ch: int, gpool: bool = False) -> None:
        super().__init__()
        half = ch // 2
        self.down = nn.Conv2d(ch, half, kernel_size=1, bias=False)
        self.up = nn.Conv2d(half, ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.up.weight, nonlinearity='linear')
        self.pair1 = _InnerPair(half)
        self.pair2 = _InnerPair(half)
        self.gpool_fc = nn.Linear(2 * half, half) if gpool else None
        if self.gpool_fc is not None:
            nn.init.zeros_(self.gpool_fc.weight)
            nn.init.zeros_(self.gpool_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.down(x))
        h = self.pair1(h)
        if self.gpool_fc is not None:
            g = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
            h = h + self.gpool_fc(g).unsqueeze(-1).unsqueeze(-1)
        h = self.pair2(h)
        branch = self.up(h)
        return F.relu((x + branch) * _INV_SQRT2)


class Net2(nn.Module):
    """t7g-net2.  See module docstring.

    Args:
        num_actions:    must be 1225 (the attention head is built on the
                        49x25 from-to structure; kept as an arg for call-site
                        uniformity with DualHeadNetwork).
        channels:       trunk width C (bottleneck runs at C/2).
        num_blocks:     nested-bottleneck block count.
        gpool_blocks:   indices of blocks that get the gpool bias; None =
                        (num_blocks//4, num_blocks//2), spread through the
                        first half of the trunk at any depth ((1, 3) for the
                        default 6 blocks).
        att_dim:        source/target vector dimension of the policy head.
    """

    def __init__(self, num_actions: int = 1225, channels: int = 128,
                 num_blocks: int = 6, gpool_blocks: 'tuple | None' = None,
                 att_dim: int = 16, value_channels: int = 32,
                 value_hidden: int = 96) -> None:
        super().__init__()
        if gpool_blocks is None:
            gpool_blocks = (num_blocks // 4, num_blocks // 2)
        assert num_actions == 1225, "Net2 policy head is specific to the 49x25 action space"
        self.channels = channels
        self.att_dim = att_dim

        self.input_conv = nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.input_conv.weight, nonlinearity='relu')
        self.blocks = nn.Sequential(
            *[NestedBottleneckBlock(channels, gpool=(i in gpool_blocks))
              for i in range(num_blocks)]
        )
        self.trunk_bn = nn.BatchNorm2d(channels)

        # Attention policy: per-square source/target vectors; soft head shares
        # the source vectors and has its own targets.
        self.policy_src = nn.Conv2d(channels, att_dim, kernel_size=1)
        self.policy_dst = nn.Conv2d(channels, att_dim, kernel_size=1)
        self.policy_dst_soft = nn.Conv2d(channels, att_dim, kernel_size=1)

        # Flat gather tables for logit(src,dst) -> 1225, and the OOB pin.
        pair_idx = (_ACT_SRC * 49 + np.where(_ACT_INB, _ACT_DST, 0)).astype(np.int64)
        self.register_buffer("_pair_idx", torch.from_numpy(pair_idx), persistent=False)
        self.register_buffer("_oob", torch.from_numpy(~_ACT_INB), persistent=False)

        # Value head, KataGo-shaped: 1x1 conv -> mean+max gpool -> FC -> heads.
        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * value_channels, value_hidden)
        self.value_wdl = nn.Linear(value_hidden, 3)
        self.margin_fc = nn.Linear(value_hidden, 1)
        self.st_value_fc = nn.Linear(value_hidden, len(ST_LAMBDAS))
        self.own_conv = nn.Conv2d(value_channels, 3, kernel_size=1)

    @staticmethod
    def is_net2_state_dict(state_dict: dict) -> bool:
        return any(k.startswith("policy_src.") for k in state_dict)

    @staticmethod
    def infer_arch(state_dict: dict) -> dict:
        """Constructor kwargs from checkpoint shapes (sizes may change in A/Bs)."""
        channels = state_dict["input_conv.weight"].shape[0]
        num_blocks = 1 + max(int(k.split(".")[1]) for k in state_dict
                             if k.startswith("blocks."))
        gpool_blocks = tuple(sorted({int(k.split(".")[1]) for k in state_dict
                                     if ".gpool_fc." in k}))
        return {
            "channels": channels,
            "num_blocks": num_blocks,
            "gpool_blocks": gpool_blocks,
            "att_dim": state_dict["policy_src.weight"].shape[0],
            "value_channels": state_dict["value_conv.weight"].shape[0],
            "value_hidden": state_dict["value_fc1.weight"].shape[0],
        }

    def _policy_logits(self, x: torch.Tensor, dst_conv: nn.Conv2d) -> torch.Tensor:
        b = x.size(0)
        s = self.policy_src(x).reshape(b, self.att_dim, 49)
        t = dst_conv(x).reshape(b, self.att_dim, 49)
        allpairs = torch.einsum("bds,bdt->bst", s, t) / (self.att_dim ** 0.5)
        logits = allpairs.reshape(b, 49 * 49)[:, self._pair_idx]
        return logits.masked_fill(self._oob, POLICY_MASK_VALUE)

    def forward(self, obs: torch.Tensor, full: bool = False):
        """Same contract as DualHeadNetwork.forward:
        (policy_logits, value, margin); full=True appends value_logits,
        ownership_logits, soft_policy_logits, st_values."""
        if obs.dim() == 4 and obs.shape[-1] == 4:
            x = obs.permute(0, 3, 1, 2).contiguous(
                memory_format=torch.channels_last
            ).float()
        else:
            x = obs.float()

        x = F.relu(self.input_conv(x))
        x = self.blocks(x)
        x = self.trunk_bn(x)

        policy_logits = self._policy_logits(x, self.policy_dst)

        v_sp = F.relu(self.value_conv(x))
        v = torch.cat([v_sp.mean(dim=(2, 3)), v_sp.amax(dim=(2, 3))], dim=1)
        v = F.relu(self.value_fc1(v))
        value_logits = self.value_wdl(v)
        probs = F.softmax(value_logits, dim=-1)
        value = (probs[:, 0] - probs[:, 2]).unsqueeze(-1)  # P(win) - P(loss)
        margin = torch.tanh(self.margin_fc(v))

        if full:
            soft_policy_logits = self._policy_logits(x, self.policy_dst_soft)
            st_values = torch.tanh(self.st_value_fc(v))
            ownership_logits = self.own_conv(v_sp)
            return (policy_logits, value, margin, value_logits,
                    ownership_logits, soft_policy_logits, st_values)
        return policy_logits, value, margin

    def forward_full(self, obs: torch.Tensor) -> dict:
        (policy_logits, value, margin, value_logits,
         ownership_logits, soft_policy_logits, st_values) = self.forward(obs, full=True)
        return {
            "policy_logits": policy_logits,
            "value": value,
            "margin": margin,
            "value_logits": value_logits,
            "ownership_logits": ownership_logits,
            "soft_policy_logits": soft_policy_logits,
            "st_values": st_values,
        }

    @torch.no_grad()
    def predict(self, board: np.ndarray, turn: bool) -> tuple[np.ndarray, float]:
        """Single-state inference for MCTS — same contract as DualHeadNetwork."""
        self.eval()
        obs = board_to_obs(board, turn)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(next(self.parameters()).device)
        policy_logits, value, _ = self.forward(obs_tensor)
        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        return policy_probs, value.cpu().item()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))


def build_from_state_dict(state_dict: dict, num_actions: int = 1225) -> nn.Module:
    """Construct the right network class (Net2 or legacy DualHeadNetwork) for
    an arbitrary checkpoint and load its weights."""
    if Net2.is_net2_state_dict(state_dict):
        net = Net2(num_actions=num_actions, **Net2.infer_arch(state_dict))
    else:
        from lib.dual_network import DualHeadNetwork
        net = DualHeadNetwork(num_actions=num_actions,
                              **DualHeadNetwork.infer_arch(state_dict))
    net.load_state_dict(state_dict)
    return net
