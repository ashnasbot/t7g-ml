"""
Legacy pre-net2 dual-head network - INFERENCE ONLY.

74 checkpoints on disk are this architecture, including the all-time rating
champion (run_bpath/iter_0180) and most of the frontier in debug/eval_db.
Rating them (scripts/eval_db.py) and playing them (scripts/play_gui.py,
the Elo anchor pool) needs to build them; training them does not.

So this file is deliberately reachable only from the play path
(lib.device_utils.build_inference_network).  ``lib.net2.build_from_state_dict``
- what training uses - still knows only net2/net2c, so no flag, checkpoint or
resume can put a legacy net back into the optimizer.  Nothing here has a
training surface: no aux heads, no forward_full, no optimizer state.

Trimmed from the trainable version deleted 2026-08-01 (git show
dae63d2:lib/dual_network.py for the original):
  * the ownership and margin heads are training-only signal, so they are built
    nowhere and their tensors are dropped at load time;
  * forward() keeps the (policy_logits, value, margin) contract that search
    unpacks (lib/mcgs.py) but returns None for margin, which no caller reads.

Architecture: input conv -> N residual blocks -> policy head + value head,
with the value head reading global mean/max pooling of the trunk alongside its
own conv features.  Everything variable across the checkpoints on disk
(width, depth, BN vs fixup trunk, tanh vs WDL value head) is inferred from the
state dict by ``infer_arch`` - there is nothing to configure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_INV_SQRT2 = 0.7071067811865476


class ResidualBlock(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> skip -> ReLU."""

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class FixupResidualBlock(nn.Module):
    """BN-free block: conv -> ReLU -> conv -> skip -> x1/sqrt(2) -> ReLU.

    The 1/sqrt(2) after the add is what holds activation variance at ~1 without
    per-conv normalization; a single BN at the trunk end absorbs the drift.
    """

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.conv2(F.relu(self.conv1(x)))
        return F.relu((x + branch) * _INV_SQRT2)


class DualHeadNetwork(nn.Module):
    """Legacy residual CNN with policy + value heads.  Play only.

    Args come from ``infer_arch``; construct via ``from_state_dict``.
    """

    def __init__(self, num_actions: int = 1225, num_filters: int = 128,
                 num_blocks: int = 6, wdl: bool = False, norm: str = "bn",
                 value_gpool: bool = True) -> None:
        super().__init__()
        self.wdl = wdl
        self.value_gpool = value_gpool

        self.input_conv = nn.Conv2d(4, num_filters, kernel_size=3, padding=1, bias=False)
        if norm == "fixup":
            self.input_bn = nn.Identity()
            block_cls: type[nn.Module] = FixupResidualBlock
            self.trunk_bn: nn.Module = nn.BatchNorm2d(num_filters)
        else:
            self.input_bn = nn.BatchNorm2d(num_filters)
            block_cls = ResidualBlock
            self.trunk_bn = nn.Identity()

        self.residual_blocks = nn.Sequential(
            *[block_cls(num_filters) for _ in range(num_blocks)]
        )

        # Policy head: 2-filter conv -> flatten -> FC
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.Identity() if norm == "fixup" else nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 7 * 7, num_actions)

        # Value head: 4-filter conv features, plus (on all but the oldest
        # checkpoints) global mean/max pooling of the trunk, so the head can
        # read board-wide quantities (material, mobility) directly instead of
        # counting them through convs.
        self.value_conv = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.value_bn = nn.Identity() if norm == "fixup" else nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 7 * 7 + (2 * num_filters if value_gpool else 0), 256)
        self.value_fc2 = nn.Linear(256, 3 if wdl else 1)

    @staticmethod
    def infer_arch(state_dict: dict) -> dict:
        """Constructor kwargs for an arbitrary legacy checkpoint."""
        return {
            "num_filters": state_dict["input_conv.weight"].shape[0],
            "num_blocks": 1 + max(int(k.split(".")[1]) for k in state_dict
                                  if k.startswith("residual_blocks.")),
            "wdl": state_dict["value_fc2.weight"].shape[0] == 3,
            "norm": "fixup" if "trunk_bn.weight" in state_dict else "bn",
            # Pre-2026-05 value heads read only their own conv features.
            "value_gpool": state_dict["value_fc1.weight"].shape[1] > 4 * 7 * 7,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict, num_actions: int = 1225) -> "DualHeadNetwork":
        """Build and load, dropping the training-only aux heads.

        The drop is explicit rather than a strict=False load so that a genuine
        shape mismatch still raises instead of being silently tolerated.
        """
        net = cls(num_actions=num_actions, **cls.infer_arch(state_dict))
        weights = {k: v for k, v in state_dict.items()
                   if not k.startswith("own_conv") and not k.startswith("margin_fc")}
        net.load_state_dict(weights)
        return net

    def forward(self, obs: torch.Tensor, full: bool = False):
        """(policy_logits, value, margin) - margin is always None here.

        obs is (batch, 7, 7, 4) or (batch, 4, 7, 7); value is (batch, 1) in
        [-1, 1] (WDL head: P(win) - P(loss), legacy head: tanh).
        """
        if obs.dim() == 4 and obs.shape[-1] == 4:
            x = obs.permute(0, 3, 1, 2).contiguous(
                memory_format=torch.channels_last
            ).float()
        else:
            x = obs.float()

        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.residual_blocks(x)
        x = self.trunk_bn(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.reshape(p.size(0), -1))

        v = F.relu(self.value_bn(self.value_conv(x))).reshape(x.size(0), -1)
        if self.value_gpool:
            v = torch.cat([v, x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        v = F.relu(self.value_fc1(v))
        if self.wdl:
            probs = F.softmax(self.value_fc2(v), dim=-1)
            value = (probs[:, 0] - probs[:, 2]).unsqueeze(-1)
        else:
            value = torch.tanh(self.value_fc2(v))

        return policy_logits, value, None


def is_legacy_state_dict(state_dict: dict) -> bool:
    """True if these weights are a legacy dual-head net rather than a net2."""
    return any(k.startswith("residual_blocks.") for k in state_dict)
