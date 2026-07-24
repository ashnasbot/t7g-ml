"""
Dual-head neural network for AlphaZero MCTS.

AlphaZero-style residual backbone with:
- Policy head: 1225 logits (one per possible move in the game)
- Value head: scalar in [-1, 1] (board evaluation)

Architecture:
    Input (4 channels) -> initial conv -> N residual blocks -> policy/value heads

2-filter policy conv so the FC is only ~120k params.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import board_to_obs


class ResidualBlock(nn.Module):
    """
    Standard AlphaZero residual block.

    conv(3×3) -> BN -> ReLU -> conv(3×3) -> BN -> skip -> ReLU
    """

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
    """
    BN-free residual block, KataGo fixed-variance style (simplified).

    conv(3×3) -> ReLU -> conv(3×3) -> skip -> ×1/√2 -> ReLU.  He-init convs;
    the 1/√2 after the add resets activation variance to ~1 each block so the
    trunk's scale stays controlled without per-conv normalization.  A single
    BatchNorm at the trunk end (see DualHeadNetwork norm="fixup") absorbs any
    residual drift before the heads.  Fresh-training only - not checkpoint
    compatible with the BN blocks.
    """

    _INV_SQRT2 = 0.7071067811865476

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.conv2(F.relu(self.conv1(x)))
        return F.relu((x + branch) * self._INV_SQRT2)


class DualHeadNetwork(nn.Module):
    """
    Residual CNN with policy + value heads for MCTS guidance.

    Args:
        num_actions:  Size of the flat action space (default 1225).
        num_filters:  Convolutional filters throughout the backbone (default 128).
        num_blocks:   Number of residual blocks (default 6).
        wdl:          Value head is a 3-way win/draw/loss classifier trained
                      with cross-entropy instead of tanh+MSE.  Motivated by the
                      2026-07-10 calibration audit: the tanh head saturates
                      (38% of outputs |v|>0.9 while only ~65-83% correct there)
                      and MSE-through-tanh gives ~zero gradient exactly on
                      confidently-wrong predictions, so the head can never
                      unlearn its errors.  forward() still returns a scalar
                      value = P(win) - P(loss), so search code is unchanged.
        ownership:    Adds an auxiliary per-cell final-ownership head
                      (3 classes/cell: mine/opponent's/empty at game end) -
                      KataGo-style dense spatial signal that teaches the trunk
                      to count territory; 49 graded targets per position vs
                      the value head's single noisy bit per game.

    Defaults are the legacy configuration - checkpoints from before 2026-07-10
    load unchanged.  Use `DualHeadNetwork.infer_arch(state_dict)` to construct
    the right configuration for an arbitrary checkpoint.
    """

    def __init__(self, num_actions: int = 1225, num_filters: int = 128, num_blocks: int = 6,
                 wdl: bool = False, ownership: bool = False, norm: str = "bn") -> None:
        super().__init__()
        self.wdl = wdl
        self.ownership = ownership
        self.norm = norm

        # Input projection: 4 channels -> num_filters
        self.input_conv = nn.Conv2d(4, num_filters, kernel_size=3, padding=1, bias=False)
        if norm == "fixup":
            # BN-free trunk (KataGo fixed-variance style, simplified): He-init
            # everywhere, variance reset in each block, one BN at trunk end.
            nn.init.kaiming_normal_(self.input_conv.weight, nonlinearity='relu')
            self.input_bn = nn.Identity()
            block_cls = FixupResidualBlock
            self.trunk_bn = nn.BatchNorm2d(num_filters)
        else:
            self.input_bn = nn.BatchNorm2d(num_filters)
            block_cls = ResidualBlock
            self.trunk_bn = nn.Identity()

        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[block_cls(num_filters) for _ in range(num_blocks)]
        )

        # Policy head: 2-filter conv -> flatten -> FC
        # 2 * 7 * 7 = 98 -> num_actions  (~120k params vs 1.92M before)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.Identity() if norm == "fixup" else nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 7 * 7, num_actions)

        # Value head: 4-filter conv (spatial features) concatenated with
        # global mean+max pooling of the trunk (global features).  The pooled
        # trunk channels give the head direct access to board-wide quantities
        # (material difference, mobility) that pure convs struggle to count —
        # KataGo's global-pooling trick, which matters a lot for a
        # material-driven game like this one.
        self.value_conv = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.value_bn = nn.Identity() if norm == "fixup" else nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 7 * 7 + 2 * num_filters, 256)
        # wdl: 3 logits (win/draw/loss, side-to-move); legacy: tanh scalar
        self.value_fc2 = nn.Linear(256, 3 if wdl else 1)
        # Auxiliary margin head: predicts final material margin scaled to
        # [-1, 1] (margin / 49).  Dense low-noise signal from every game that
        # trains the same trunk the win/loss value head reads from.
        self.margin_fc = nn.Linear(256, 1)
        if ownership:
            # Auxiliary ownership head: per-cell 3-class logits over the
            # *final* board (0=mine, 1=opponent's, 2=empty), side-to-move
            # relative like the obs planes.  Read only via forward_full().
            self.own_conv = nn.Conv2d(num_filters, 3, kernel_size=1)

    @staticmethod
    def infer_arch(state_dict: dict) -> dict:
        """Infer constructor kwargs (wdl/ownership/norm) from a checkpoint's shapes."""
        v_out = state_dict["value_fc2.weight"].shape[0]
        return {
            "wdl": v_out == 3,
            "ownership": any(k.startswith("own_conv.") for k in state_dict),
            "norm": "fixup" if "trunk_bn.weight" in state_dict else "bn",
        }

    def forward(self, obs: torch.Tensor, full: bool = False):
        """
        Forward pass.

        Args:
            obs: tensor of shape (batch, 7, 7, 4) or (batch, 4, 7, 7)

        Returns:
            policy_logits: (batch, num_actions)
            value:         (batch, 1) — predicted game outcome in [-1, 1]
                           (wdl head: P(win) - P(loss); legacy head: tanh)
            margin:        (batch, 1) — predicted final material margin / 49
        (full=True appends value_logits and ownership_logits — use
        forward_full() instead, which labels them.)
        """
        # Handle both NHWC and NCHW input. The permute from (N,7,7,4) to
        # (N,4,7,7) produces strides that are already channels_last-compatible
        # so the following .contiguous(...) is typically a no-op — the call
        # is there to explicitly mark memory_format so cuDNN picks NHWC
        # Tensor Core kernels when the model has been .to(channels_last).
        if obs.dim() == 4 and obs.shape[-1] == 4:
            x = obs.permute(0, 3, 1, 2).contiguous(
                memory_format=torch.channels_last
            ).float()
        else:
            x = obs.float()

        # Backbone
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.residual_blocks(x)
        x = self.trunk_bn(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head: spatial conv features + global mean/max pooled trunk
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        g_mean = x.mean(dim=(2, 3))
        g_max = x.amax(dim=(2, 3))
        v = torch.cat([v, g_mean, g_max], dim=1)
        v = F.relu(self.value_fc1(v))
        if self.wdl:
            value_logits = self.value_fc2(v)
            probs = F.softmax(value_logits, dim=-1)
            value = (probs[:, 0] - probs[:, 2]).unsqueeze(-1)  # P(win) - P(loss)
        else:
            value_logits = None
            value = torch.tanh(self.value_fc2(v))
        margin = torch.tanh(self.margin_fc(v))

        if full:
            ownership_logits = self.own_conv(x) if self.ownership else None
            return policy_logits, value, margin, value_logits, ownership_logits
        return policy_logits, value, margin

    def forward_full(self, obs: torch.Tensor) -> dict:
        """
        Training-time forward: everything forward() computes plus the raw
        head logits.  Keys: policy_logits, value, margin, value_logits
        (B,3 or None), ownership_logits (B,3,7,7 or None).

        Kept separate so forward()'s 3-tuple contract (search, eval workers,
        ONNX export) never changes shape with the architecture flags.
        """
        policy_logits, value, margin, value_logits, ownership_logits = (
            self.forward(obs, full=True)
        )
        return {
            "policy_logits": policy_logits,
            "value": value,
            "margin": margin,
            "value_logits": value_logits,
            "ownership_logits": ownership_logits,
        }

    @torch.no_grad()
    def predict(self, board: np.ndarray, turn: bool) -> tuple[np.ndarray, float]:
        """
        Single-state inference for MCTS.

        Args:
            board: 7x7x2 numpy bool array
            turn:  True=Blue, False=Green

        Returns:
            policy_probs: num_actions-element numpy array (softmax probabilities)
            value:        float in [-1, 1]
        """
        self.eval()
        obs = board_to_obs(board, turn)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(next(self.parameters()).device)

        policy_logits, value, _ = self.forward(obs_tensor)

        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        value_scalar = value.cpu().item()

        return policy_probs, value_scalar

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))
