"""
Dual-head neural network for AlphaZero MCTS.

AlphaZero-style residual backbone with:
- Policy head: 1225 logits (one per possible move in the game)
- Value head: scalar in [-1, 1] (board evaluation)

Architecture:
    Input (4 channels) → initial conv → N residual blocks → policy/value heads

The old design had a single dense layer policy_fc (32*49 → 1225) that
accounted for ~87% of total parameters with no useful spatial abstraction.
This version uses a 2-filter policy conv so the FC is only ~120k params.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.t7g import board_to_obs


class ResidualBlock(nn.Module):
    """
    Standard AlphaZero residual block.

    conv(3×3) → BN → ReLU → conv(3×3) → BN → skip → ReLU
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


class DualHeadNetwork(nn.Module):
    """
    Residual CNN with policy + value heads for MCTS guidance.

    Args:
        num_actions:  Size of the flat action space (default 1225).
        num_filters:  Convolutional filters throughout the backbone (default 128).
        num_blocks:   Number of residual blocks (default 6).
    """

    def __init__(self, num_actions: int = 1225, num_filters: int = 128, num_blocks: int = 6) -> None:
        super().__init__()

        # Input projection: 4 channels → num_filters
        self.input_conv = nn.Conv2d(4, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head: 2-filter conv → flatten → FC
        # 2 * 7 * 7 = 98 → num_actions  (~120k params vs 1.92M before)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 7 * 7, num_actions)

        # Value head: 1-filter conv → flatten → FC → FC → tanh
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 7 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: tensor of shape (batch, 7, 7, 4) or (batch, 4, 7, 7)

        Returns:
            policy_logits: (batch, num_actions)
            value:         (batch, 1)
        """
        # Handle both NHWC and NCHW input
        if obs.dim() == 4 and obs.shape[-1] == 4:
            x = obs.permute(0, 3, 1, 2).float()
        else:
            x = obs.float()

        # Backbone
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.residual_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

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

        policy_logits, value = self.forward(obs_tensor)

        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        value_scalar = value.cpu().item()

        return policy_probs, value_scalar

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))
