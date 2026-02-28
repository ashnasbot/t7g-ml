"""
Dual-head neural network for AlphaZero MCTS.

Single CNN backbone with:
- Policy head: 1225 logits (one per possible move in the game)
- Value head: scalar in [-1, 1] (board evaluation)

Architecture mirrors MicroscopeCNN from networks.py but with dual outputs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.mcts import board_to_obs


class DualHeadNetwork(nn.Module):
    """CNN with policy + value heads for MCTS guidance."""

    def __init__(self, num_actions=1225, backbone_filters=128):
        super().__init__()

        # Backbone (4 input channels: green, blue, turn, selected_piece)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, backbone_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(backbone_filters)

        # Residual connection
        self.residual = nn.Conv2d(32, backbone_filters, kernel_size=1)

        self.conv4 = nn.Conv2d(backbone_filters, backbone_filters, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(backbone_filters)

        # Policy head: conv -> flatten -> FC -> 1225 logits
        self.policy_conv = nn.Conv2d(backbone_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 7 * 7, num_actions)

        # Value head: conv -> pool -> FC -> tanh -> scalar
        self.value_conv = nn.Conv2d(backbone_filters, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 7 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: tensor of shape (batch, 7, 7, 4) or (batch, 4, 7, 7)

        Returns:
            policy_logits: (batch, 1225)
            value: (batch, 1)
        """
        # Handle both NHWC and NCHW input
        if obs.dim() == 4 and obs.shape[-1] == 4:
            x = obs.permute(0, 3, 1, 2).float()
        else:
            x = obs.float()

        # Backbone
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)) + self.residual(x1))
        features = F.relu(self.bn4(self.conv4(x3)))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(features)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(features)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(self, board, turn):
        """
        Single-state inference for MCTS.

        Args:
            board: 7x7x2 numpy bool array
            turn: True=Blue, False=Green

        Returns:
            policy_probs: 1225-element numpy array (softmax probabilities)
            value: float in [-1, 1]
        """
        self.eval()
        obs = board_to_obs(board, turn)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        obs_tensor = obs_tensor.to(next(self.parameters()).device)

        policy_logits, value = self.forward(obs_tensor)

        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        value_scalar = value.cpu().item()

        return policy_probs, value_scalar

    def save(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model weights."""
        self.load_state_dict(torch.load(path, weights_only=True))
